import json
import logging
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import sys

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)

from VLA_finetune.open_clip import ClipLoss
from .distributed import is_master
from .precision import get_autocast
from VLA_finetune.open_clip.loss import gather_features


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model


def train_one_epoch(model, data, epoch, optimizer, scaler, scheduler, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)

    model.train()
    loss = ClipLoss(
        local_loss=args.local_loss,
        gather_with_grad=args.gather_with_grad,
        cache_labels=True,
        rank=args.rank,
        world_size=args.world_size)

    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    loss_m = AverageMeter()
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    total_time=0

    for i, batch in enumerate(dataloader):

        step = num_batches_per_epoch * epoch + i
        scheduler(step)
        images, texts = batch

        images = images.to(device=device, non_blocking=True)
        texts = texts.to(device=device, non_blocking=True)

        start_t = time.time()

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()


        image_features_list = []
        text_features_list = []

        # Subject, Object, Action
        channels = 3

        with autocast():
            for k in range(channels):
                image_channel = images[:, k, :, :, :]
                text_channel = texts[:, k, :]
                
                if args.flava:
                    image_channel_features = model.module.get_image_features(pixel_values=image_channel)[:, 0]
                    text_channel_features = model.module.get_text_features(input_ids=text_channel)[:, 0]
                    logit_scale = math.exp(model.module.logit_scale)
                else:
                    image_channel_features, text_channel_features, logit_scale = model(image_channel, text_channel)
        
                image_features_list.append(image_channel_features)
                text_features_list.append(text_channel_features)
            
            # Stack the features along dimension 1 (the channel dimension)
            image_features = torch.stack(image_features_list, dim=1)
            text_features = torch.stack(text_features_list, dim=1)

            total_loss = loss(image_features, text_features, logit_scale)

        if scaler is not None:
            scaler.scale(total_loss).backward()
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                if args.norm_gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.norm_gradient_clip, norm_type=2.0)
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                if args.norm_gradient_clip is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.norm_gradient_clip, norm_type=2.0)
                scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            if args.norm_gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.norm_gradient_clip, norm_type=2.0)
            optimizer.step()

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        if not args.flava:
            with torch.no_grad():
                unwrap_model(model).logit_scale.clamp_(0, math.log(100))


        total_time = total_time + time.time()-start_t

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i + 1


        if is_master(args) and (i % 100 == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            loss_m.update(total_loss.item(), batch_size)

            if args.flava:
                logit_scale_scalar = logit_scale
            else:
                logit_scale_scalar = logit_scale.item()
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Loss: {loss_m.val:#.5g} ({loss_m.avg:#.4g}) "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {args.batch_size*args.world_size / batch_time_m.val:#g}/s "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f}"
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "loss": loss_m.val,
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_scond": args.batch_size*args.world_size / batch_time_m.val,
                "scale":  logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"]
            }
            for name, val in log_data.items():
                name = "train/" + name
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, step)

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
    # end for



def evaluate(model, data, epoch, args, tb_writer=None):
    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    model.eval()


    autocast = get_autocast(args.precision)


    if 'val' in data and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        dataloader = data['val'].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        # FIXME this does not scale past small eval datasets
        # all_image_features @ all_text_features will blow up memory and compute very quickly
        cumulative_loss = 0.0
        all_image_features, all_text_features = [], []
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                images, texts = batch

                images = images.to(device=device, non_blocking=True)
                texts = texts.to(device=device, non_blocking=True)

                channels = 3
                image_features_list = []
                text_features_list = []

                with autocast():
                    for j in range(channels):
                        image_channel = images[:, j, :, :, :]
                        text_channel = texts[:, j, :]
                        if args.flava:
                            image_channel_features = model.get_image_features(pixel_values=image_channel)[:, 0]
                            text_channel_features = model.get_text_features(input_ids=text_channel)[:, 0]
                            logit_scale = math.exp(model.logit_scale)
                        else:
                            image_channel_features, text_channel_features, logit_scale = model(image_channel, text_channel)
                        image_features_list.append(image_channel_features)
                        text_features_list.append(text_channel_features)

                    # Stack the features along dimension 1 (the channel dimension)
                    image_features = torch.stack(image_features_list, dim=1)
                    text_features = torch.stack(text_features_list, dim=1)

                    # features are accumulated in CPU tensors, otherwise GPU memory exhausted quickly
                    # however, system RAM is easily exceeded and compute time becomes problematic
                    all_image_features.append(image_features.cpu())
                    all_text_features.append(text_features.cpu())
                    if not args.flava:
                        logit_scale = logit_scale.mean()

                    logits_per_image_list = []
                    logits_per_text_list = []
                    for j in range(channels):
                        image_channel = image_features[:, j, :]
                        text_channel = text_features[:, j, :]
                        logits_per_image_channel = logit_scale * image_channel @ text_channel.T
                        logits_per_text_channel = logits_per_image_channel.T

                        logits_per_image_list.append(logits_per_image_channel)
                        logits_per_text_list.append(logits_per_text_channel)

                    logits_per_image = sum(logits_per_image_list) / channels
                    logits_per_text = sum(logits_per_text_list) / channels

                    batch_size = images.shape[0]
                    labels = torch.arange(batch_size, device=device).long()
                    total_loss = (
                        F.cross_entropy(logits_per_image, labels) +
                        F.cross_entropy(logits_per_text, labels)
                    ) / 2

                cumulative_loss += total_loss * batch_size
                num_samples += batch_size
                if is_master(args) and (i % 100) == 0:
                    logging.info(
                        f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t"
                        f"Loss: {cumulative_loss / num_samples:.6f}\t")

            val_metrics = get_metrics(
                image_features=torch.cat(all_image_features),
                text_features=torch.cat(all_text_features),
                logit_scale = logit_scale.cpu() if isinstance(logit_scale, torch.Tensor) else logit_scale
            )
            loss = cumulative_loss / num_samples
            metrics.update(
                {**val_metrics, "val_loss": loss.item(), "epoch": epoch, "num_samples": num_samples}
            )
            # metrics.update(
            #     {"val_loss": loss.item(), "epoch": epoch, "num_samples": num_samples}
            # )

    if not metrics:
        return metrics

    logging.info(
        f"Eval Epoch: {epoch} "
        + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )

    if args.save_logs:
        for name, val in metrics.items():
            if tb_writer is not None:
                tb_writer.add_scalar(f"val/{name}", val, epoch)

        with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

    return metrics


def get_metrics(image_features, text_features, logit_scale):
    metrics = {}

    channels = 3

    logits_per_image_list = []
    logits_per_text_list = []
    for j in range(channels):
        image_channel = image_features[:, j, :]
        text_channel = text_features[:, j, :]
        logits_per_image_channel = logit_scale * image_channel @ text_channel.T
        logits_per_text_channel = logits_per_image_channel.T

        logits_per_image_list.append(logits_per_image_channel)
        logits_per_text_list.append(logits_per_text_channel)
        
    logits_per_image = (sum(logits_per_image_list) / channels).detach().cpu()
    logits_per_text = (sum(logits_per_text_list) / channels).detach().cpu()


    logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
    ground_truth = torch.arange(len(text_features)).view(-1, 1)

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)

    return metrics
