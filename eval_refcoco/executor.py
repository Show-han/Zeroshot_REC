from typing import List, Dict, Union, Tuple

from PIL import Image, ImageDraw, ImageFilter
import spacy
import hashlib
import os

import torch
import torchvision
import torchvision.transforms as transforms
import clip
from transformers import BertTokenizer, RobertaTokenizerFast
import ruamel.yaml as yaml
from peft import get_peft_model
from peft import LoraConfig, TaskType
from interpreter import Box


import sys
import os
from transformers import FlavaFeatureExtractor, FlavaModel, BertTokenizer

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)


from VLA_finetune.open_clip import create_model_and_transforms


class Executor:
    def __init__(self, device: str = "cpu", box_representation_method: str = "crop", method_aggregator: str = "max", enlarge_boxes: int = 0, expand_position_embedding: bool = False, square_size: bool = False, blur_std_dev: int = 100, cache_path: str = None) -> None:
        IMPLEMENTED_METHODS = ["crop", "blur", "shade"]
        if any(m not in IMPLEMENTED_METHODS for m in box_representation_method.split(",")):
            raise NotImplementedError
        IMPLEMENTED_AGGREGATORS = ["max", "sum"]
        if method_aggregator not in IMPLEMENTED_AGGREGATORS:
            raise NotImplementedError
        self.box_representation_method = box_representation_method
        self.method_aggregator = method_aggregator
        self.enlarge_boxes = enlarge_boxes
        self.device = device
        self.expand_position_embedding = expand_position_embedding
        self.square_size = square_size
        self.blur_std_dev = blur_std_dev
        self.cache_path = cache_path

    def preprocess_image(self, image: Image) -> List[torch.Tensor]:
        return [preprocess(image) for preprocess in self.preprocesses]

    def preprocess_text(self, text: str) -> torch.Tensor:
        raise NotImplementedError

    def call_model(self, model: torch.nn.Module, images: torch.Tensor, text: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
        raise NotImplementedError

    def tensorize_inputs(self, caption: str, image: Image, boxes: Union[List[Box], List[Tuple[Box, Box]]], image_name: str = None) -> Tuple[List[torch.Tensor], torch.Tensor]:
        images = []
        for preprocess in self.preprocesses:
            images.append([])
        if self.cache_path is None or any([not os.path.exists(os.path.join(self.cache_path, model_name, image_name, method_name+".pt")) for model_name in self.model_names for method_name in self.box_representation_method.split(',')]):
            if "crop" in self.box_representation_method:
                if isinstance(boxes[0], Box):
                    for i in range(len(boxes)):
                        image_i = image.copy()
                        box = [
                            max(boxes[i].left-self.enlarge_boxes, 0),
                            max(boxes[i].top-self.enlarge_boxes, 0),
                            min(boxes[i].right+self.enlarge_boxes, image_i.width),
                            min(boxes[i].bottom+self.enlarge_boxes, image_i.height)
                        ]
                        image_i = image_i.crop(box)
                        preprocessed_images = self.preprocess_image(image_i)
                        for j, img in enumerate(preprocessed_images):
                            images[j].append(img.to(self.device))
                elif isinstance(boxes[0], tuple):
                    for i in range(len(boxes)):
                        box1, box2 = boxes[i]  # Unpack the tuple
                        image_i = image.copy()
                        # Compute the union of both boxes
                        union_box = box1.min_bounding(box2)
                        # Now create the bounding box for cropping with enlargement if necessary
                        box = [
                            max(union_box.left - self.enlarge_boxes, 0),
                            max(union_box.top - self.enlarge_boxes, 0),
                            min(union_box.right + self.enlarge_boxes, image_i.width),
                            min(union_box.bottom + self.enlarge_boxes, image_i.height)
                        ]
                        image_i = image_i.crop(box)
                        preprocessed_images = self.preprocess_image(image_i)
                        for j, img in enumerate(preprocessed_images):
                            images[j].append(img.to(self.device))
            if "blur" in self.box_representation_method:
                for i in range(len(boxes)):
                    image_i = image.copy()
                    mask = Image.new('L', image_i.size, 0)
                    draw = ImageDraw.Draw(mask)
                    
                    if isinstance(boxes[i], Box):
                        box = (
                            max(boxes[i].left - self.enlarge_boxes, 0),
                            max(boxes[i].top - self.enlarge_boxes, 0),
                            min(boxes[i].right + self.enlarge_boxes, image_i.width),
                            min(boxes[i].bottom + self.enlarge_boxes, image_i.height)
                        )
                        draw.rectangle([box[:2], box[2:]], fill=255)
                    elif isinstance(boxes[i], tuple):
                        box1, box2 = boxes[i]
                        for b in [box1, box2]:
                            box = (
                                max(b.left - self.enlarge_boxes, 0),
                                max(b.top - self.enlarge_boxes, 0),
                                min(b.right + self.enlarge_boxes, image_i.width),
                                min(b.bottom + self.enlarge_boxes, image_i.height)
                            )
                            draw.rectangle([box[:2], box[2:]], fill=255)

                    blurred = image_i.filter(ImageFilter.GaussianBlur(self.blur_std_dev))
                    blurred.paste(image_i, mask=mask)
                    preprocessed_images = self.preprocess_image(blurred)
                    for j, img in enumerate(preprocessed_images):
                        images[j].append(img.to(self.device))
            if "shade" in self.box_representation_method:
                for i in range(len(boxes)):
                    TINT_COLOR = (240, 0, 30)
                    image_i = image.copy().convert('RGBA')
                    overlay = Image.new('RGBA', image_i.size, TINT_COLOR+(0,))
                    draw = ImageDraw.Draw(overlay)
                    box = [
                        max(boxes[i].left-self.enlarge_boxes, 0),
                        max(boxes[i].top-self.enlarge_boxes, 0),
                        min(boxes[i].right+self.enlarge_boxes, image_i.width),
                        min(boxes[i].bottom+self.enlarge_boxes, image_i.height)
                    ]
                    draw.rectangle((tuple(box[:2]), tuple(box[2:])), fill=TINT_COLOR+(127,))
                    shaded_image = Image.alpha_composite(image_i, overlay)
                    shaded_image = shaded_image.convert('RGB')
                    preprocessed_images = self.preprocess_image(shaded_image) # []
                    for j, img in enumerate(preprocessed_images):
                        images[j].append(img.to(self.device))
            imgs = [torch.stack(image_list) for image_list in images]
        else:
            imgs = [[] for _ in self.models]
        text_tensor = self.preprocess_text(caption.lower()).to(self.device)
        return imgs, text_tensor

    @torch.no_grad()
    def __call__(self, caption: str, image: Image, boxes: List[Box], image_name: str = None) -> torch.Tensor:
        images, text_tensor = self.tensorize_inputs(caption, image, boxes, image_name)
        all_logits_per_image = []
        all_logits_per_text = []
        box_representation_methods = self.box_representation_method.split(',')
        caption_hash = hashlib.md5(caption.encode('utf-8')).hexdigest()
        for model, images_t, model_name in zip(self.models, images, self.model_names):
            if self.cache_path is not None:
                text_cache_path = os.path.join(self.cache_path, model_name, "text"+("_shade" if self.box_representation_method == "shade" else ""))
            image_features = None
            text_features = None
            if self.cache_path is not None and os.path.exists(os.path.join(self.cache_path, model_name)):
                if os.path.exists(os.path.join(text_cache_path, caption_hash+".pt")):
                    text_features = torch.load(os.path.join(text_cache_path, caption_hash+".pt"), map_location=self.device)
                if os.path.exists(os.path.join(self.cache_path, model_name, image_name)):
                    if all([os.path.exists(os.path.join(self.cache_path, model_name, image_name, method_name+".pt")) for method_name in box_representation_methods]):
                        image_features = []
                        for method_name in box_representation_methods:
                            features = torch.load(os.path.join(self.cache_path, model_name, image_name, method_name+".pt"), map_location=self.device)
                            image_features.append(torch.stack([
                                features[(box.x, box.y, box.w, box.h)]
                                for box in boxes
                            ]))
                        image_features = torch.stack(image_features)
                        image_features = image_features.view(-1, image_features.shape[-1])
            logits_per_image, logits_per_text, image_features, text_features = self.call_model(model, images_t, text_tensor, image_features=image_features, text_features=text_features)
            all_logits_per_image.append(logits_per_image)
            all_logits_per_text.append(logits_per_text)
            if self.cache_path is not None and image_name is not None and image_features is not None:
                image_features = image_features.view(len(box_representation_methods), len(boxes), image_features.shape[-1])
                if not os.path.exists(os.path.join(self.cache_path, model_name, image_name)):
                    os.makedirs(os.path.join(self.cache_path, model_name, image_name))
                for i in range(image_features.shape[0]):
                    method_name = box_representation_methods[i]
                    if not os.path.exists(os.path.join(self.cache_path, model_name, image_name, method_name+".pt")):
                        image_features_dict = {(box.x, box.y, box.w, box.h): image_features[i,j,:].cpu() for j, box in enumerate(boxes)}
                        torch.save(image_features_dict, os.path.join(self.cache_path, model_name, image_name, method_name+".pt"))
            if self.cache_path is not None and not os.path.exists(os.path.join(text_cache_path, caption_hash+".pt")) and text_features is not None:
                assert text_features.shape[0] == 1
                if not os.path.exists(text_cache_path):
                    os.makedirs(text_cache_path)
                torch.save(text_features.cpu(), os.path.join(text_cache_path, caption_hash+".pt"))

        all_logits_per_image = torch.stack(all_logits_per_image).sum(0)
        all_logits_per_text = torch.stack(all_logits_per_text).sum(0)
        if self.method_aggregator == "max":
            all_logits_per_text = all_logits_per_text.view(-1, len(boxes)).max(dim=0, keepdim=True)[0]
        elif self.method_aggregator == "sum":
            all_logits_per_text = all_logits_per_text.view(-1, len(boxes)).sum(dim=0, keepdim=True)
        return all_logits_per_text.view(-1)


class FLAVAExecutor(Executor):
    def __init__(self, flava_model: str = "facebook/flava-full", device: str = "cpu", box_representation_method: str = "crop", method_aggregator: str = "max", enlarge_boxes: int = 0, expand_position_embedding: bool = False, square_size: bool = False, blur_std_dev: int = 100, cache_path: str = None, enable_lora: bool = False, lora_path: str = None) -> None:
        super().__init__(device, box_representation_method, method_aggregator, enlarge_boxes, expand_position_embedding, square_size, blur_std_dev, cache_path)
        self.flava_model = flava_model
        # model = FlavaModel.from_pretrained(flava_model).to(self.device).eval()


        base_model = FlavaModel.from_pretrained(flava_model)
        if enable_lora:
            config = LoraConfig(
                r=16,
                lora_alpha=16,
                target_modules=["query", "value"],
                lora_dropout=0,
                bias="none",
                modules_to_save=[],
            )
            model = get_peft_model(base_model, config)

            state_dict = torch.load(lora_path)['state_dict']
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("module."):

                    new_key = k[7:] 
                else:
                    new_key = k
                new_state_dict[new_key] = v

            model.load_state_dict(new_state_dict)
        else:
            model = base_model
        model = model.to(self.device).eval()

        preprocess = FlavaFeatureExtractor.from_pretrained(flava_model)
        self.tokenizer = BertTokenizer.from_pretrained(flava_model)
        self.models = []
        self.model_names = [flava_model]
        self.preprocesses = []
        self.models.append(model)
        self.preprocesses.append(preprocess)
        self.models = torch.nn.ModuleList(self.models)

    def preprocess_image(self, image: Image) -> List[torch.Tensor]:
        return [preprocess(image, return_tensors="pt")['pixel_values'].squeeze(0) for preprocess in self.preprocesses]

    def preprocess_text(self, text: str) -> torch.Tensor:
        inputs = self.tokenizer(["a photo of "+text.lower()], return_tensors="pt", padding="max_length", max_length=77)
        return inputs

    def call_model(self, model: torch.nn.Module, images: torch.Tensor, text: Dict, image_features: torch.Tensor = None, text_features: torch.Tensor = None) -> torch.Tensor:
        batch_size = 256
        if image_features is None:
            # print('computing image features')
            # image_features = model.get_image_features(**images)[:, 0]
            # image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            image_features_list = []
            for i in range(0, len(images), batch_size):
                batch_images = images[i:i+batch_size]
                batch_features = model.get_image_features(pixel_values=batch_images)[:, 0]
                image_features_list.append(batch_features)
            image_features = torch.cat(image_features_list, dim=0)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        if text_features is None:
            # print('computing text features')
            text_features = model.get_text_features(**text)[:, 0]
            # normalized features
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logits_per_image = image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        return logits_per_image, logits_per_text, image_features, text_features


    def __call__(self, caption: str, image: Image, boxes: List[Box], image_name: str = None) -> torch.Tensor:
        result = super().__call__(caption, image, boxes, image_name)
        return result


class ClipExecutor(Executor):
    def __init__(self, clip_model: str = "ViT-B/32", device: str = "cpu", box_representation_method: str = "crop", method_aggregator: str = "max", enlarge_boxes: int = 0, expand_position_embedding: bool = False, square_size: bool = False, blur_std_dev: int = 100, cache_path: str = None, enable_lora: bool = False, lora_path: str = None) -> None:
        super().__init__(device, box_representation_method, method_aggregator, enlarge_boxes, expand_position_embedding, square_size, blur_std_dev, cache_path)
        self.clip_models = clip_model.split(",")
        if enable_lora:
            self.lora_paths = lora_path.split(",")
        self.model_names = [model_name.replace("/", "_") for model_name in self.clip_models]
        self.models = []
        self.preprocesses = []
        for i, model_name in enumerate(self.clip_models):
            if enable_lora:
                model, _, preprocess = create_model_and_transforms(model_name, self.lora_paths[i], device=self.device, lora=4, jit=False)
            else:
                model, _, preprocess = create_model_and_transforms(model_name, "openai", device=self.device, lora=-1, jit=False)
            self.models.append(model)
            if self.square_size:
                print("Square size!")
                preprocess.transforms[0] = transforms.Resize(model.visual.image_size, interpolation=transforms.InterpolationMode.BICUBIC)
            self.preprocesses.append(preprocess)
        self.models = torch.nn.ModuleList(self.models)

    def preprocess_text(self, text: str) -> torch.Tensor:
        if "shade" in self.box_representation_method:
            return clip.tokenize([text.lower()+" is in red color."])
        return clip.tokenize(["a photo of "+text.lower()])

    def call_model(self, model: torch.nn.Module, images: torch.Tensor, text: torch.Tensor, image_features: torch.Tensor = None, text_features: torch.Tensor = None) -> torch.Tensor:
        batch_size = 256
        if image_features is None:
            # print('computing image features')
            # image_features = model.encode_image(images)
            # image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            image_features_list = []
            for i in range(0, len(images), batch_size):
                batch_images = images[i:i+batch_size]
                batch_features = model.encode_image(batch_images)
                image_features_list.append(batch_features)
            image_features = torch.cat(image_features_list, dim=0)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        if text_features is None:
            # print('computing text features')
            text_features = model.encode_text(text)
            # normalized features
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = model.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        return logits_per_image, logits_per_text, image_features, text_features

    def __call__(self, caption: str, image: Image, boxes: List[Box], image_name: str = None) -> torch.Tensor:
        if self.expand_position_embedding:
            original_preprocesses = self.preprocesses
            new_preprocesses = []
            original_position_embeddings = []
            for model_name, model, preprocess in zip(self.clip_models, self.models, self.preprocesses):
                if "RN" in model_name:
                    model_spatial_dim = int((model.visual.attnpool.positional_embedding.shape[0]-1)**0.5)
                    patch_size = model.visual.input_resolution // model_spatial_dim
                    original_positional_embedding = model.visual.attnpool.positional_embedding.clone()
                    model.visual.attnpool.positional_embedding = torch.nn.Parameter(torch.nn.functional.interpolate(
                        model.visual.attnpool.positional_embedding[1:,:].permute(1, 0).view(1, -1, model_spatial_dim, model_spatial_dim),
                        size=(image.height // patch_size, image.width // patch_size),
                        mode='bicubic',
                        align_corners=False
                    ).squeeze(0).permute(1, 2, 0).view(-1, original_positional_embedding.shape[-1]))
                    model.visual.attnpool.positional_embedding = torch.nn.Parameter(torch.cat((
                        original_positional_embedding[:1,:],
                        model.visual.attnpool.positional_embedding
                    ), dim=0))
                    transform = transforms.Compose([
                        transforms.Resize(((image.height // patch_size)*patch_size, (image.width // patch_size)*patch_size), interpolation=Image.BICUBIC),
                        lambda image: image.convert("RGB"),
                        transforms.ToTensor(),
                        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                    ])
                else:
                    model_spatial_dim = int((model.visual.positional_embedding.shape[0]-1)**0.5)
                    patch_size = model.visual.input_resolution // model_spatial_dim
                    original_positional_embedding = model.visual.positional_embedding.clone()
                    model.visual.positional_embedding = torch.nn.Parameter(torch.nn.functional.interpolate(
                        model.visual.positional_embedding[1:,:].permute(1, 0).view(1, -1, model_spatial_dim, model_spatial_dim),
                        size=(image.height // patch_size, image.width // patch_size),
                        mode='bicubic',
                        align_corners=False
                    ).squeeze(0).permute(1, 2, 0).view(-1, original_positional_embedding.shape[-1]))
                    model.visual.positional_embedding = torch.nn.Parameter(torch.cat((
                        original_positional_embedding[:1,:],
                        model.visual.positional_embedding
                    ), dim=0))
                    transform = transforms.Compose([
                        transforms.Resize(((image.height // patch_size)*patch_size, (image.width // patch_size)*patch_size), interpolation=Image.BICUBIC),
                        lambda image: image.convert("RGB"),
                        transforms.ToTensor(),
                        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                    ])
                new_preprocesses.append(transform)
                original_position_embeddings.append(original_positional_embedding)
            self.preprocesses = new_preprocesses
        result = super().__call__(caption, image, boxes, image_name)
        if self.expand_position_embedding:
            self.preprocesses = original_preprocesses
            for model, model_name, pos_embedding in zip(self.models, self.clip_models, original_position_embeddings):
                if "RN" in model_name:
                    model.visual.attnpool.positional_embedding = torch.nn.Parameter(pos_embedding)
                else:
                    model.visual.positional_embedding = torch.nn.Parameter(pos_embedding)
        return result