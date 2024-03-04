import ast
import json
import logging
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from multiprocessing import Value

import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, IterableDataset, get_worker_info
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, AutoModel
import json
import re
import time
from torch.utils.data import ConcatDataset
import configparser


try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

from VLA_finetune.open_clip import tokenize

from transformers import FlavaFeatureExtractor, FlavaModel, BertTokenizer

flava_preprocess = FlavaFeatureExtractor.from_pretrained("facebook/flava-full")
flava_tokenizer = BertTokenizer.from_pretrained("facebook/flava-full")


def union_bbox(bbox1, bbox2):
    """Returns the union of two bounding boxes."""
    x1, y1, x2, y2 = bbox1
    x1_, y1_, x2_, y2_ = bbox2
    union_x1 = min(x1, x1_)
    union_y1 = min(y1, y1_)
    union_x2 = max(x2, x2_)
    union_y2 = max(y2, y2_)
    return [union_x1, union_y1, union_x2, union_y2]

def is_bbox_invalid(bbox):
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    return width <= 10 or height <= 10



class HICODataset(Dataset):
    def __init__(self, annotation_file, verb_names_file, img_dir, transforms, flava=False):
        self.img_dir = img_dir
        self.transforms = transforms
        self.flava = flava
        # Load annotation file
        with open(annotation_file, 'r') as f:
            raw_data = json.load(f)
        
        # Load verb and coco names
        with open(verb_names_file, 'r') as f:
            names_data = json.load(f)
            self.verb_names = names_data["verb_names"]
            self.coco_names = names_data["coco_names"]
            
        self.grouped_data = {}
        for entry in raw_data:
            for hoi_annotation in entry["hoi_annotation"]:
                subject_id = entry["annotations"][hoi_annotation["subject_id"]]["category_id"]
                object_id = entry["annotations"][hoi_annotation["object_id"]]["category_id"]
                action_id = hoi_annotation["category_id"]

                key = (subject_id, object_id, action_id)
                img_path = os.path.join(self.img_dir, entry["file_name"])

                subject_bbox = entry["annotations"][hoi_annotation["subject_id"]]["bbox"]
                object_bbox = entry["annotations"][hoi_annotation["object_id"]]["bbox"]
                
                union_bbox_coords = union_bbox(subject_bbox, object_bbox)

                data_to_add = {
                    'img_path': img_path,
                    'subject_bbox': subject_bbox,
                    'object_bbox': object_bbox,
                    'union_bbox': union_bbox_coords
                }

                if key not in self.grouped_data:
                    self.grouped_data[key] = []
                self.grouped_data[key].append(data_to_add)

        self.keys = list(self.grouped_data.keys())

    def __len__(self):
        return len(self.grouped_data)

    def __getitem__(self, idx):
        key = self.keys[idx]
        entries = self.grouped_data[key]
        selected_entry = random.choice(entries)

        img_path = selected_entry['img_path']
        subject_bbox = selected_entry['subject_bbox']
        object_bbox = selected_entry['object_bbox']
        union_bbox = selected_entry['union_bbox']

        img = Image.open(img_path).convert('RGB')

        # Crop images
        subject_img = TF.crop(img, subject_bbox[1], subject_bbox[0], subject_bbox[3] - subject_bbox[1], subject_bbox[2] - subject_bbox[0])
        object_img = TF.crop(img, object_bbox[1], object_bbox[0], object_bbox[3] - object_bbox[1], object_bbox[2] - object_bbox[0])
        action_img = TF.crop(img, union_bbox[1], union_bbox[0], union_bbox[3] - union_bbox[1], union_bbox[2] - union_bbox[0])

        img.close()
        del img, img_path, selected_entry

        # Applying transformations on images
        if self.flava:
            subject_img = flava_preprocess(subject_img, return_tensors="pt")['pixel_values'].squeeze(0)
            object_img = flava_preprocess(object_img, return_tensors="pt")['pixel_values'].squeeze(0) 
            action_img = flava_preprocess(action_img, return_tensors="pt")['pixel_values'].squeeze(0)
        else:
            subject_img = self.transforms(subject_img)
            object_img = self.transforms(object_img)  
            action_img = self.transforms(action_img)

        # Getting names
        sub = re.sub('_+', ' ', self.coco_names[key[0]])
        obj = re.sub('_+', ' ', self.coco_names[key[1]])
        act = re.sub('_+', ' ', self.verb_names[key[2]])

        subject_name = f"a photo of {sub}"
        object_name = f"a photo of {obj}"
        action_name = f"{sub} {act} {obj}"
        
        # Tokenizing text 

        if self.flava:
            subject_tokens = flava_tokenizer([subject_name], return_tensors="pt", padding="max_length", max_length=100, truncation=True)['input_ids'].squeeze(0)
            object_tokens = flava_tokenizer([object_name], return_tensors="pt", padding="max_length", max_length=100, truncation=True)['input_ids'].squeeze(0)
            action_tokens = flava_tokenizer([action_name], return_tensors="pt", padding="max_length", max_length=100, truncation=True)['input_ids'].squeeze(0)
        else:
            subject_tokens = tokenize(subject_name)[0]
            object_tokens = tokenize(object_name)[0]
            action_tokens = tokenize(action_name)[0]

        stacked_images = torch.stack([subject_img, object_img, action_img], dim=0)
        stacked_tokens = torch.stack([subject_tokens, object_tokens, action_tokens], dim=0)
        
        return stacked_images, stacked_tokens


class SWIGDataset(Dataset):
    def __init__(self, transforms, annotation_train, annotation_val, imsitu_space_file, img_dir, triplets_file, flava=False):
        self.img_dir = img_dir
        self.flava = flava
        self.transforms = transforms

        with open(annotation_train, "r") as f:
            annotation_train = json.load(f)

        with open(annotation_val, "r") as f:
            annotation_val = json.load(f)

        with open(triplets_file, "r") as f:
            triplets = json.load(f)

        with open(imsitu_space_file, "r") as f:
            self.imsitu = json.load(f)

        annotations = {**annotation_train, **annotation_val}
        self.result_dict = {}

        for img_name, data in annotations.items():
            img_path = f"{img_dir}/" + img_name
            entities = {}
            for entity, bbox in data['bb'].items():
                if bbox != [-1, -1, -1, -1] and data['frames'][0].get(entity, "") != "":
                    entities[entity] = data['frames'][0][entity]
                    
            verb = data['verb']
            triplets_for_verb = triplets.get(verb, [])
            
            for triplet in triplets_for_verb:
                # print('triplet: ', triplet)
                subject_key = None
                object_key = None

                formatted_subject = triplet[0]
                formatted_object = triplet[2]
                for word in triplet[0].split():
                    if word.lower() in entities:
                        subject_key = word.lower()
                        formatted_subject = formatted_subject.replace(word.upper(), "{" + entities[subject_key] + "}")
                        break

                for word in triplet[2].split():
                    if word.lower() in entities:
                        object_key = word.lower()
                        formatted_object = formatted_object.replace(word.upper(), "{" + entities[object_key] + "}")
                        break
                
                if subject_key in entities and object_key in entities:
                    key = (formatted_subject, formatted_object, triplet[1])
                    subject_bbox = data['bb'][subject_key]
                    object_bbox = data['bb'][object_key]

                    if is_bbox_invalid(subject_bbox) or is_bbox_invalid(object_bbox):
                        print(f"Invalid bbox detected in image: {img_path}")
                        continue
                    union_bbox_value = union_bbox(subject_bbox, object_bbox)
                    data_to_add = {
                        'img_path': img_path,
                        'subject_bbox': subject_bbox,
                        'object_bbox': object_bbox,
                        'union_bbox': union_bbox_value
                    }
                    if key not in self.result_dict:
                        self.result_dict[key] = []
                    self.result_dict[key].append(data_to_add)

        self.keys = list(self.result_dict.keys())

    def __len__(self):
        return len(self.result_dict)

    def __getitem__(self, idx):

        key = self.keys[idx]
        entries = self.result_dict[key]
        selected_entry = random.choice(entries)

        img_path = selected_entry['img_path']
        subject_bbox = selected_entry['subject_bbox']
        object_bbox = selected_entry['object_bbox']
        union_bbox = selected_entry['union_bbox']

        img = Image.open(img_path).convert('RGB')

        # Crop images
        subject_img = TF.crop(img, subject_bbox[1], subject_bbox[0], subject_bbox[3] - subject_bbox[1], subject_bbox[2] - subject_bbox[0])
        object_img = TF.crop(img, object_bbox[1], object_bbox[0], object_bbox[3] - object_bbox[1], object_bbox[2] - object_bbox[0])
        action_img = TF.crop(img, union_bbox[1], union_bbox[0], union_bbox[3] - union_bbox[1], union_bbox[2] - union_bbox[0])

        img.close()
        del img, img_path, selected_entry
        # Applying transformations on images
        if self.flava:
            subject_img = flava_preprocess(subject_img, return_tensors="pt")['pixel_values'].squeeze(0)
            object_img = flava_preprocess(object_img, return_tensors="pt")['pixel_values'].squeeze(0) 
            action_img = flava_preprocess(action_img, return_tensors="pt")['pixel_values'].squeeze(0)
        else:
            subject_img = self.transforms(subject_img)
            object_img = self.transforms(object_img)  
            action_img = self.transforms(action_img)

        # Getting names
        names = []
        for k in key:
            matches = re.findall(r'\{n[0-9]+\}', k)
            for match in matches:
                if match[1:-1] in self.imsitu['nouns']:
                    replacement = random.choice(self.imsitu['nouns'][match[1:-1]]['gloss'])
                    k = k.replace(match, replacement)
            names.append(k)
        subject_name = f"a photo of {names[0]}"
        object_name = f"a photo of {names[1]}"
        action_name = f"{names[0]} {names[2]} {names[1]}"

        if self.flava:
            subject_tokens = flava_tokenizer([subject_name], return_tensors="pt", padding="max_length", max_length=100, truncation=True)['input_ids'].squeeze(0)
            object_tokens = flava_tokenizer([object_name], return_tensors="pt", padding="max_length", max_length=100, truncation=True)['input_ids'].squeeze(0)
            action_tokens = flava_tokenizer([action_name], return_tensors="pt", padding="max_length", max_length=100, truncation=True)['input_ids'].squeeze(0)
        else:
            subject_tokens = tokenize(subject_name)[0]
            object_tokens = tokenize(object_name)[0]
            action_tokens = tokenize(action_name)[0]    

        stacked_images = torch.stack([subject_img, object_img, action_img], dim=0)
        stacked_tokens = torch.stack([subject_tokens, object_tokens, action_tokens], dim=0)
        
        return stacked_images, stacked_tokens


class VGDataset(Dataset):
    def __init__(self, transforms, img_dir, meta_path, relation_path, attribute_path, flava=False):
        self.transforms = transforms
        self.flava = flava


        with open(meta_path, "r") as f:
            meta_data = json.load(f)

        with open(relation_path, "r") as f:
            relations = json.load(f)

        with open(attribute_path, "r") as f:
            attributes = json.load(f)

        path_without_coco = []
        relations_wo_coco = []
        attributes_wo_coco = []

        # filter COCO
        for image, attr, rel in zip(meta_data, attributes, relations):
            if image['coco_id'] is not None:
                continue
            else:
                assert image['image_id'] == attr['image_id'] and image['image_id'] == rel['image_id']
                path_without_coco.append('/'.join(image['url'].split('/')[-2:]))
                relations_wo_coco.append(rel['relationships'])
                attributes_wo_coco.append(attr['attributes'])


        self.result_dict = {}

        for idx, relation in enumerate(relations_wo_coco):
            image_path = os.path.join(img_dir, path_without_coco[idx])
            attribute = attributes_wo_coco[idx]

            attribute_dict = {}
            for item in attribute:
                object_id = item['object_id']
                attributes = item.get('attributes', [])
                attribute_dict[object_id] = attributes

            for rel in relation:
                # {'predicate': 'reads', 'object': {'name': 'pape av.', 'h': 83, 'synsets': [], 'object_id': 4462790, 'w': 131, 'y': 92, 'x': 378}, 'relationship_id': 4748822, 'synsets': [], 'subject': {'name': 'sign', 'h': 145, 'synsets': [], 'object_id': 4462789, 'w': 217, 'y': 50, 'x': 331}}
                sub = rel['subject']
                obj = rel['object']
                sub_attr = attribute_dict[sub['object_id']]
                obj_attr = attribute_dict[obj['object_id']]
                sub_name = sub.get('name', "")
                obj_name = obj.get('name', "")
                predicate_name = rel.get('predicate', "")
                if sub_name == "" or obj_name == "" or predicate_name == "":
                    continue
                sub_box = [int(sub['x']), int(sub['y']), int(sub['x'])+int(sub['w']), int(sub['y'])+int(sub['h'])]
                obj_box = [int(obj['x']), int(obj['y']), int(obj['x'])+int(obj['w']), int(obj['y'])+int(obj['h'])]

                if is_bbox_invalid(sub_box) or is_bbox_invalid(obj_box):
                    continue

                sub_attr = []
                obj_attr = []

                if len(sub_attr) > 1:
                    formatted_attributes = ', '.join(sub_attr[:-1]) + ' and ' + sub_attr[-1]
                    sub_name = f"{formatted_attributes} {sub_name}"
                elif len(sub_attr) == 1:
                    sub_name = f"{sub_attr[0]} {sub_name}"
                else:
                    sub_name = sub_name

                if len(obj_attr) > 1:
                    formatted_attributes = ', '.join(obj_attr[:-1]) + ' and ' + obj_attr[-1]
                    obj_name = f"{formatted_attributes} {obj_name}"
                elif len(obj_attr) == 1:
                    obj_name = f"{obj_attr[0]} {obj_name}"
                else:
                    obj_name = obj_name

                key = (sub_name.lower(), obj_name.lower(), predicate_name.lower())              
                data_to_add = {
                    'img_path': image_path,
                    'subject_bbox': sub_box,
                    'object_bbox': obj_box,
                    'union_bbox': union_bbox(sub_box, obj_box)
                }

                if key not in self.result_dict:
                    self.result_dict[key] = []
                self.result_dict[key].append(data_to_add)

        self.keys = list(self.result_dict.keys())

    def __len__(self):
        return len(self.result_dict)

    def __getitem__(self, idx):
        key = self.keys[idx]
        entries = self.result_dict[key]
        selected_entry = random.choice(entries)

        img_path = selected_entry['img_path']
        subject_bbox = selected_entry['subject_bbox']
        object_bbox = selected_entry['object_bbox']
        union_bbox = selected_entry['union_bbox']

        img = Image.open(img_path).convert('RGB')

        # Crop images
        subject_img = TF.crop(img, subject_bbox[1], subject_bbox[0], subject_bbox[3] - subject_bbox[1], subject_bbox[2] - subject_bbox[0])
        object_img = TF.crop(img, object_bbox[1], object_bbox[0], object_bbox[3] - object_bbox[1], object_bbox[2] - object_bbox[0])
        action_img = TF.crop(img, union_bbox[1], union_bbox[0], union_bbox[3] - union_bbox[1], union_bbox[2] - union_bbox[0])

        img.close()
        del img, img_path, selected_entry
        
        # Applying transformations on images
        if self.flava:
            subject_img = flava_preprocess(subject_img, return_tensors="pt")['pixel_values'].squeeze(0)
            object_img = flava_preprocess(object_img, return_tensors="pt")['pixel_values'].squeeze(0) 
            action_img = flava_preprocess(action_img, return_tensors="pt")['pixel_values'].squeeze(0)
        else:
            subject_img = self.transforms(subject_img)
            object_img = self.transforms(object_img)  
            action_img = self.transforms(action_img)

        # Getting names
        subject_name = f"a photo of {key[0]}"
        object_name = f"a photo of {key[1]}"
        action_name = f"{key[0]} {key[2]} {key[1]}"
        
        if self.flava:
            subject_tokens = flava_tokenizer([subject_name], return_tensors="pt", padding="max_length", max_length=100, truncation=True)['input_ids'].squeeze(0)
            object_tokens = flava_tokenizer([object_name], return_tensors="pt", padding="max_length", max_length=100, truncation=True)['input_ids'].squeeze(0)
            action_tokens = flava_tokenizer([action_name], return_tensors="pt", padding="max_length", max_length=100, truncation=True)['input_ids'].squeeze(0)
        else:
            subject_tokens = tokenize(subject_name)[0]
            object_tokens = tokenize(object_name)[0]
            action_tokens = tokenize(action_name)[0]

        stacked_images = torch.stack([subject_img, object_img, action_img], dim=0)
        stacked_tokens = torch.stack([subject_tokens, object_tokens, action_tokens], dim=0)
        
        return stacked_images, stacked_tokens


class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value('i', epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)




def get_relation_dataset(args, preprocess_fn, is_train, epoch=0):
    config = configparser.ConfigParser()
    data_config_path = args.data_config_path
    config.read(data_config_path)

    HICO_data_path = config['HICO']['train_data'] if is_train else config['HICO']['val_data']
    HICO_annotation = config['HICO']['train_annotation_file'] if is_train else config['HICO']['val_annotation_file']

    dataset_hico = HICODataset(
        annotation_file = HICO_annotation,
        verb_names_file = config['HICO']['verb_names_file'],
        img_dir = HICO_data_path,
        transforms = preprocess_fn,
        flava = args.flava
    )

    if is_train:
        dataset_swig = SWIGDataset(transforms=preprocess_fn, annotation_train=config['SWIG']['annotation_train'], annotation_val=config['SWIG']['annotation_val'], imsitu_space_file=config['SWIG']['imsitu_space_file'], img_dir=config['SWIG']['img_dir'], triplets_file=config['SWIG']['triplets_file'], flava=args.flava)

        dataset_vg = VGDataset(transforms=preprocess_fn, img_dir=config['VG']['data_dir'], meta_path=config['VG']['meta_path'], relation_path=config['VG']['relation_path'], attribute_path=config['VG']['attribute_path'], flava = args.flava)


    if not is_train:
        dataset = dataset_hico
    else:
        dataset = ConcatDataset([dataset_hico, dataset_swig, dataset_vg])

    num_samples = len(dataset)
    print('len dataset: ', num_samples)
    
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)




def get_data(args, preprocess_fns, epoch=0):
    preprocess_train, preprocess_val = preprocess_fns
    data = {}

    data["train"] = get_relation_dataset(args, preprocess_train, is_train=True, epoch=epoch)
    data["val"] = get_relation_dataset(args, preprocess_val, is_train=False)


    return data

