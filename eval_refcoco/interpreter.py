from typing import NamedTuple, List, Callable
import sys
import re
import numpy as np
import torch
from numpy.linalg import norm
from itertools import product, groupby
from PIL import Image
import itertools
import spacy


# Do two line segments intersect? Copied from
# https://stackoverflow.com/questions/3838329/how-can-i-check-if-two-segments-intersect


def ccw(A, B, C):
    return (C.y - A.y) * (B.x - A.x) > (B.y - A.y) * (C.x - A.x)


def intersect(A, B, C, D):
    """Do line segments AB and CD intersect?"""
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


class Box(NamedTuple):
    x: int
    y: int
    w: int = 0
    h: int = 0

    @property
    def left(self):
        return self.x

    @property
    def right(self):
        return self.x + self.w

    @property
    def top(self):
        return self.y

    @property
    def bottom(self):
        return self.y + self.h

    @property
    def center(self):
        return Box(self.x + self.w // 2, self.y + self.h // 2)

    def corners(self):
        yield Box(self.x, self.y)
        yield Box(self.x + self.w, self.y)
        yield Box(self.x + self.w, self.y + self.h)
        yield Box(self.x, self.y + self.h)

    @property
    def area(self):
        return self.w * self.h

    def intersect(self, other: "Box") -> "Box":
        x1 = max(self.x, other.x)
        x2 = max(x1, min(self.x+self.w, other.x+other.w))
        y1 = max(self.y, other.y)
        y2 = max(y1, min(self.y+self.h, other.y+other.h))
        return Box(x=x1, y=y1, w=x2-x1, h=y2-y1)

    def min_bounding(self, other: "Box") -> "Box":
        corners = list(self.corners())
        corners.extend(other.corners())
        min_x = min_y = float("inf")
        max_x = max_y = -float("inf")

        for item in corners:
            min_x = min(min_x, item.x)
            min_y = min(min_y, item.y)
            max_x = max(max_x, item.x)
            max_y = max(max_y, item.y)

        return Box(min_x, min_y, max_x - min_x, max_y - min_y)

    def expand(self, growth: float = .1) -> "Box":
        factor = 1 + growth
        w = factor * self.w
        h = factor * self.h
        return Box(min_x - (w - self.w) / 2, min_y - (h - self.h) / 2, w, h)


def iou(box1, box2):
    x1 = max(box1.x, box2.x)
    x2 = max(x1, min(box1.x+box1.w, box2.x+box2.w))
    y1 = max(box1.y, box2.y)
    y2 = max(y1, min(box1.y+box1.h, box2.y+box2.h))
    intersection = Box(x=x1, y=y1, w=x2-x1, h=y2-y1)
    intersection_area = intersection.area
    union_area = box1.area+box2.area-intersection_area
    return intersection_area / union_area


def all_equal(iterable):
    """Are all elements the same?"""
    g = groupby(iterable)
    return next(g, True) and not next(g, False)


class spatial:
    """A decorator that converts a predicate over boxes to a function that returns a tensor over all boxes."""

    def __init__(self, arity: int = 2, enforce_antisymmetry: bool = False):
        self.arity = arity
        self.enforce_antisymmetry = enforce_antisymmetry  # Zero out any entries where two boxes are the same.

    def __call__(self, predicate: Callable[[Box], float]) -> Callable[["Environment"], np.ndarray]:
        def _rel(env):
            n_boxes = len(env.boxes)
            tensor = np.empty([n_boxes for _ in range(self.arity)])
            enum_boxes = list(enumerate(env.boxes))
            for pairs in product(*[enum_boxes for _ in range(self.arity)]):
                indices, boxes = zip(*pairs)
                if self.enforce_antisymmetry and len(set(indices)) < len(indices):
                    tensor[indices] = 0.
                else:
                    tensor[indices] = predicate(*boxes)
            return tensor
        return _rel


class Environment:
    def __init__(self, image: Image, boxes: List[Box], executor: "Executor" = None, image_name: str = None, gpt_triplets: List[List] = None):
        self.image = image
        self.boxes = boxes
        self.executor = executor  # An object or callback that can query CLIP with captions/images.
        self.image_name = image_name
        self.gpt_triplets = gpt_triplets

    def uniform(self) -> np.ndarray:
        n_boxes = len(self.boxes)
        return 1 / n_boxes * np.ones(n_boxes)

    def filter(self,
               caption: str,
               temperature: float = 1.,
               area_threshold: float = 0.0,
               softmax: bool = False,
               expand: float = None
              ) -> np.ndarray:
        """Return a new distribution reflecting the likelihood that `caption` describes the content of each box."""
        area_filtered_dist = torch.from_numpy(self.filter_area(area_threshold)).to(self.executor.device)
        candidate_indices = [i for i in range(len(self.boxes)) if float(area_filtered_dist[i]) > 0.0]
        boxes = [self.boxes[i] for i in candidate_indices]
        if len(boxes) == 0:
            boxes = self.boxes
            candidate_indices = list(range(len(boxes)))
        if expand is not None:
            boxes = [box.expand(expand) for box in boxes]
        result_partial = self.executor(caption, self.image, boxes, image_name=self.image_name)
        result_partial = result_partial.float()
        if not softmax:
            result_partial = (result_partial-result_partial.mean()) / (result_partial.std() + 1e-9)
            result_partial = (temperature * result_partial).sigmoid()
            result = torch.zeros((len(self.boxes))).to(result_partial.device)
            result[candidate_indices] = result_partial
        else:
            result = torch.zeros((len(self.boxes))).to(result_partial.device)
            result[candidate_indices] = result_partial.softmax(dim=-1)
        return result.cpu().numpy()

    def matching(self,
               caption: str,
               temperature: float = 1.,
               area_threshold: float = 0.0,
               softmax: bool = False,
               expand: float = None,
               nlp: object = None,
               rule_filter: bool = False,
               heuristics: object = None
              ) -> np.ndarray:
        area_filtered_dist = torch.from_numpy(self.filter_area(area_threshold)).to(self.executor.device)
        # invalid_filtered_dist = torch.from_numpy(self.filter_invalid_box()).to(self.executor.device)
        # candidate_indices = [i for i in range(len(self.boxes)) if (float(area_filtered_dist[i]) > 0.0 and float(invalid_filtered_dist[i]) > 0.0)]
        candidate_indices = [i for i in range(len(self.boxes)) if float(area_filtered_dist[i]) > 0.0]
        boxes = [self.boxes[i] for i in candidate_indices]
        if len(boxes) == 0:
            boxes = self.boxes
            candidate_indices = list(range(len(boxes)))
        if expand is not None:
            boxes = [box.expand(expand) for box in boxes]

        candidate_pair_indices = list(itertools.product(candidate_indices, repeat=2))
        box_pairs = [(self.boxes[i], self.boxes[j]) for i, j in candidate_pair_indices]
        # print(len(box_pairs))

        tri_entity = self.gpt_triplets["entity"]
        tri_relations = self.gpt_triplets["relations"] 
        filtered_relations = []
        filtered_tri_relations = []
        entity_to_index = {tri_entity: 0}
        index_counter = 1

        for i, triplet in enumerate(tri_relations):
            entity1, relation, entity2 = triplet

            if entity1 != tri_entity and entity2 != tri_entity:
                continue
            relation = f"{entity1} {relation} {entity2}"

            if entity1 == tri_entity:
                index1 = 0
            elif entity1 == "":
                index1 = 0
                entity1 = tri_entity
            else:
                if entity1 not in entity_to_index:
                    entity_to_index[entity1] = index_counter
                    index_counter += 1
                index1 = entity_to_index[entity1]
            
            if entity2 == tri_entity:
                index2 = 0
            elif entity2 == "":
                index2 = 0
                entity2 = tri_entity
            else:
                if entity2 not in entity_to_index:
                    entity_to_index[entity2] = index_counter
                    index_counter += 1
                index2 = entity_to_index[entity2]

            if entity1 == tri_entity:
                entity1 = caption
            if entity2 == tri_entity:
                entity2 = caption

            filtered_tri_relations.append([entity1, relation, entity2])
            filtered_relations.append([index1, index2])
            
        prompts = [[triplet[0], triplet[1], triplet[2]] for triplet in filtered_tri_relations]

        original_candidate_pair_indices = candidate_pair_indices
        original_box_pairs = box_pairs
        # print(prompts)

        filtered_relations = torch.tensor(filtered_relations)
        pair_matrix = []
        for prompt in prompts:
            candidate_pair_indices = original_candidate_pair_indices
            box_pairs = original_box_pairs
            matrix=[]
            sub = prompt[0]
            action = prompt[1]
            obj = prompt[2]

            if rule_filter:
                doc = nlp(action)

                # filter those redundant box pairs
                for heuristic in heuristics.relations + heuristics.superlatives:
                    if any(token.text in heuristic.keywords for token in doc):
                        box_rel = heuristic.callback(self)
                        candidate_pair_indices_filtered = [
                            (i, j) for i, j in candidate_pair_indices if box_rel[i][j] == 1
                        ]
                        candidate_pair_indices = candidate_pair_indices_filtered
                        box_pairs = [(self.boxes[i], self.boxes[j]) for i, j in candidate_pair_indices]

            if len(candidate_pair_indices) == 0:
                print("No box pairs left after rule filtering")
                print(action)
                candidate_pair_indices = original_candidate_pair_indices
                box_pairs = original_box_pairs
            sub_sim = self.executor(sub, self.image, boxes, image_name=self.image_name)
            action_sim = self.executor(action, self.image, box_pairs, image_name=self.image_name)
            obj_sim = self.executor(obj, self.image, boxes, image_name=self.image_name)
            for idx, pair in enumerate(original_candidate_pair_indices):
                if pair not in candidate_pair_indices:
                    matrix.append(torch.tensor(float('-inf')).to('cuda'))
                else:
                    i, j = pair
                    sub_idx = candidate_indices.index(i)
                    obj_idx = candidate_indices.index(j)
                    act_idx = candidate_pair_indices.index(pair)
                    # matrix.append((sub_sim[sub_idx] + action_sim[act_idx] + obj_sim[obj_idx])/3)
                    matrix.append((sub_sim[sub_idx] + obj_sim[obj_idx])/2)
            pair_matrix.append(torch.stack(matrix))
        pair_matrix = torch.stack(pair_matrix)
        textual_pairs_pred_index = pair_matrix.argmax(dim=1)

        instance_pos = torch.nonzero(0 == filtered_relations).to('cpu')

        # all_zeros_pos = torch.nonzero(0 == filtered_relations).to('cpu')

        # instance_pos = all_zeros_pos[all_zeros_pos[:, 0] != 1]


        textual_pair_index = instance_pos[:, 0]
        visual_pair_index = textual_pairs_pred_index[textual_pair_index].to('cpu')
        pair_score = pair_matrix[textual_pair_index, visual_pair_index]
        original_candidate_pair_indices = torch.tensor(original_candidate_pair_indices)
        predicted_visual_instances = original_candidate_pair_indices[visual_pair_index].gather(1, instance_pos[:, 1].unsqueeze(-1)).squeeze(-1)
        assert(pair_score.shape[0] == predicted_visual_instances.shape[0])
        instance_score = torch.zeros((len(self.boxes))).to(pair_score.device)
        for i, idx in enumerate(predicted_visual_instances):
            instance_score[idx] += pair_score[i]
        
        result = instance_score.softmax(dim=-1)
        return result.cpu().numpy()

    def filter_area(self, area_threshold: float) -> np.ndarray:
        """Return a new distribution in which all boxes whose area as a fraction of the image is less than the threshold."""
        image_area = self.image.width*self.image.height
        return np.array([1 if self.boxes[i].area/image_area > area_threshold else 0 for i in range(len(self.boxes))])

    # def filter_invalid_box(self) -> np.ndarray:
    #     return np.array([1 if (self.boxes[i].w > 3 and self.boxes[i].h > 3) else 0 for i in range(len(self.boxes))])

    @spatial()
    def left_of(b1, b2):
        return (b1.right+b1.left) / 2 < (b2.right+b2.left) / 2

    @spatial()
    def right_of(b1, b2):
        return (b1.right+b1.left) / 2 > (b2.right+b2.left) / 2

    @spatial()
    def above(b1, b2):
        return (b1.bottom+b1.top) < (b2.bottom+b2.top)

    @spatial()
    def below(b1, b2):
        return (b1.bottom+b1.top) > (b2.bottom+b2.top)

    @spatial()
    def bigger_than(b1, b2):
        return b1.area > b2.area

    @spatial()
    def smaller_than(b1, b2):
        return b1.area < b2.area

    @spatial(enforce_antisymmetry=False)
    def within(box1, box2):
        """Return percent of box1 inside box2."""
        intersection = box1.intersect(box2)
        return intersection.area / box1.area

    @spatial(arity=3, enforce_antisymmetry=True)
    def between(box1, box2, box3):
        """How much of box1 lies in min bounding box over box2 and box3?"""
        min_bounding = box2.min_bounding(box3)
        intersect = box1.intersect(min_bounding)
        return intersect.area / box1.area
