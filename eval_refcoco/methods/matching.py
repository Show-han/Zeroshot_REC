"""Matching with CLIP and ChatGPT"""

from overrides import overrides
from typing import Dict, Any, List
import numpy as np
import torch
import spacy
from argparse import Namespace

from .ref_method import RefMethod
from lattice import Product as L
from heuristics import Heuristics


class Matching(RefMethod):
    """Matching with CLIP and ChatGPT"""

    def __init__(self, args: Namespace):
        self.args = args
        self.box_area_threshold = args.box_area_threshold
        self.batch_size = args.batch_size
        self.batch = []
        self.nlp = spacy.load('en_core_web_sm')
        self.heuristics = Heuristics(args)

    @overrides
    def execute(self, caption: str, env: "Environment") -> Dict[str, Any]:
        probs = env.matching(caption, area_threshold = self.box_area_threshold, softmax=True, nlp = self.nlp, rule_filter=self.args.rule_filter, heuristics = self.heuristics)
        pred = np.argmax(probs)
        return {
            "probs": probs,
            "pred": pred,
            "box": env.boxes[pred],
        }
