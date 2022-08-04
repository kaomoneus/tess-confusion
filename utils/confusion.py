import dataclasses
import json
import logging
import multiprocessing
from collections import defaultdict
from copy import copy
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, Tuple, Union

import numpy as np
import textblob.en

from utils.needleman_wunsch import nw


LOG = logging.getLogger(__name__)


CharFrequencies = Dict[str, float]
_ConfusionMatrixDict = Dict[str, CharFrequencies]

confusion_vec_lock = multiprocessing.Lock()

ABSENSE_PROBABILITY = 1.e-9

CUSTOM_WORDS = [
    ("olesia", 1),
    ("babushkina", 1),
    ("andrey", 1),
    ("khalzov", 1),
    ("alexey", 1),
    ("kuzmenko", 1),
]


@dataclasses.dataclass
class ConfusionMatrix:
    confusion_matrix: _ConfusionMatrixDict = dataclasses.field(
        default_factory=defaultdict(lambda: defaultdict(lambda: 0.))
    )

    def train(self, recognition_ground_truth_gen: Iterable[Tuple[str, str]]):
        confusion_matrix = self.confusion_matrix

        for recognized, gt_str in recognition_ground_truth_gen:
            recognized_aligned, gt_aligned = nw(recognized, gt_str)

            # Get confusion vector

            confusion_vec = dict()

            for rec_char, gt_char in zip(
                    recognized_aligned.char_items, gt_aligned.char_items
            ):
                confusion_vec[rec_char] = gt_char

            # Aggregate it

            with confusion_vec_lock:
                for rec, correct in confusion_vec.items():
                    confusion_matrix[rec][correct] += 1

        # Normalize confusion matrix
        for rec, vec in confusion_matrix.items():
            total = sum([freq for freq in vec.values()])
            for key, freq in vec.items():
                vec[key] = freq / total

    @staticmethod
    def load(p: Union[str, Path]):
        if isinstance(p, Path):
            p = str(p)
        with open(p, "r") as mf:
            confusion_matrix = json.load(mf)
            return ConfusionMatrix(confusion_matrix)

    def save(self, p: Path):
        LOG.debug("Writing confusion matrix...")
        with open(p, "w") as mf:
            json.dump(self.confusion_matrix, mf, indent=2)
        LOG.debug(f"Matrix is written into '{p}'")

    def confusion_probability(self, recognized: str, correct: str):
        rec_aligned, correct_aligned = nw(recognized, correct)
        prob = 1.
        for rec, cor in zip(rec_aligned.char_items, correct_aligned.char_items):
            prob *= self.confusion_matrix[rec].get(cor, ABSENSE_PROBABILITY)

        return prob

    def suggest(self, recognized: str):

        tb_suggestions = textblob_all_suggestions(recognized)
        freqs = np.array([sug[1] for sug in tb_suggestions])

        conf_suggestions = [
            (c, self.confusion_probability(recognized, c))
            for c, _ in tb_suggestions
        ]
        confs = np.array([sug[1] for sug in conf_suggestions])

        total = freqs.dot(confs)
        prods = np.multiply(freqs, confs)
        prods /= total

        conf_suggestions = sorted([
            (w, sug)
            for (w, _), sug in zip(conf_suggestions, prods)
        ], key=lambda wsug: -wsug[1])

        return conf_suggestions


@lru_cache
def _get_spelling():
    _spelling = copy(textblob.en.spelling)
    _spelling.update(set(CUSTOM_WORDS))
    return _spelling


def textblob_all_suggestions(w: str):
    """
    Returns all suggestions provided by textblob lib.
    :param w: words suggestions are requested for
    :return: list of pairs <suggestion, frequency>. Note: we're not
        returning probability for it implies additional performance costs.
    """
    spelling = _get_spelling()

    lower_w = w.lower()
    all_suggestions = spelling._known([w])
    all_suggestions.update(spelling._known(spelling._edit1(lower_w)))
    all_suggestions.update(spelling._known(spelling._edit2(lower_w)))

    candidates = [(spelling.get(c, 0.0), c) for c in all_suggestions]
    candidates = sorted(((p, word) for p, word in candidates), reverse=True)
    if w.istitle():  # Preserve capitalization
        candidates = [(word.title(), p) for p, word in candidates]
    else:
        candidates = [(word, p) for p, word in candidates]
    return candidates
