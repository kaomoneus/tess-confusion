import logging
import multiprocessing
from collections import defaultdict
from typing import Dict, Iterable, Tuple

from utils.needleman_wunsch import nw


LOG = logging.getLogger(__name__)


CharFrequencies = Dict[str, float]
ConfusionMatrix = Dict[str, CharFrequencies]


def create_confusion(
    recognition_and_gt: Iterable[Tuple[str, str]]
) -> ConfusionMatrix:

    confusion_matrix: ConfusionMatrix = defaultdict(
        lambda: defaultdict(lambda: 0)
    )

    for recognized, gt_str in recognition_and_gt:
        recognized_aligned, gt_aligned = nw(recognized, gt_str)

        # Get confusion vector

        confusion_vec = dict()
        left = []
        right = []

        for rec_char, gt_char in zip(recognized_aligned, gt_aligned):
            if rec_char != " ":
                left.extend(rec_char)

            if gt_char != " ":
                right.extend(gt_char)

            if left and right:
                assert len(left) == 1 or len(right) == 1, \
                    "We can map only single chars to multiple chars or" \
                    " multiple chars to single char"

                confusion_vec["".join(left)] = "".join(right)
                left, right = [], []

        # Aggregate it

        for rec, correct in confusion_vec.items():
            confusion_matrix[rec][correct] += 1

    # Normalize confusion matrix
    for rec, vec in confusion_matrix.items():
        total = sum([freq for freq in vec.values()])
        for key, freq in vec.items():
            vec[key] = freq / total

    return confusion_matrix
