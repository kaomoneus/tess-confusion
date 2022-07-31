import json
import logging
import multiprocessing
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, Callable

import click
from tqdm import tqdm

from ocrs.tesseract import Tesseract
from utils.confusion import create_confusion

LOG = logging.getLogger(__name__)

KNOWN_IMAGE_SUFFIXES = {".png", ".tiff", ".tif", ".jpg"}

OCRType = Callable[[Path], str]
OCRS: Dict[str, OCRType] = dict(
    tesseract=Tesseract()
)


def _get_image_for(item_path: Path):
    images = {
        p
        for p in map(
            lambda s: item_path.with_suffix(s),
            KNOWN_IMAGE_SUFFIXES
        ) if p.exists()
    }

    if not images:
        return None

    return next(iter(images))


tess_lock = multiprocessing.Lock()


def _recognize(ocr_lstmf_path: Path):
    ocr, lstmf_path = ocr_lstmf_path
    img_path = _get_image_for(lstmf_path)
    with tess_lock:
        return ocr(img_path) if img_path else None


def _gt(lstmf_path):
    gt_path: Path = lstmf_path.with_suffix(".gt.txt")
    if not gt_path.exists():
        return None
    with open(gt_path, "r") as gt_f:
        return gt_f.readline().strip()


def _check_rec_gt(recognized, gt, lstmf_path):
    if recognized is None:
        LOG.warning(f"Unable to find images for {lstmf_path}")
        return False
    if gt is None:
        LOG.warning(f"Unable to find ground-truth for {lstmf_path}")
        return False
    return True


def _get_rec_and_gt(items_list_path, ocr: OCRType, limit: int, jobs=None):
    LOG.info("Reading list file...")
    with open(items_list_path, "r") as images_list_file:
        lines = [*map(Path, images_list_file)]

    if limit:
        lines = lines[:limit]

    ocr = [ocr]*len(lines)
    ocr_lines = [*zip(ocr, lines)]

    if not jobs:
        mapf = map
        desc = "Processing images"
    else:
        # FIXME: when using parallel mode, we get broken confusion matrix.
        pool = Pool(processes=jobs)
        mapf = pool.imap_unordered
        desc = f"Processing images, jobs={jobs}"

    for recognized, lstmf_path in tqdm(
        zip(mapf(_recognize, ocr_lines), lines),
        total=len(lines), desc=desc
    ):
        gt_str = _gt(lstmf_path)
        if not _check_rec_gt(recognized, gt_str, lstmf_path):
            continue

        yield recognized, gt_str


@click.command()
@click.option(
    "--ocr",
    default="tesseract",
    type=click.Choice(list(OCRS.keys()), case_sensitive=False)
)
@click.option(
    "--items-list-path",
    required=True,
    type=Path
)
@click.option(
    "--limit",
    type=int
)
@click.option(
    "--output",
    default=Path("confusion_matrix.json"),
    type=Path
)
def make(ocr: str, items_list_path: Path, output: Path, limit: int):
    ocr = OCRS[ocr]

    confusion_matrix = create_confusion(
        _get_rec_and_gt(items_list_path, ocr, limit)
    )

    LOG.info("Writing confusion matrix...")
    with open(output, "w") as confusion_matrix_file:
        json.dump(confusion_matrix, confusion_matrix_file, indent=4)

    LOG.info(f"Done. Matrix is written into '{output}'")