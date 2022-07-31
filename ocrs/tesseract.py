import dataclasses
from enum import Enum
from os import environ
from pathlib import Path
from typing import Optional

import pytesseract


# Environment variabe that defines tesseract language ('-l' parameter) for be
# used during confusion matrix generation
TESS_LANG_DEFAULT = "eng"
TESS_LANG_ENV_VAR = "TESS_LANG"


class TesseractPSM(Enum):
    """
    Tesseract page segmentation modes, corresponds to --psm value
    of 'tesseract' CLI command.
    Run 'tesseract --help-psm' for detailed description.
    """
    OSDOnly = 0
    AutoWithOSD = 1
    AutoWithoutOSD = 3
    SingleColumnVariableSize = 4
    SingleUniformBlockVerticalAlignment = 5
    SingleUniformBlock = 6
    SingleTextLine = 7
    SingleWord = 8
    SingleWordInCircle = 9
    SingleCharacter = 10
    SparseWithoutOCR = 11
    SparseWithOCR = 12
    RAWLine = 13


def _get_tess_lang():
    lang = environ.get(TESS_LANG_ENV_VAR)
    return lang or TESS_LANG_DEFAULT


@dataclasses.dataclass
class Tesseract:
    """
    Tesseract OCR recognizer.
    Actually just a thin wrapper for pytesseract.
    """
    model: str = dataclasses.field(default_factory=_get_tess_lang)
    psm: Optional[TesseractPSM] = TesseractPSM.RAWLine

    def __call__(self, line_image_path: Path) -> str:
        return pytesseract.image_to_string(
            str(line_image_path), self.model,
            config=f"--psm {self.psm.value}" if self.psm else ""
        )
