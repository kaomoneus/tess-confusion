"""
Needleman-Wunsch implementation.
Originally taken (and slightly modified) from
https://github.com/shaneweisz/OCR-Character-Confusion/blob/master/confusion_matrix/needleman_wunsch.py
"""


# Use these values to calculate scores
import dataclasses
from typing import Tuple, List, Optional, Any

gap_penalty = -1
match_award = 1
mismatch_penalty = -1

GAP_RENDER_CHAR = " "
GAP_ITEM = "\0"

# A function for making a matrix of zeroes


def _zeros(rows, cols):
    # Define an empty list
    retval = []
    # Set up the rows of the matrix
    for x in range(rows):
        # For each row, add an empty list
        retval.append([])
        # Set up the columns in each row
        for y in range(cols):
            # Add a zero to each column in each row
            retval[-1].append(0)
    # Return the matrix of zeros
    return retval


# A function for determining the score between any two bases in alignment
def _match_score(alpha, beta):
    if alpha == beta:
        return match_award
    elif alpha == ' ' or beta == ' ':
        return gap_penalty
    else:
        return mismatch_penalty

# The function that actually fills out a matrix of scores


@dataclasses.dataclass
class AlignedChars:
    char_items: List[Any] = dataclasses.field(default_factory=list)

    def __str__(self):
        return "".join([c if c is not GAP_ITEM else GAP_RENDER_CHAR for c in self.char_items])


def nw(seq1, seq2) -> Tuple[AlignedChars, AlignedChars]:
    def grow_subchain(aligned_raw: AlignedChars, c):
        if not aligned_raw.char_items:
            aligned_raw.char_items.append(c)
        else:
            aligned_raw.char_items = [c] + aligned_raw.char_items

    # Store length of two sequences
    n = len(seq1)
    m = len(seq2)

    # Generate matrix of zeros to store scores
    score = _zeros(m + 1, n + 1)

    # Calculate score table

    # Fill out first column
    for i in range(0, m + 1):
        score[i][0] = gap_penalty * i

    # Fill out first row
    for j in range(0, n + 1):
        score[0][j] = gap_penalty * j

    # Fill out all other values in the score matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # Calculate the score by checking the top, left, and diagonal cells
            match = score[i - 1][j - 1] + _match_score(seq1[j - 1], seq2[i - 1])
            delete = score[i - 1][j] + gap_penalty
            insert = score[i][j - 1] + gap_penalty
            # Record the maximum score from the three possible scores calculated above
            score[i][j] = max(match, delete, insert)

    # Traceback and compute the alignment

    # Create variables to store alignment
    # FIXME: cover 'nw' with tests and remove align1 and align2,
    #    currently I keep it only because I'm not 100% tested by subchains
    #    enhancement.
    align1 = ""
    align2 = ""
    align1_raw = AlignedChars()
    align2_raw = AlignedChars()

    # Start from the bottom right cell in matrix
    i = m
    j = n

    # We'll use i and j to keep track of where we are in the matrix, just like above
    while i > 0 and j > 0:  # end touching the top or the left edge
        score_current = score[i][j]
        score_diagonal = score[i-1][j-1]
        score_up = score[i][j-1]
        score_left = score[i-1][j]

        # Check to figure out which cell the current score was calculated from,
        # then update i and j to correspond to that cell.
        if score_current == score_diagonal + _match_score(seq1[j - 1], seq2[i - 1]):
            align1 += seq1[j-1]
            align2 += seq2[i-1]
            grow_subchain(align1_raw, seq1[j - 1])
            grow_subchain(align2_raw, seq2[i - 1])
            i -= 1
            j -= 1
        elif score_current == score_up + gap_penalty:
            align1 += seq1[j-1]
            align2 += GAP_RENDER_CHAR
            grow_subchain(align1_raw, seq1[j - 1])
            grow_subchain(align2_raw, GAP_ITEM)

            j -= 1
        elif score_current == score_left + gap_penalty:
            align1 += GAP_RENDER_CHAR
            align2 += seq2[i-1]
            grow_subchain(align1_raw, GAP_ITEM)
            grow_subchain(align2_raw, seq2[i - 1])

            i -= 1

    # Finish tracing up to the top left cell
    while j > 0:
        align1 += seq1[j-1]
        align2 += GAP_RENDER_CHAR
        grow_subchain(align1_raw, seq1[j - 1])
        grow_subchain(align2_raw, GAP_ITEM)
        j -= 1
    while i > 0:
        align1 += GAP_RENDER_CHAR
        align2 += seq2[i-1]
        grow_subchain(align1_raw, GAP_ITEM)
        grow_subchain(align2_raw, seq2[i - 1])
        i -= 1

    # Since we traversed the score matrix from the bottom right, our two sequences will be reversed.
    # These two lines reverse the order of the characters in each sequence.
    align1 = align1[::-1]
    align2 = align2[::-1]

    assert str(align1_raw) == align1
    assert str(align2_raw) == align2
    assert len(align1_raw.char_items) == len(align2_raw.char_items)

    return align1_raw, align2_raw
