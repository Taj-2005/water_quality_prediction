"""sklearn preprocessing pipeline: imputation, scaling, missingness flags.

Owner: Parthraj Singh Bhati. Fitted on the TRAINING SET ONLY, then applied to validation and test.
Returning a Pipeline (rather than transformed arrays) is what keeps that guarantee
mechanical instead of a thing people remember to do.
"""

from __future__ import annotations


def build_pipeline():
    raise NotImplementedError
