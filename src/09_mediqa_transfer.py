"""
09_mediqa_transfer.py
++++++++++++++++++++++++++++++++++++++++++
Transfer evaluation: tests MedNLI-trained models on MEDIQA NLI without
retraining, to assess generalisation beyond MedNLI.

NOTE: This evaluation could not be completed in this study. The MEDIQA
NLI test set requires credentialed PhysioNet access and a signed Data
Use Agreement which was not obtained within the study timeframe.
See Section 7.1 Limitation 3 of the paper for details.

To run this script, access needs to be obtained at:
    https://physionet.org/content/mednli/1.0.0/


Usage:
    python src/09_mediqa_transfer.py
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

MEDIQA_TEST_PATH = Path("data/mednli_test.jsonl")


def load_mediqa(path: Path):
    """Load MEDIQA NLI test set from PhysioNet download."""
    if not path.exists():
        raise FileNotFoundError(
            f"MEDIQA NLI test file not found at {path}. "
            "Obtain credentialed access at https://physionet.org/content/mednli/1.0.0/ "
            "and place the test file at the path above."
        )
    # TODO: implement loading once access is obtained
    raise NotImplementedError("Data loading not yet implemented.")


def main():
    logger.warning(
        "09_mediqa_transfer.py: MEDIQA NLI evaluation not completed. "
        "See script docstring and paper Section 7.1 Limitation."
    )
    load_mediqa(MEDIQA_TEST_PATH)


if __name__ == "__main__":
    main()