"""
core/errors.py
==============
E4 — error handling: typed exceptions + input validation.

These let the pipeline fail *gracefully and informatively* instead of crashing
with a stack trace or silently producing an empty report. Each maps to a clear,
user-facing message at the CLI layer.
"""


class AletheonError(Exception):
    """Base class for all expected, handled Aletheon errors."""


class InvalidDrugName(AletheonError):
    """The drug name is empty, too short, or obviously not a drug query."""


class NoEvidenceFound(AletheonError):
    """Every source returned nothing for this drug (likely misspelled / unknown)."""


class PipelineError(AletheonError):
    """A core step (embed / index / retrieve / generate) failed unrecoverably."""


# Basic sanity checks on a drug name. We keep this permissive (real drug names
# are weird) but catch the obvious garbage: empty, whitespace, single char,
# pure punctuation/numbers, or absurdly long input.
def validate_drug_name(drug: str) -> str:
    if drug is None:
        raise InvalidDrugName("No drug name provided.")
    cleaned = drug.strip()
    if len(cleaned) < 2:
        raise InvalidDrugName(
            f"Drug name {drug!r} is too short — please enter a real drug name.")
    if len(cleaned) > 100:
        raise InvalidDrugName("Drug name is unreasonably long — please check it.")
    # must contain at least one letter (drug names are not pure numbers/symbols)
    if not any(ch.isalpha() for ch in cleaned):
        raise InvalidDrugName(
            f"{drug!r} doesn't look like a drug name (no letters).")
    return cleaned
