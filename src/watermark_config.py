"""Shared SynthID watermark configuration and processor factory.

Single source of truth for the watermark keys and sampling parameters used
across generation, detection and training.
"""

WATERMARK_KEYS = [101, 202, 303, 404, 505, 606, 707, 808, 909]
SAMPLING_TABLE_SIZE = 2 ** 16
SAMPLING_TABLE_SEED = 0
CONTEXT_HISTORY_SIZE = 1024


def make_synthid_processor(ngram_len, device):
    """Build a SynthIDTextWatermarkLogitsProcessor with the project's fixed config."""
    from transformers import SynthIDTextWatermarkLogitsProcessor

    return SynthIDTextWatermarkLogitsProcessor(
        keys=WATERMARK_KEYS,
        ngram_len=ngram_len,
        sampling_table_size=SAMPLING_TABLE_SIZE,
        sampling_table_seed=SAMPLING_TABLE_SEED,
        context_history_size=CONTEXT_HISTORY_SIZE,
        device=device,
    )
