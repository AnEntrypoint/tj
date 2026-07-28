"""HuggingFace dataset loader for negative eval data.

Loads public coding datasets (the-stack, codeparrot, etc.) as negative
training examples. Negative eval data is used for contrastive training:
the model learns to distinguish Claude Code agent behavior from generic
public code.

Datasets are streamed to avoid loading entire corpora into memory.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterator, List, Optional

logger = logging.getLogger(__name__)

# Well-known public coding datasets suitable for negative eval.
# Each entry: (hf_path, default_split, description)
KNOWN_DATASETS = {
    "the-stack": ("bigcode/the-stack", "train", "The Stack v2 — permissive code"),
    "starcoder": ("bigcode/starcoderdata", "train", "StarCoder training data"),
    "codeparrot": ("codeparrot/codeparrot-clean", "train", "CodeParrot cleaned"),
    "github-code": ("codeparrot/github-code", "train", "GitHub code corpus"),
    "stackoverflow": ("wikitext", "train", "WikiText (general text negative)"),
}


@dataclass
class HFDatasetConfig:
    """Configuration for a HuggingFace dataset source."""
    path: str = ""                     # HF dataset path, e.g. "bigcode/the-stack"
    split: str = "train"              # dataset split
    name: Optional[str] = None         # subset/config name
    streaming: bool = True             # stream dataset (don't load into memory)
    max_samples: int = 0               # max samples to load (0 = unlimited)
    text_field: str = "content"        # field containing the text content
    filter_lang: Optional[str] = None  # filter by programming language
    sample_ratio: float = 0.3          # ratio of negative samples in mixed batch


def _import_datasets():
    """Lazy import datasets library."""
    try:
        from datasets import load_dataset as _load
        return _load
    except ImportError:
        raise ImportError(
            "HuggingFace datasets library required. Install with: "
            "pip install datasets"
        )


def _known_dataset(name: str) -> HFDatasetConfig:
    """Resolve a known dataset name to its config."""
    if name in KNOWN_DATASETS:
        path, split, desc = KNOWN_DATASETS[name]
        return HFDatasetConfig(path=path, split=split)
    return HFDatasetConfig(path=name, split="train")


def iter_hf_texts(
    dataset: str,
    split: str = "train",
    name: Optional[str] = None,
    streaming: bool = True,
    max_samples: int = 0,
    text_field: str = "content",
    filter_lang: Optional[str] = None,
) -> Iterator[str]:
    """Iterate over text samples from a HuggingFace dataset.

    Yields one text string per sample. The dataset is streamed by default
    to avoid loading the entire corpus into memory.

    Args:
        dataset: HF dataset path or known dataset name.
        split: Dataset split name.
        name: Subset/config name.
        streaming: Stream the dataset (recommended for large datasets).
        max_samples: Stop after this many samples (0 = unlimited).
        text_field: Dataset field containing the text.
        filter_lang: Only include samples matching this language.

    Yields:
        Text strings from the dataset.
    """
    _load = _import_datasets()
    kwargs = {"path": dataset, "split": split, "streaming": streaming}
    if name:
        kwargs["name"] = name

    try:
        ds = _load(**kwargs)
    except Exception as e:
        logger.warning(f"Failed to load dataset {dataset}: {e}")
        return

    count = 0
    for sample in ds:
        if max_samples and count >= max_samples:
            break
        if filter_lang and sample.get("lang") != filter_lang:
            continue
        text = sample.get(text_field)
        if text and isinstance(text, str) and len(text.strip()) > 10:
            yield text
            count += 1


def iter_mixed_batches(
    positive_iter: Iterator[str],
    negative_dataset: str,
    negative_config: Optional[HFDatasetConfig] = None,
    batch_size: int = 32,
    sample_ratio: float = 0.3,
) -> Iterator[tuple[List[str], List[str]]]:
    """Yield mixed batches of positive and negative samples.

    Each batch contains both positive (Claude Code) and negative (public)
    samples, interleaved according to ``sample_ratio``.

    Args:
        positive_iter: Iterator over positive samples (ccsniff data).
        negative_dataset: HF dataset name or path for negative samples.
        negative_config: Optional config overriding defaults.
        batch_size: Total batch size.
        sample_ratio: Fraction of negative samples per batch.

    Yields:
        Tuple of (positive_texts, negative_texts) per batch.
    """
    if negative_config is None:
        negative_config = _known_dataset(negative_dataset)

    neg_iter = iter_hf_texts(
        dataset=negative_config.path,
        split=negative_config.split,
        name=negative_config.name,
        streaming=negative_config.streaming,
        text_field=negative_config.text_field,
        filter_lang=negative_config.filter_lang,
    )

    n_neg = max(1, int(batch_size * sample_ratio))
    n_pos = batch_size - n_neg

    pos_buffer: List[str] = []
    neg_buffer: List[str] = []

    while True:
        # Fill positive buffer
        while len(pos_buffer) < n_pos:
            try:
                pos_buffer.append(next(positive_iter))
            except StopIteration:
                break
        # Fill negative buffer
        while len(neg_buffer) < n_neg:
            try:
                neg_buffer.append(next(neg_iter))
            except StopIteration:
                # Restart negative iterator for infinite streaming
                neg_iter = iter_hf_texts(
                    dataset=negative_config.path,
                    split=negative_config.split,
                    name=negative_config.name,
                    streaming=negative_config.streaming,
                    text_field=negative_config.text_field,
                    filter_lang=negative_config.filter_lang,
                )
                try:
                    neg_buffer.append(next(neg_iter))
                except StopIteration:
                    break

        if len(pos_buffer) < n_pos and len(neg_buffer) < n_neg:
            break

        pos_batch = pos_buffer[:n_pos]
        neg_batch = neg_buffer[:n_neg]
        pos_buffer = pos_buffer[n_pos:]
        neg_buffer = neg_buffer[n_neg:]

        if pos_batch or neg_batch:
            yield pos_batch, neg_batch


def list_known_datasets() -> List[str]:
    """Return list of known dataset names."""
    return list(KNOWN_DATASETS.keys())