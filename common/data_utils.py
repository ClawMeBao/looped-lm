"""
common/data_utils.py — Dataset utilities dùng chung cho tất cả các phase.
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset, Subset


class TokenizedTextDataset(Dataset):
    """Tokenize text thành fixed-length blocks để train/eval LM."""

    def __init__(self, texts: list[str], tokenizer, max_length: int = 512):
        all_ids = []
        for text in texts:
            ids = tokenizer.encode(text, add_special_tokens=False)
            all_ids.extend(ids)
            all_ids.append(tokenizer.eos_token_id or 0)

        self.blocks = [
            all_ids[i : i + max_length]
            for i in range(0, len(all_ids) - max_length, max_length)
        ]
        self.max_length = max_length

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, idx):
        return torch.tensor(self.blocks[idx], dtype=torch.long)


def load_text_dataset(
    tokenizer,
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    split: str = "validation",
    max_length: int = 512,
    max_samples: Optional[int] = None,
    min_text_len: int = 100,
) -> Dataset:
    """
    Load một HuggingFace text dataset và tokenize thành blocks.

    Fallback sang dummy text nếu datasets không available.
    """
    from typing import Optional

    try:
        from datasets import load_dataset
        ds = load_dataset(dataset_name, dataset_config, split=split)
        texts = [x["text"] for x in ds if len(x["text"].strip()) >= min_text_len]
        print(f"[data] Loaded {dataset_name}/{dataset_config} ({split}): "
              f"{len(texts)} documents")
    except Exception as e:
        print(f"[data] ⚠️  Cannot load dataset ({e}). Using dummy text.")
        texts = [
            "The history of artificial intelligence began in antiquity. " * 60,
            "Machine learning is a type of artificial intelligence. " * 60,
            "Deep learning models consist of multiple layers of neurons. " * 60,
            "Natural language processing enables computers to understand text. " * 60,
            "Transformers have revolutionized the field of NLP. " * 60,
        ]

    dataset = TokenizedTextDataset(texts, tokenizer, max_length=max_length)

    if max_samples is not None and len(dataset) > max_samples:
        indices = torch.randperm(len(dataset))[:max_samples].tolist()
        dataset = Subset(dataset, indices)

    print(f"[data] Dataset size: {len(dataset)} blocks × {max_length} tokens")
    return dataset
