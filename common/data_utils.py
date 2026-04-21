"""
common/data_utils.py — Dataset utilities dùng chung cho tất cả các phase.
"""

from __future__ import annotations

from typing import Optional

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


# ---------------------------------------------------------------------------
# ChatDataset — instruction-following dataset with chat template + label mask
# ---------------------------------------------------------------------------

def _build_chat_example(
    tokenizer,
    messages: list[dict],
    max_length: int,
) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
    """
    Tokenize một conversation với label mask: loss chỉ tính trên assistant tokens.

    Với Qwen3 template, assistant turn có format:
        <|im_start|>assistant\n<think>\n{reasoning}\n</think>\n\n{answer}<|im_end|>

    Nếu message có field 'reasoning', inject vào content trước khi apply template.
    Unknown fields (e.g. 'reasoning', 'metadata') bị strip trước apply_chat_template.

    Returns (input_ids, labels) tensors or None nếu không có assistant token.
    """
    # Strip unknown fields; inject reasoning into content if present
    clean_msgs = []
    for m in messages:
        content = m.get("content", "") or ""
        if m.get("role") == "assistant" and m.get("reasoning"):
            content = f"<think>\n{m['reasoning']}\n</think>\n\n{content}"
        clean_msgs.append({"role": m["role"], "content": content})

    # Full sequence text
    full_text = tokenizer.apply_chat_template(
        clean_msgs, tokenize=False, add_generation_prompt=False
    )
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]

    # Truncate to max_length
    full_ids = full_ids[:max_length]
    labels = [-100] * len(full_ids)

    # For each assistant message: mark its tokens as loss targets
    for i, msg in enumerate(clean_msgs):
        if msg["role"] != "assistant":
            continue
        # Prefix ends at the start of this assistant's content (after im_start\nassistant\n)
        prefix_text = tokenizer.apply_chat_template(
            clean_msgs[:i], tokenize=False, add_generation_prompt=True
        )
        prefix_ids = tokenizer(prefix_text, add_special_tokens=False)["input_ids"]

        # End of this turn
        end_text = tokenizer.apply_chat_template(
            clean_msgs[: i + 1], tokenize=False, add_generation_prompt=False
        )
        end_ids = tokenizer(end_text, add_special_tokens=False)["input_ids"]

        start = len(prefix_ids)
        end   = min(len(end_ids), max_length)
        for j in range(start, end):
            labels[j] = full_ids[j]

    if all(lbl == -100 for lbl in labels):
        return None

    return (
        torch.tensor(full_ids, dtype=torch.long),
        torch.tensor(labels,   dtype=torch.long),
    )


class ChatDataset(Dataset):
    """
    Instruction-following dataset with per-turn label masking.

    Each example returns:
        {"input_ids": Tensor[max_length], "labels": Tensor[max_length]}

    input_ids is right-padded with pad_token_id; labels with -100.
    """

    def __init__(
        self,
        tokenizer,
        messages_list: list[list[dict]],
        max_length: int = 512,
    ):
        self.max_length    = max_length
        self.pad_token_id  = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
        self.examples: list[tuple[torch.Tensor, torch.Tensor]] = []

        skipped = 0
        for messages in messages_list:
            result = _build_chat_example(tokenizer, messages, max_length)
            if result is not None:
                self.examples.append(result)
            else:
                skipped += 1

        if skipped:
            print(f"[data] Skipped {skipped} examples (no assistant tokens or too long)")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        input_ids, labels = self.examples[idx]
        pad_len = self.max_length - len(input_ids)
        if pad_len > 0:
            input_ids = torch.cat([
                input_ids,
                torch.full((pad_len,), self.pad_token_id, dtype=torch.long),
            ])
            labels = torch.cat([
                labels,
                torch.full((pad_len,), -100, dtype=torch.long),
            ])
        return {"input_ids": input_ids, "labels": labels}


def load_instruction_dataset(
    tokenizer,
    dataset_name: str = "Roman1111111/claude-opus-4.6-10000x",
    split: str = "train[:-500]",
    max_length: int = 512,
    max_samples: Optional[int] = None,
) -> Dataset:
    """
    Load instruction-following dataset và build ChatDataset với chat template.

    Dataset mặc định: Roman1111111/claude-opus-4.6-10000x
      - Columns: messages (list[dict]), metadata
      - Assistant messages có extra field 'reasoning' (chain-of-thought)
      - Format: [system, user, assistant{content, reasoning}]

    split hỗ trợ HuggingFace slice syntax:
      - "train[:-500]"   → training split (bỏ 500 cuối làm eval)
      - "train[-500:]"   → eval split
    """
    from datasets import load_dataset

    ds = load_dataset(dataset_name, split=split)
    messages_list: list[list[dict]] = [row["messages"] for row in ds]

    if max_samples is not None:
        messages_list = messages_list[:max_samples]

    print(f"[data] Loaded {dataset_name} ({split}): {len(messages_list)} conversations")
    dataset = ChatDataset(tokenizer, messages_list, max_length=max_length)
    print(f"[data] ChatDataset: {len(dataset)} valid examples × {max_length} tokens")
    return dataset
