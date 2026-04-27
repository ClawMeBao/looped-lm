"""
common/data_utils.py — Dataset utilities dùng chung cho tất cả các phase.
"""

from __future__ import annotations

import re
from typing import Optional

import torch
from torch.utils.data import Dataset, Subset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks (GLM / Qwen3 reasoning) from text."""
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()


def _normalize_sharegpt(conversations: list[dict]) -> list[dict]:
    """Convert ShareGPT from/value format → standard role/content format."""
    role_map = {"human": "user", "gpt": "assistant", "system": "system"}
    return [
        {"role": role_map.get(m["from"], m["from"]), "content": m.get("value", "")}
        for m in conversations
    ]


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

def _get_assistant_token_ids(tokenizer):
    """
    Return (im_start_id, im_end_id, assistant_header_ids) for span detection.

    assistant_header_ids = token IDs for 'assistant\\n' (without im_start).
    These follow immediately after <|im_start|> in every assistant turn.
    """
    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end_id   = tokenizer.convert_tokens_to_ids("<|im_end|>")
    # Tokenize 'assistant\n' without special tokens to get the header suffix
    header_ids  = tokenizer("assistant\n", add_special_tokens=False)["input_ids"]
    return im_start_id, im_end_id, header_ids


def _find_assistant_spans(
    full_ids: list[int],
    im_start_id: int,
    im_end_id: int,
    header_ids: list[int],
    skip_think_wrapper: bool = False,
    think_id: int = 151667,
    end_think_id: int = 151668,
) -> list[tuple[int, int]]:
    """
    Scan full_ids for assistant turns and return (content_start, content_end) spans.

    Pattern searched:  [im_start_id] + header_ids + [content...] + [im_end_id]
    content_start: index of first content token (right after the header)
    content_end:   index right after <|im_end|> (exclusive, includes im_end itself)

    skip_think_wrapper=True: if the content starts with <think>...\n\n</think>\n\n,
    advance content_start past it (used with no_think=True to train on answer only).
    """
    spans: list[tuple[int, int]] = []
    h_len = len(header_ids)
    i = 0
    while i < len(full_ids):
        # Match <|im_start|>assistant\n
        if (full_ids[i] == im_start_id
                and i + h_len < len(full_ids)
                and full_ids[i + 1 : i + 1 + h_len] == header_ids):
            content_start = i + 1 + h_len

            # Optionally skip <think>...\n\n</think>\n\n boilerplate
            if skip_think_wrapper and content_start < len(full_ids) and full_ids[content_start] == think_id:
                # Scan for </think>
                j = content_start + 1
                while j < len(full_ids) and full_ids[j] != end_think_id:
                    j += 1
                if j < len(full_ids):  # found </think>
                    j += 1  # skip </think>
                    # skip trailing whitespace tokens (\n, \n\n etc.)
                    while j < len(full_ids) and full_ids[j] not in (im_end_id, im_start_id) and full_ids[j] in (198, 271, 148, 220):
                        j += 1
                    content_start = j

            # Walk forward to find matching <|im_end|>
            j = content_start
            while j < len(full_ids) and full_ids[j] != im_end_id:
                j += 1
            # Include <|im_end|> in the span
            content_end = j + 1 if j < len(full_ids) else j
            spans.append((content_start, content_end))
            i = content_end
        else:
            i += 1
    return spans


def _build_chat_example(
    tokenizer,
    messages: list[dict],
    max_length: int,
    no_think: bool = False,
) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
    """
    Tokenize một conversation với label mask: loss chỉ tính trên assistant tokens.

    Span detection dùng token-ID pattern search (<|im_start|>assistant\\n … <|im_end|>)
    thay vì re-tokenize prefix/end riêng lẻ — tránh BPE boundary mismatch.

    no_think=True: strip <think>…</think> từ content trước khi render template.

    Returns (input_ids, labels) tensors or None nếu không có assistant token.
    """
    template_kwargs: dict = {}
    if no_think:
        template_kwargs["enable_thinking"] = False

    # Normalise messages: strip unknown fields, inject reasoning or strip think
    clean_msgs = []
    for m in messages:
        content = m.get("content", "") or ""
        role    = m["role"]
        if role == "assistant":
            if no_think:
                content = _strip_think_tags(content)
            elif m.get("reasoning"):
                content = f"<think>\n{m['reasoning']}\n</think>\n\n{content}"
        clean_msgs.append({"role": role, "content": content})

    # Tokenize full sequence once via apply_chat_template
    try:
        result = tokenizer.apply_chat_template(
            clean_msgs,
            tokenize=True,
            add_generation_prompt=False,
            add_special_tokens=False,
            **template_kwargs,
        )
    except TypeError:
        # Older transformers: no add_special_tokens kwarg
        try:
            result = tokenizer.apply_chat_template(
                clean_msgs, tokenize=True, add_generation_prompt=False, **template_kwargs
            )
        except Exception:
            result = tokenizer.apply_chat_template(
                clean_msgs, tokenize=True, add_generation_prompt=False
            )

    # apply_chat_template may return a BatchEncoding/dict or a plain list
    if hasattr(result, "input_ids"):
        full_ids: list[int] = list(result["input_ids"])
    elif isinstance(result, dict):
        full_ids = list(result["input_ids"])
    else:
        full_ids = list(result)

    # Truncate
    full_ids = full_ids[:max_length]
    labels   = [-100] * len(full_ids)

    # Find and mark assistant spans using token-ID pattern search
    im_start_id, im_end_id, header_ids = _get_assistant_token_ids(tokenizer)
    spans = _find_assistant_spans(
        full_ids, im_start_id, im_end_id, header_ids,
        skip_think_wrapper=no_think,
    )

    for start, end in spans:
        end = min(end, max_length)
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

    __init__ only filters and stores raw messages — no tokenization.
    Tokenization happens lazily in __getitem__, making it safe to use
    with multi-worker DataLoaders (num_workers > 0).

    Each example returns:
        {"input_ids": Tensor[max_length], "labels": Tensor[max_length]}

    input_ids is right-padded with pad_token_id; labels with -100.
    """

    def __init__(
        self,
        tokenizer,
        messages_list: list[list[dict]],
        max_length: int = 512,
        no_think: bool = False,
    ):
        self.tokenizer    = tokenizer
        self.max_length   = max_length
        self.no_think     = no_think
        self.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0

        # Cheap pre-filter: keep only conversations with at least one assistant turn.
        # Full validity is checked lazily in __getitem__.
        skipped = 0
        self.messages_list: list[list[dict]] = []
        for messages in messages_list:
            has_assistant = any(
                m.get("role") == "assistant" or m.get("from") == "gpt"
                for m in messages
            )
            if has_assistant:
                self.messages_list.append(messages)
            else:
                skipped += 1

        if skipped:
            print(f"[data] Skipped {skipped} examples (no assistant turns)")

    def __len__(self) -> int:
        return len(self.messages_list)

    def __getitem__(self, idx: int) -> dict:
        result = _build_chat_example(
            self.tokenizer, self.messages_list[idx], self.max_length, no_think=self.no_think
        )
        if result is None:
            # Edge case: assistant tokens all truncated — return all-pad sample
            # (loss mask is all -100, so this sample contributes 0 gradient)
            input_ids = torch.full((self.max_length,), self.pad_token_id, dtype=torch.long)
            labels    = torch.full((self.max_length,), -100,              dtype=torch.long)
        else:
            input_ids, labels = result
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
    split: str = "train",
    max_length: int = 512,
    max_samples: Optional[int] = None,
    no_think: bool = False,
) -> Dataset:
    """
    Load instruction-following dataset và build ChatDataset với chat template.

    Dataset mặc định: Roman1111111/claude-opus-4.6-10000x
      - Columns: messages (list[dict]), metadata
      - Assistant messages có extra field 'reasoning' (chain-of-thought)

    Caller chịu trách nhiệm split train/eval/test (dùng random_split).

    no_think=True: không inject reasoning → train on answer-only.
    """
    from datasets import load_dataset

    ds = load_dataset(dataset_name, split=split)
    messages_list: list[list[dict]] = [row["messages"] for row in ds]

    if max_samples is not None:
        messages_list = messages_list[:max_samples]

    think_tag = " [no_think]" if no_think else " [think]"
    print(f"[data] Loaded {dataset_name} ({split}){think_tag}: {len(messages_list)} conversations")
    dataset = ChatDataset(tokenizer, messages_list, max_length=max_length, no_think=no_think)
    print(f"[data] ChatDataset: {len(dataset)} valid examples × {max_length} tokens")
    return dataset


def load_glm_dataset(
    tokenizer,
    dataset_name: str = "Jackrong/GLM-5.1-Reasoning-1M-Cleaned",
    split: str = "train",
    max_length: int = 512,
    max_samples: Optional[int] = None,
    no_think: bool = False,
    local_dir: str = "data/glm_dataset",
) -> Dataset:
    """
    Load GLM-5.1 reasoning dataset (ShareGPT format) và build ChatDataset.

    Dataset: Jackrong/GLM-5.1-Reasoning-1M-Cleaned (~1M rows, single train split)
      - Columns: id, conversations, input, output, domain, meta
      - conversations: [{from: human, value: ...}, {from: gpt, value: <think>…</think>\\n\\nanswer}]

    Caching: lần đầu tải từ HuggingFace và lưu vào local_dir (Arrow format).
    Lần sau load từ local_dir nếu tồn tại → không cần internet.

    Khuyến nghị: dùng --max_samples để giới hạn với 1M-row dataset.
    Caller chịu trách nhiệm split train/eval/test.

    no_think=True: strip <think>...</think> từ gpt responses.
    """
    import os
    from datasets import load_dataset, load_from_disk

    def _is_dataset_dir(path: str) -> bool:
        """Check nếu path là valid Arrow Dataset dir (có dataset_info.json)."""
        return os.path.isfile(os.path.join(path, "dataset_info.json"))

    # Thử load theo thứ tự ưu tiên:
    #   1. local_dir trực tiếp  (user save thẳng vào thư mục)
    #   2. local_dir/split      (code tự tạo khi download)
    local_split_dir = os.path.join(local_dir, split)
    if _is_dataset_dir(local_dir):
        print(f"[data] Loading GLM dataset from local cache: {local_dir}")
        ds = load_from_disk(local_dir)
    elif _is_dataset_dir(local_split_dir):
        print(f"[data] Loading GLM dataset from local cache: {local_split_dir}")
        ds = load_from_disk(local_split_dir)
    else:
        print(f"[data] Downloading {dataset_name} from HuggingFace …")
        os.makedirs(local_split_dir, exist_ok=True)
        ds = load_dataset(dataset_name, split=split)
        print(f"[data] Saving to local cache: {local_split_dir}")
        ds.save_to_disk(local_split_dir)
        print(f"[data] ✓ Saved {len(ds)} rows → {local_split_dir}")

    messages_list: list[list[dict]] = [
        _normalize_sharegpt(row["conversations"]) for row in ds
    ]

    if max_samples is not None:
        messages_list = messages_list[:max_samples]

    think_tag = " [no_think]" if no_think else " [think]"
    print(f"[data] Loaded {dataset_name} ({split}){think_tag}: {len(messages_list)} conversations")
    dataset = ChatDataset(tokenizer, messages_list, max_length=max_length, no_think=no_think)
    print(f"[data] ChatDataset: {len(dataset)} valid examples × {max_length} tokens")
    return dataset
