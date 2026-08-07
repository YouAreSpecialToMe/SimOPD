"""a2's cold-start SFT dataset: render once, loss-mask through the empty think block.

verl's MultiTurnSFTDataset tokenizes each turn separately and asserts the pieces
concatenate to the whole-conversation render. The Qwen3-Base template fails that
assertion on every row: the per-turn assistant render has no think block, while the
full render inserts the empty `<think>\n\n</think>\n\n` scaffold in the final
assistant turn (audit 2026-08-07 C2, reproduced with the real tokenizer). The
obvious escape hatch -- ignore_input_ids_mismatch=True -- runs, but trains
P(answer | header) while the OPD phase and every eval condition on
P(answer | header + empty block): a silent conditioning shift on the protocol's
signature scaffold.

So this class does what build_prefixes does for i1: measure, don't assume. Each row
is rendered ONCE with the full conversation under enable_thinking=False; the
supervision boundary is the generation prefix (same kwargs), asserted per row to be
a strict prefix of the full render -- the empty think block therefore sits inside
the UNsupervised span, exactly where verl's rollout puts it at OPD time. Ids go
through verl's own normalize_token_ids, the gen_priv_cot lesson (a bare
apply_chat_template(tokenize=True) can hand back an Encoding whose len() is its
key count).
"""

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class ColdstartSFTDataset(Dataset):
    def __init__(self, parquet_files, tokenizer, config, processor=None, max_samples=-1):
        from verl.utils.tokenizer.tokenizer import normalize_token_ids  # verl's, by design

        self._norm = normalize_token_ids
        self.tokenizer = tokenizer
        if isinstance(parquet_files, str):
            parquet_files = [parquet_files]
        self.df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
        if max_samples and max_samples > 0:
            self.df = self.df.iloc[:max_samples]
        self.messages_key = config.get("messages_key", "messages")
        self.max_length = int(config.get("max_length", 17408))
        self.truncation = config.get("truncation", "error")
        pad_mode = str(config.get("pad_mode", "right")).lower()
        assert "right" in pad_mode or pad_mode == "datasetpadmode.right", (
            f"ColdstartSFTDataset implements right-padding only, got pad_mode={pad_mode!r}"
        )

    def __len__(self):
        return len(self.df)

    def _ids(self, msgs, add_generation_prompt):
        return list(
            self._norm(
                self.tokenizer.apply_chat_template(
                    msgs,
                    add_generation_prompt=add_generation_prompt,
                    tokenize=True,
                    enable_thinking=False,
                )
            )
        )

    def __getitem__(self, item):
        msgs = list(self.df.iloc[item][self.messages_key])
        msgs = [dict(m) for m in msgs]
        assert msgs and msgs[-1].get("role") == "assistant", (
            f"row {item}: expected a trailing assistant turn, got roles "
            f"{[m.get('role') for m in msgs]}"
        )
        prefix = self._ids(msgs[:-1], add_generation_prompt=True)
        full = self._ids(msgs, add_generation_prompt=False)
        # The faithfulness gate: supervised span = full minus the OPD-time prefix.
        # If the template ever stops rendering the generation prompt (incl. the
        # empty think block) as a prefix of the full conversation, this must fail
        # loudly rather than shift the conditioning silently.
        if full[: len(prefix)] != prefix:
            raise RuntimeError(
                f"row {item}: generation prefix is not a prefix of the full render; "
                "the template changed and the loss boundary is undefined"
            )
        input_ids = torch.tensor(full, dtype=torch.long)
        loss_mask = torch.zeros(len(full), dtype=torch.long)
        loss_mask[len(prefix):] = 1
        attention_mask = torch.ones(len(full), dtype=torch.long)

        n = input_ids.shape[0]
        if n > self.max_length:
            if self.truncation == "error":
                raise ValueError(f"sequence_length={n} is larger than max_length={self.max_length}")
            if self.truncation == "right":
                input_ids = input_ids[: self.max_length]
                loss_mask = loss_mask[: self.max_length]
                attention_mask = attention_mask[: self.max_length]
            elif self.truncation == "left":
                input_ids = input_ids[-self.max_length:]
                loss_mask = loss_mask[-self.max_length:]
                attention_mask = attention_mask[-self.max_length:]
            else:
                raise ValueError(f"Unknown truncation method {self.truncation}")
            n = self.max_length
        position_ids = torch.arange(n, dtype=torch.long)
        if n < self.max_length:
            pad = self.max_length - n
            pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            input_ids = torch.cat((input_ids, torch.full((pad,), pad_id, dtype=torch.long)))
            loss_mask = torch.cat((loss_mask, torch.zeros(pad, dtype=torch.long)))
            attention_mask = torch.cat((attention_mask, torch.zeros(pad, dtype=torch.long)))
            position_ids = F.pad(position_ids, (0, pad), value=0)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
        }
