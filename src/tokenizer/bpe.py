"""
bpe.py — BPE Tokenizer 加载与编解码

封装 HuggingFace `tokenizers` 库（包名 tokenizers，与本目录 src/tokenizer 不同）。
"""

from __future__ import annotations

from pathlib import Path

try:
    from tokenizers import Tokenizer as HFTokenizer
except ImportError as exc:
    raise ImportError(
        "未安装 HuggingFace tokenizers 库。请运行: pip install tokenizers"
    ) from exc


class ProjectTokenizer:
    """Thin wrapper around tokenizers.Tokenizer."""

    def __init__(
        self,
        tokenizer: HFTokenizer,
        bos: str = "<s>",
        eos: str = "</s>",
        pad: str = "<pad>",
    ) -> None:
        self._tok = tokenizer
        self.bos = bos
        self.eos = eos
        self.pad = pad
        self.bos_id = tokenizer.token_to_id(bos)
        self.eos_id = tokenizer.token_to_id(eos)
        self.pad_id = tokenizer.token_to_id(pad)
        self.vocab_size = tokenizer.get_vocab_size()

    @classmethod
    def load(cls, tokenizer_dir: str | Path) -> ProjectTokenizer:
        path = Path(tokenizer_dir) / "tokenizer.json"
        if not path.exists():
            raise FileNotFoundError(f"Tokenizer not found: {path}. Run train_bpe.py first.")
        return cls(HFTokenizer.from_file(str(path)))

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = True) -> list[int]:
        ids = self._tok.encode(text).ids
        if add_bos and self.bos_id is not None:
            ids = [self.bos_id] + ids
        if add_eos and self.eos_id is not None:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        return self._tok.decode(ids, skip_special_tokens=skip_special)

    def encode_batch(self, texts: list[str], add_eos: bool = True) -> list[list[int]]:
        return [self.encode(t, add_eos=add_eos) for t in texts]
