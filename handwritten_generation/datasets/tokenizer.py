import torch

from collections.abc import Iterable

class CharLevelTokenizer:
    def __init__(
        self,
        char2idx: dict[str, int],
        idx2char: list[str], 
        pad: str = "<pad>"
    ):        
        self.char2idx = char2idx
        self.idx2char = idx2char
        
        if pad not in self.char2idx:
            self.char2idx[pad] = len(self.char2idx)
            self.idx2char.append(pad)            
        
        self.pad = pad
        self.pad_idx = self.char2idx[self.pad]

    def __len__(self) -> int:
        return len(self.idx2char)

    def encode_sequence(self, seq: str) -> list[int]:
        encoded_seq = []
        for char in seq:
            # Skip unknown characters
            if char in self.char2idx:
                encoded_seq.append(self.char2idx[char])

        return encoded_seq

    def encode(self, seq_batch: str | Iterable[str]) -> list[list[int]]:
        if isinstance(seq_batch, str):
            batch = [seq_batch]
        else:
            batch = seq_batch

        encoded_batch = []
        for seq in batch:
            encoded_batch.append(self.encode_sequence(seq))

        return encoded_batch

    def decode_sequence(self, seq: Iterable[int]) -> str:
        decoded_seq = []
        if isinstance(seq, torch.Tensor):
            for idx in seq:
                decoded_seq.append(self.idx2char[idx.item()])
        else:
            for idx in seq:
                decoded_seq.append(self.idx2char[idx])

        return ''.join(decoded_seq)

    def decode(self, seq_batch: list[list[int]] | list[int]) -> list[str]:
        if isinstance(seq_batch[0], int):
            print("list 1-d")
            batch = [seq_batch]
        elif isinstance(seq_batch, torch.Tensor) and len(seq_batch.size()) == 1:
            print("torch 1-d")
            batch = [seq_batch]
        else:
            print("multidim")
            batch = seq_batch

        decoded_batch = []
        for seq in batch:
            decoded_batch.append(self.decode_sequence(seq))

        return decoded_batch

def build_tokenizer(data: Iterable[str], pad: str = "<pad>", pad_idx: int = 0) -> CharLevelTokenizer:
    char2idx = {pad: pad_idx}
    for seq in data:
        for char in seq:
            if char not in char2idx:
                char2idx[char] = len(char2idx)

    idx2char = [None,] * len(char2idx)
    for char, idx in char2idx.items():
        idx2char[idx] = char

    return CharLevelTokenizer(char2idx=char2idx, idx2char=idx2char, pad=pad)