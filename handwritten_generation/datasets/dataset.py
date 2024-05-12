import pandas as pd

import torch

from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as TF2
from torch.utils.data import Dataset, DataLoader

from PIL import Image
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

from handwritten_generation.datasets.tokenizer import (
    build_tokenizer,
    CharLevelTokenizer,
)


class IAMWordsDataset(Dataset):
    def __init__(
        self,
        config: DictConfig,
        data: dict[str, list[str]],
        tokenizer: CharLevelTokenizer,
        is_train: bool = True,
    ):
        super().__init__()

        self.config = config
        self.is_train = is_train

        self.tokenizer = tokenizer
        self.data = data

        self.preprocess = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.uint8, scale=True),
            ]
        )
        self.normalize = v2.Compose(
            [
                v2.ToDtype(dtype=torch.float32, scale=True),
                v2.Normalize(
                    mean=self.config.preprocessor.normalize.mean,
                    std=self.config.preprocessor.normalize.std,
                ),
            ]
        )
        self.postprocess = v2.Compose(
            [
                v2.Lambda(
                    lambda image: image
                    * torch.tensor(self.config.preprocessor.normalize.std)
                    + torch.tensor(self.config.preprocessor.normalize.mean)
                ),
                v2.ToPILImage(),
            ]
        )

    def __len__(self):
        return len(self.data["filename"])

    def __getitem__(self, idx: int) -> tuple[torch.LongTensor, torch.FloatTensor]:
        image = Image.open(self.data["filename"][idx])
        image = self.preprocess(image)
        tokens_ids = torch.LongTensor(self.data["tokenized_text"][idx])

        return self.normalize(image), tokens_ids, len(tokens_ids)


def collate_fn(
    batch: list[tuple[torch.FloatTensor, torch.LongTensor, int]],
    pad_idx: int,
    max_seq_len: int = None,
) -> tuple[torch.LongTensor, torch.FloatTensor]:
    images_batch = [
        None,
    ] * len(batch)
    tokens_ids_batch = [
        None,
    ] * len(batch)
    tokens_ids_len_batch = [
        0,
    ] * len(batch)

    for i, (image, tokens_ids_seq, tokens_ids_len) in enumerate(batch):
        images_batch[i] = image
        tokens_ids_batch[i] = tokens_ids_seq
        tokens_ids_len_batch[i] = tokens_ids_len

    max_len = (
        max_seq_len if max_seq_len else max([len(seq) for seq in tokens_ids_batch])
    )
    tokens_ids_matrix = torch.zeros(len(batch), max_len, dtype=torch.long) + pad_idx
    for i, tokens_ids_seq in enumerate(tokens_ids_batch):
        tokens_ids_matrix[i, : len(tokens_ids_seq)] = tokens_ids_seq

    return (
        torch.stack(images_batch),
        tokens_ids_matrix,
        torch.LongTensor(tokens_ids_len_batch),
    )


def create_annotations(
    annotations_path: Path, images_dir: Path
) -> dict[str, list[str]]:
    """
    This function parses annotation file
    and creates dict with filename and text convenient use.
    """
    annotations = {"filename": [], "text": []}
    with open(annotations_path) as f:
        for line in f.readlines():
            image_filename, text = line.split(" ")
            image_filename = images_dir / (image_filename.split(",")[1] + ".png")
            try:
                Image.open(image_filename)
            except:
                continue
            annotations["filename"].append(image_filename)
            annotations["text"].append(text.strip())

    return annotations


def build_dataset_from_config(config: DictConfig, is_train: bool, tokenizer: CharLevelTokenizer = None) -> Dataset:
    data = {"filename": [], "text": []}
    for dataset_part in config.dataset_parts:
        annotations_part = create_annotations(
            Path(dataset_part.annotations_path), Path(dataset_part.images_dir)
        )
        data["filename"] += annotations_part["filename"]
        data["text"] += annotations_part["text"]

    if tokenizer is None:
      tokenizer = build_tokenizer(data["text"])
    data["tokenized_text"] = tokenizer.encode(data["text"])

    return IAMWordsDataset(
        data=data,
        config=config,
        tokenizer=tokenizer,
        is_train=is_train,
    )


def build_dataloader_from_config(
    config: DictConfig, is_train: bool = True, max_seq_len: int = 0, tokenizer: CharLevelTokenizer = None,
) -> DataLoader:
    dataset = build_dataset_from_config(config=config, is_train=is_train, tokenizer=tokenizer)

    max_seq_len = max_seq_len if max_seq_len > 0 else None
    dataset_max_seq_len = max([len(s) for s in dataset.data["tokenized_text"]])

    if max_seq_len is not None and max_seq_len < dataset_max_seq_len:
        raise ValueError(
            f"Max sequence length in dataset is greater than in config. Received from config: {max_seq_len}, in dataset: {dataset_max_seq_len}"
        )

    return DataLoader(
        dataset,
        collate_fn=lambda batch: collate_fn(
            batch=batch,
            pad_idx=dataset.tokenizer.pad_idx,
            max_seq_len=max_seq_len,
        ),
        shuffle=is_train,
        drop_last=is_train,
        **config.dataloader,
    )
