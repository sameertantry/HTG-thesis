import pandas as pd

import torch
import torchvision

from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as TF2
from torch.utils.data import Dataset, DataLoader

from PIL import Image

from omegaconf import DictConfig, OmegaConf

from pathlib import Path


class IAMWordsDataset(Dataset):
    def __init__(self, config: DictConfig, data: dict[str, list[str]], tokenizer: CharLevelTokenizer, is_train: bool = True):
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
                v2.Normalize(mean=self.config.preprocessor.normalize.mean, std=self.config.preprocessor.normalize.std),
            ]
        )
        self.postprocess = v2.Compose([
            v2.Lambda(lambda image: image * torch.tensor(self.config.preprocessor.normalize.std) + torch.tensor(self.config.preprocessor.normalize.mean)),
            v2.ToPILImage(),
        ])

    def __len__(self):
        return len(self.data["filename"])

    def __getitem__(self, idx: int) -> tuple[torch.LongTensor, torch.FloatTensor]:
        image = Image.open(self.data["filename"][idx])
        image = self.preprocess(image)

        return torch.LongTensor(self.data["tokenized_text"][idx]), self.normalize(image)


def collate_fn(batch: list[tuple[torch.LongTensor, torch.FloatTensor]], pad_idx: int) -> tuple[torch.LongTensor, torch.FloatTensor]:
    tokens_batch = [None,] * len(batch)
    images_batch = [None,] * len(batch)
    for i, (tokens_seq, image) in enumerate(batch):
        tokens_batch[i] = tokens_seq
        images_batch[i] = image

    max_len = max([len(seq) for seq in tokens_batch])
    tokens_matrix = torch.zeros(len(batch), max_len, dtype=torch.long) + pad_idx
    for i, tokens_seq in enumerate(tokens_batch):
        tokens_matrix[i, :len(tokens_seq)] = tokens_seq

    return tokens_matrix, torch.stack(images_batch)


def create_annotations(annotations_path: Path, images_dir: Path) -> dict[str, list[str]]:
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
            annotations["text"].append(text)

    return annotations


def build_dataset_from_config(config: DictConfig, is_train: bool) -> Dataset:
    data = create_annotations(Path(config.annotations_path), Path(config.images_dir))
    tokenizer = build_tokenizer(data["text"])
    data["tokenized_text"] = tokenizer.encode(data["text"])

    if config.name == "words":
        return IAMWordsDataset(
            data=data,
            config=config,
            tokenizer=tokenizer,
            is_train=is_train,
        )
    else:
        raise ValueError("Unsupported dataset")


def build_dataloader_from_config(config: DictConfig, is_train: bool = True) -> DataLoader:
    dataset = build_dataset_from_config(config=config, is_train=is_train)

    return DataLoader(
        dataset,
        collate_fn=lambda batch: collate_fn(batch=batch, pad_idx=dataset.tokenizer.pad_idx),
        shuffle=is_train,
        **config.dataloader,
    )
