import pandas as pd

import torch
import torchvision

from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as TF2
from torch.utils.data import Dataset, DataLoader

from PIL import Image

from omegaconf import DictConfig, OmegaConf

from pathlib import Path

from handwritten_generation.datasets.tokenizer import build_tokenizer


class IAMLinesDataset(Dataset):
    def __init__(self, data: pd.DataFrame, config: DictConfig, is_train: bool = True):
        self.config = config
        self.is_train = is_train
        self.tokenizer = build_tokenizer(data["line"])
        self.data = data
        self.tokenized_data = self.tokenizer.encode(data["line"])

        self.margin = self.config.preprocessor.margin
        
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
        
        self.train_transform = v2.Compose([
            v2.RandomZoomOut(**self.config.preprocessor.random_zoom_out),
            v2.RandomAffine(**self.config.preprocessor.random_affine),
        ])
        self.postprocess = v2.Compose([
            v2.Lambda(lambda image: image * torch.tensor(self.config.preprocessor.normalize.std) + torch.tensor(self.config.preprocessor.normalize.mean)),
            #v2.ToPILImage(),
        ])

    def _rescale(self, image_tensor: torch.FloatTensor) -> torch.FloatTensor:
        #if self.is_train:
        #    image_tensor = self.train_transform(image_tensor)
            
        _, h, w = image_tensor.size()
        desired_h = self.config.preprocessor.h
        desired_w = self.config.preprocessor.w
        if w > desired_w:
            scale = w // desired_w
            scaled_h = h // scale
            scaled_h -= scaled_h % 2
            scaled_w = desired_w
        elif h > desired_h:
            scale = h // desired_h
            scaled_w = w // scale
            scaled_w -= scaled_w % 2
            scaled_h = desired_h
        else:
            scaled_w = w - w % 2
            scaled_h = h - h % 2

        image_tensor = TF2.resize(image_tensor, (scaled_h, scaled_w), antialias=True)
        image_tensor = self.normalize(image_tensor)
        image_tensor = TF2.pad(image_tensor, ((desired_w - scaled_w) // 2, (desired_h - scaled_h) // 2, (desired_w - scaled_w) // 2, (desired_h - scaled_h) // 2), fill=1.)
        
        return image_tensor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.LongTensor, torch.FloatTensor]:
        row = self.data.iloc[idx]
        image = Image.open(row["image_path"])
        
        image = self.preprocess(image)[:, int(row["t"]) - self.margin : -(int(row["b"]) - self.margin), int(row["l"]) - self.margin : -(int(row["r"]) - self.margin)]
        image = self._rescale(image)

        return torch.LongTensor(self.tokenized_data[idx]), image


class IAMWordsDataset(Dataset):
    def __init__(self, config: DictConfig, data: pd.DataFrame, is_train: bool = True):
        super().__init__()

        self.config = config
        self.is_train = is_train

        data = data.fillna("None")
        self.tokenizer = build_tokenizer(data["word"])
        self.data = data
        self.tokenized_data = self.tokenizer.encode(data["word"])

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
            #v2.ToPILImage(),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.LongTensor, torch.FloatTensor]:
        row = self.data.iloc[idx]
        image = Image.open(row["filename"])
        image = self.preprocess(image)

        return torch.LongTensor(self.tokenized_data[idx]), self.normalize(image)


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


def build_dataset_from_config(config: DictConfig, is_train: bool) -> Dataset:
    data = []
    for df_path in config.datasets:
        data.append(pd.read_csv(df_path))

    data = pd.concat(data)

    if config.name == "lines":
        return IAMLinesDataset(
            data=data,
            config=config,
            is_train=is_train,
        )
    elif config.name == "words":
        return IAMWordsDataset(
            data=data,
            config=config,
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