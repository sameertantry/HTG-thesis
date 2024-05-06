import numpy as np


def crop_text(image: np.ndarray) -> np.ndarray:
    height, width, num_channels = image.shape
    tmp = image.sum(axis=(0, 2))
    image = image[:, (tmp < height * 245 * num_channels), :]

    return image


def pad_right(image: np.ndarray, desired_width: int = 256) -> np.ndarray:
    height, width, num_channels = image.shape
    if width == desired_width:
        return image
    elif width > desired_width:
        raise ValueError(
            f"desired_width must be greater than current width. Received desired_width: {desired_width} and width: {width}"
        )
    image = np.pad(
        image,
        ((0, 0), (0, desired_width - width), (0, 0)),
        mode="constant",
        constant_values=255,
    )

    return image
