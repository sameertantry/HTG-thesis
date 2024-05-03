import argparse
import numpy as np

from pathlib import Path
from PIL import Image

from handwritten_generation.tools.utils import crop_text, pad_right


def parse_args():
    parser = argparse.ArgumentParser(description="Remove white borders")
    parser.add_argument("--input_dir", type=Path, help="Images directory path")
    parser.add_argument("--output_dir", type=Path, help="Directory to store processed images")

    return parser.parse_args()

def main(input_dir: Path, output_dir: Path):
    output_dir.mkdir(exist_ok=True)
    for image_path in input_dir.iterdir():
        image = Image.open(image_path)
        image = np.array(image)
        image = crop_text(image)
        image = pad_right(image)
        image = Image.fromarray(image)
        image.save(output_dir / image_path.name)
    

if __name__ == "__main__":
    args = parse_args()
    main(input_dir=args.input_dir, output_dir=args.output_dir)
