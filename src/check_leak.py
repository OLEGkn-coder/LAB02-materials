import argparse
import json
import os
import re
import numpy as np
from PIL import Image

def class_name(file_name):
    return re.match(r"[a-z]+", file_name).group()

def number(file_name):
    return int(re.search(r"\d+", file_name).group())

def small_vector(data_dir, file_name):
    img = Image.open(os.path.join(data_dir, class_name(file_name), file_name)).convert("L").resize((64, 48))
    v = np.asarray(img, dtype=np.float32).ravel()
    v -= v.mean()
    return v / np.linalg.norm(v)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/dataset-resized")
    parser.add_argument("--results", default="results")
    args = parser.parse_args()

    for split in ["random", "block"]:
        with open(os.path.join(args.results, split, "split.json")) as f:
            names = json.load(f)
        train = np.stack([small_vector(args.data, n) for n in names["train"]])
        test = np.stack([small_vector(args.data, n) for n in names["test"]])
        max_similarity = (test @ train.T).max(axis=1)

        train_numbers = {(class_name(n), number(n)) for n in names["train"]}
        neighbours = [
            (class_name(n), number(n) - 1) in train_numbers or (class_name(n), number(n) + 1) in train_numbers
            for n in names["test"]
        ]

        print(f"Split '{split}':")
        print(f"  test images with a train image of the same class and neighbouring number: {np.mean(neighbours) * 100:.1f}%")
        print(f"  test images with a very similar train image (cosine > 0.9): {np.mean(max_similarity > 0.9) * 100:.1f}%")
        
if __name__ == "__main__":
    main()
