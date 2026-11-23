import glob
import hashlib
import json
import os
import re
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
SEED = 42
BLOCK_SIZE = 20

def file_number(path):
    return int(re.findall(r"(\d+)\.jpg$", path)[0])

def load_dataset(data_dir):
    paths = []
    labels = []
    for label, name in enumerate(CLASSES):
        files = glob.glob(os.path.join(data_dir, name, "*.jpg"))
        files.sort(key=file_number)
        paths += files
        labels += [label] * len(files)

    by_hash = {}
    for i, path in enumerate(paths):
        with open(path, "rb") as f:
            h = hashlib.md5(f.read()).hexdigest()
        by_hash.setdefault(h, []).append(i)

    bad = set()
    for group in by_hash.values():
        if len({labels[i] for i in group}) > 1:
            bad.update(group)

    keep = [i for i in range(len(paths)) if i not in bad]
    print(f"Images: {len(paths)}, removed duplicates with different labels: {len(bad)}, left: {len(keep)}")
    return [paths[i] for i in keep], np.array([labels[i] for i in keep])

def split_data(labels, mode):
    idx = np.arange(len(labels))
    if mode == "random":
        train, rest = train_test_split(idx, test_size=0.3, stratify=labels, random_state=SEED)
        val, test = train_test_split(rest, test_size=0.5, stratify=labels[rest], random_state=SEED)
        return train, val, test

    rng = np.random.RandomState(SEED)
    train, val, test = [], [], []
    for c in range(len(CLASSES)):
        class_idx = idx[labels == c]
        blocks = [class_idx[i:i + BLOCK_SIZE] for i in range(0, len(class_idx), BLOCK_SIZE)]
        order = rng.permutation(len(blocks))
        n_test = round(0.15 * len(blocks))
        n_val = round(0.15 * len(blocks))
        for k, b in enumerate(order):
            if k < n_test:
                test += list(blocks[b])
            elif k < n_test + n_val:
                val += list(blocks[b])
            else:
                train += list(blocks[b])
    return np.array(train), np.array(val), np.array(test)

def prepare(data_dir, split, out_dir):
    paths, labels = load_dataset(data_dir)
    train, val, test = split_data(labels, split)
    print(f"Split '{split}': train {len(train)}, val {len(val)}, test {len(test)}")

    os.makedirs(out_dir, exist_ok=True)
    names = {
        "train": [os.path.basename(paths[i]) for i in train],
        "val": [os.path.basename(paths[i]) for i in val],
        "test": [os.path.basename(paths[i]) for i in test],
    }
    with open(os.path.join(out_dir, "split.json"), "w") as f:
        json.dump(names, f)
    return paths, labels, train, val, test

def load_image(path, size):
    return Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)

def save_results(out_dir, name, y_true, y_pred, info):
    acc = accuracy_score(y_true, y_pred)
    results = {
        "model": name,
        "accuracy": acc,
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "f1_per_class": dict(zip(CLASSES, f1_score(y_true, y_pred, average=None).tolist())),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "y_true": [int(v) for v in y_true],
        "y_pred": [int(v) for v in y_pred],
    }
    results.update(info)
    with open(os.path.join(out_dir, f"{name}.json"), "w") as f:
        json.dump(results, f, indent=1)
    print(f"[{name}] test accuracy = {acc:.4f}, macro F1 = {results['macro_f1']:.4f}")
