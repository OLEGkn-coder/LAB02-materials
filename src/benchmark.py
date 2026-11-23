import argparse
import json
import os
import pickle
import re
import time
import numpy as np
import torch
from efficientnet_pytorch import EfficientNet
from torch import nn
from cnn_model import SmallCNN
from data import CLASSES, load_image
from nn_utils import make_test_transform
from svm_model import extract_features

def measure(predict_one, paths):
    predict_one(paths[0])
    times = []
    for path in paths:
        start = time.perf_counter()
        predict_one(path)
        times.append((time.perf_counter() - start) * 1000)
    return float(np.median(times))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/dataset-resized")
    parser.add_argument("--results", default="results")
    parser.add_argument("--split", default="random")
    parser.add_argument("--n", type=int, default=100)
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    model_dir = os.path.join(args.results, args.split)
    with open(os.path.join(model_dir, "split.json")) as f:
        names = json.load(f)["test"][:args.n]
    paths = [os.path.join(args.data, re.match(r"[a-z]+", n).group(), n) for n in names]

    with open(os.path.join(model_dir, "svm.pkl"), "rb") as f:
        svm = pickle.load(f)

    cnn = SmallCNN(len(CLASSES))
    cnn.load_state_dict(torch.load(os.path.join(model_dir, "cnn.pt"), weights_only=True))
    cnn.eval()
    cnn_transform = make_test_transform(128)

    effnet = EfficientNet.from_name("efficientnet-b0")
    effnet._fc = nn.Linear(effnet._fc.in_features, len(CLASSES))
    effnet.load_state_dict(torch.load(os.path.join(model_dir, "effnet.pt"), weights_only=True))
    effnet.eval()
    effnet_transform = make_test_transform(224)

    def svm_one(path):
        return svm.predict(extract_features(path)[None])

    def cnn_one(path):
        return cnn(cnn_transform(load_image(path, 150))[None]).argmax(1)
    
    def effnet_one(path):
        return effnet(effnet_transform(load_image(path, 256))[None]).argmax(1)

    results = {}
    for name, fn, file in [("svm", svm_one, "svm.pkl"), ("cnn", cnn_one, "cnn.pt"), ("effnet", effnet_one, "effnet.pt")]:
        ms = measure(fn, paths)
        size_mb = os.path.getsize(os.path.join(model_dir, file)) / 1024 / 1024
        results[name] = {"ms_per_image": ms, "images_per_second": 1000 / ms, "file_mb": size_mb}
        print(f"{name}: {ms:.1f} ms per image ({1000 / ms:.0f} images/s), model file {size_mb:.1f} MB")

    with open(os.path.join(args.results, "benchmark.json"), "w") as f:
        json.dump(results, f, indent=1)

if __name__ == "__main__":
    main()
