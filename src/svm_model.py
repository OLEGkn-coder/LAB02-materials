import argparse
import os
import pickle
import time
import numpy as np
from skimage.color import rgb2gray, rgb2hsv
from skimage.feature import hog, local_binary_pattern
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from data import load_image, prepare, save_results

def extract_features(path):
    img = np.asarray(load_image(path, 128))

    hsv = rgb2hsv(img).reshape(-1, 3)
    color, _ = np.histogramdd(hsv, bins=(12, 6, 6), range=((0, 1), (0, 1), (0, 1)))
    color = np.sqrt(color.ravel() / color.sum())

    gray = rgb2gray(img)
    shape = hog(gray, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2))

    lbp = local_binary_pattern((gray * 255).astype(np.uint8), P=8, R=1, method="uniform")
    texture, _ = np.histogram(lbp, bins=10, range=(0, 10), density=True)

    return np.concatenate([color, shape, texture])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/dataset-resized")
    parser.add_argument("--split", default="random", choices=["random", "block"])
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    out_dir = os.path.join(args.out, args.split)
    paths, labels, train, val, test = prepare(args.data, args.split, out_dir)

    start = time.time()
    X = np.array([extract_features(p) for p in paths])
    print(f"Features: {X.shape[1]} per image")

    best_f1 = -1
    search = []
    for C in [1, 3, 10, 30, 100]:
        for gamma in [0.0002, 0.0005, 0.001, 0.002, 0.005]:
            model = make_pipeline(
                StandardScaler(),
                PCA(n_components=150, random_state=42),
                SVC(C=C, gamma=gamma, class_weight="balanced"),
            )
            model.fit(X[train], labels[train])
            f1 = f1_score(labels[val], model.predict(X[val]), average="macro")
            print(f"C={C}, gamma={gamma}: val macro F1 = {f1:.4f}")
            search.append({"C": C, "gamma": gamma, "val_macro_f1": f1})
            if f1 > best_f1:
                best_f1, best_model, best_params = f1, model, {"C": C, "gamma": gamma}
    train_time = time.time() - start

    print(f"Best: {best_params}")
    val_acc = accuracy_score(labels[val], best_model.predict(X[val]))

    start = time.time()
    pred = best_model.predict(X[test])
    predict_ms = (time.time() - start) / len(test) * 1000

    with open(os.path.join(out_dir, "svm.pkl"), "wb") as f:
        pickle.dump(best_model, f)

    support_vectors = best_model[-1].support_vectors_
    save_results(out_dir, "svm", labels[test], pred, {
        "best_params": best_params,
        "search": search,
        "n_features": X.shape[1],
        "n_support_vectors": support_vectors.shape[0],
        "params": support_vectors.size,
        "val_accuracy": val_acc,
        "val_macro_f1": best_f1,
        "train_minutes": train_time / 60,
        "predict_ms": predict_ms,
    })
    
if __name__ == "__main__":
    main()
