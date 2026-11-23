import argparse
import json
import os
import matplotlib
import numpy as np
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from data import CLASSES, load_image

MODELS = {"svm": "SVM", "cnn": "Own CNN", "effnet": "EfficientNet-B0"}
SPLITS = ["random", "block"]

def load_results(results_dir):
    results = {}
    for split in SPLITS:
        for model in MODELS:
            path = os.path.join(results_dir, split, f"{model}.json")
            if os.path.exists(path):
                with open(path) as f:
                    results[split, model] = json.load(f)
    return results

def main_table(results):
    lines = [
        "| Split | Model | Test accuracy | Macro F1 | Val accuracy | Parameters | Training, min |",
        "|---|---|---|---|---|---|---|",
    ]
    for split in SPLITS:
        for model, name in MODELS.items():
            if (split, model) not in results:
                continue
            r = results[split, model]
            params = r["params"]
            lines.append(
                f"| {split} | {name} | {r['accuracy'] * 100:.1f}% | {r['macro_f1']:.3f} | "
                f"{r['val_accuracy'] * 100:.1f}% | {params:,} | {r['train_minutes']:.1f} |"
            )
    return lines

def class_table(results):
    lines = ["| Split | Model | " + " | ".join(CLASSES) + " |", "|---|---|" + "---|" * len(CLASSES)]
    for split in SPLITS:
        for model, name in MODELS.items():
            if (split, model) in results:
                f1 = results[split, model]["f1_per_class"]
                lines.append(f"| {split} | {name} | " + " | ".join(f"{f1[c]:.3f}" for c in CLASSES) + " |")
    return lines

def plot_accuracy(results, out_dir):
    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.35
    x = np.arange(len(MODELS))
    for i, split in enumerate(SPLITS):
        acc = [results[split, m]["accuracy"] * 100 if (split, m) in results else 0 for m in MODELS]
        bars = ax.bar(x + (i - 0.5) * width, acc, width, label=split)
        ax.bar_label(bars, fmt="%.1f")
    ax.set_xticks(x, MODELS.values())
    ax.set_ylabel("Test accuracy, %")
    ax.set_ylim(0, 100)
    ax.legend(title="Split")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "accuracy.png"), dpi=150)
    plt.close(fig)
    
def plot_confusion(results, out_dir):
    for split in SPLITS:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        for ax, (model, name) in zip(axes, MODELS.items()):
            if (split, model) not in results:
                ax.axis("off")
                continue
            cm = np.array(results[split, model]["confusion_matrix"])
            ax.imshow(cm, cmap="Blues")
            for i in range(len(CLASSES)):
                for j in range(len(CLASSES)):
                    color = "white" if cm[i, j] > cm.max() / 2 else "black"
                    ax.text(j, i, cm[i, j], ha="center", va="center", color=color)
            ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=45)
            ax.set_yticks(range(len(CLASSES)), CLASSES)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_title(f"{name}, accuracy {results[split, model]['accuracy'] * 100:.1f}%")
        fig.suptitle(f"Confusion matrices, split '{split}'")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"confusion_{split}.png"), dpi=150)
        plt.close(fig)
        
def plot_history(results, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for model in ["cnn", "effnet"]:
        for split in SPLITS:
            if (split, model) not in results:
                continue
            history = results[split, model]["history"]
            epochs = range(1, len(history) + 1)
            label = f"{MODELS[model]}, {split}"
            axes[0].plot(epochs, [h["loss"] for h in history], label=label)
            axes[1].plot(epochs, [h["val_accuracy"] * 100 for h in history], label=label)
    axes[0].set_title("Training loss")
    axes[1].set_title("Validation accuracy, %")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "training.png"), dpi=150)
    plt.close(fig)
    
def correct_patterns(results, split):
    y = np.array(results[split, "svm"]["y_true"])
    ok = {m: np.array(results[split, m]["y_pred"]) == y for m in MODELS}
    patterns = {}
    for s in [True, False]:
        for c in [True, False]:
            for e in [True, False]:
                names = [MODELS[m] for m, v in zip(MODELS, [s, c, e]) if v]
                label = " + ".join(names) if names else "nobody"
                patterns[label] = int(np.sum((ok["svm"] == s) & (ok["cnn"] == c) & (ok["effnet"] == e)))
    return patterns

def agreement_table(results):
    lines = ["| Who is correct | random | block |", "|---|---|---|"]
    random_p = correct_patterns(results, "random")
    block_p = correct_patterns(results, "block")
    for label in random_p:
        lines.append(f"| {label} | {random_p[label]} | {block_p[label]} |")
    return lines

def benchmark_table(results, bench):
    lines = ["| Model | Accuracy (random) | Training, min | ms per image | Images per second | Parameters | File, MB |",
             "|---|---|---|---|---|---|---|"]
    for model, name in MODELS.items():
        r = results["random", model]
        b = bench[model]
        lines.append(f"| {name} | {r['accuracy'] * 100:.1f}% | {r['train_minutes']:.1f} | {b['ms_per_image']:.1f} | "
                     f"{b['images_per_second']:.0f} | {r['params']:,} | {b['file_mb']:.1f} |")
    return lines

def plot_speed_accuracy(results, bench, out_dir):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for model, name in MODELS.items():
        ms = bench[model]["ms_per_image"]
        acc = [results[s, model]["accuracy"] * 100 for s in SPLITS]
        size = results["random", model]["params"] / 20000
        ax.scatter([ms, ms], acc, s=size, alpha=0.6)
        ax.plot([ms, ms], acc, color="gray", linewidth=1)
        ax.annotate(name, (ms, max(acc)), textcoords="offset points", xytext=(10, 5))
    ax.set_xscale("log")
    ax.set_xlim(4, 40)
    ax.set_xlabel("Time to classify one photo, ms (log scale)")
    ax.set_ylabel("Test accuracy, % (both splits)")
    ax.set_title("Speed vs accuracy (circle size = number of parameters)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "speed_vs_accuracy.png"), dpi=150)
    plt.close(fig)
    
def plot_cost(results, bench, out_dir):
    names = list(MODELS.values())
    values = [
        ("Training time, min", [results["random", m]["train_minutes"] for m in MODELS]),
        ("Time per photo, ms", [bench[m]["ms_per_image"] for m in MODELS]),
        ("Parameters, millions", [results["random", m]["params"] / 1e6 for m in MODELS]),
        ("Model file, MB", [bench[m]["file_mb"] for m in MODELS]),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, (title, vals) in zip(axes, values):
        bars = ax.bar(names, vals, color=["#8c8c8c", "#5b8fd6", "#1f4e9c"])
        ax.bar_label(bars, fmt="%.1f")
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "cost.png"), dpi=150)
    plt.close(fig)
    
def plot_f1_heatmap(results, out_dir):
    rows = [(s, m) for s in SPLITS for m in MODELS]
    data = np.array([[results[r]["f1_per_class"][c] for c in CLASSES] for r in rows])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.imshow(data, cmap="RdYlGn", vmin=0.4, vmax=1)
    for i in range(len(rows)):
        for j in range(len(CLASSES)):
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center")
    ax.set_xticks(range(len(CLASSES)), CLASSES)
    ax.set_yticks(range(len(rows)), [f"{MODELS[m]}, {s}" for s, m in rows])
    ax.set_title("F1 by class")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "f1_by_class.png"), dpi=150)
    plt.close(fig)
    
def plot_agreement(results, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), sharex=True)
    for ax, split in zip(axes, SPLITS):
        patterns = correct_patterns(results, split)
        labels = list(patterns)[::-1]
        bars = ax.barh(labels, [patterns[k] for k in labels])
        ax.bar_label(bars)
        ax.set_title(f"Which models are correct on each test photo, split '{split}'")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "agreement.png"), dpi=150)
    plt.close(fig)
    
def plot_svm_grid(results, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, split in zip(axes, SPLITS):
        search = results[split, "svm"]["search"]
        cs = sorted({s["C"] for s in search})
        gammas = sorted({s["gamma"] for s in search})
        grid = np.zeros((len(cs), len(gammas)))
        for s in search:
            grid[cs.index(s["C"]), gammas.index(s["gamma"])] = s["val_macro_f1"]
        ax.imshow(grid, cmap="viridis")
        for i in range(len(cs)):
            for j in range(len(gammas)):
                ax.text(j, i, f"{grid[i, j]:.3f}", ha="center", va="center", color="white")
        ax.set_xticks(range(len(gammas)), gammas)
        ax.set_yticks(range(len(cs)), cs)
        ax.set_xlabel("gamma")
        ax.set_ylabel("C")
        ax.set_title(f"SVM: validation macro F1, split '{split}'")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "svm_grid.png"), dpi=150)
    plt.close(fig)
    
def plot_errors(results, results_dir, data_dir, out_dir):
    with open(os.path.join(results_dir, "random", "split.json")) as f:
        test_names = json.load(f)["test"]
    r = results["random", "effnet"]
    wrong = [i for i, (t, p) in enumerate(zip(r["y_true"], r["y_pred"])) if t != p]
    cols = 7
    rows = (len(wrong) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, i in zip(axes.ravel(), wrong):
        name = test_names[i]
        folder = CLASSES[r["y_true"][i]]
        ax.imshow(load_image(os.path.join(data_dir, folder, name), 160))
        ax.set_title(f"{CLASSES[r['y_true'][i]]} -> {CLASSES[r['y_pred'][i]]}", fontsize=9)
    fig.suptitle("EfficientNet-B0 mistakes, split 'random' (true -> predicted)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "effnet_errors.png"), dpi=120)
    plt.close(fig)
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results")
    parser.add_argument("--data", default="data/dataset-resized")
    args = parser.parse_args()

    results = load_results(args.results)
    plots_dir = os.path.join(args.results, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    bench_file = os.path.join(args.results, "benchmark.json")
    bench = None
    if os.path.exists(bench_file):
        with open(bench_file) as f:
            bench = json.load(f)

    text = ["# Results", "", "## Main metrics", ""] + main_table(results)
    if bench:
        text += ["", "## Speed and size", ""] + benchmark_table(results, bench)
    text += ["", "## F1 by class", ""] + class_table(results)
    text += ["", "## Which models are correct", ""] + agreement_table(results)
    text += ["", "## Plots", ""]
    plots = ["accuracy", "speed_vs_accuracy", "cost", "f1_by_class", "confusion_random", "confusion_block",
             "agreement", "training", "svm_grid", "effnet_errors"]
    for p in plots:
        text += [f"![{p}](plots/{p}.png)", ""]
    with open(os.path.join(args.results, "summary.md"), "w") as f:
        f.write("\n".join(text))

    plot_accuracy(results, plots_dir)
    plot_confusion(results, plots_dir)
    plot_history(results, plots_dir)
    plot_f1_heatmap(results, plots_dir)
    plot_agreement(results, plots_dir)
    plot_svm_grid(results, plots_dir)
    if bench:
        plot_speed_accuracy(results, bench, plots_dir)
        plot_cost(results, bench, plots_dir)
    if os.path.isdir(args.data):
        plot_errors(results, args.results, args.data, plots_dir)

    print("\n".join(main_table(results)))
    print(f"\nSaved {os.path.join(args.results, 'summary.md')} and plots in {plots_dir}")
    
if __name__ == "__main__":
    main()
