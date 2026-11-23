import argparse
import os
import time
import torch
from efficientnet_pytorch import EfficientNet
from torch import nn
from data import CLASSES, SEED, prepare, save_results
from nn_utils import class_weights, device, make_loaders, test_model, train_epochs

def load_pretrained(weights_file):
    if weights_file:
        model = EfficientNet.from_name("efficientnet-b0")
        model.load_state_dict(torch.load(weights_file, map_location="cpu", weights_only=True))
    else:
        model = EfficientNet.from_pretrained("efficientnet-b0")
    model._fc = nn.Linear(model._fc.in_features, len(CLASSES))
    return model.to(device) 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/dataset-resized")
    parser.add_argument("--split", default="random", choices=["random", "block"])
    parser.add_argument("--out", default="results")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--head-epochs", type=int, default=3)
    parser.add_argument("--finetune-epochs", type=int, default=12)
    args = parser.parse_args()

    torch.manual_seed(SEED)
    out_dir = os.path.join(args.out, args.split)
    paths, labels, train, val, test = prepare(args.data, args.split, out_dir)
    train_loader, val_loader, test_loader = make_loaders(paths, labels, train, val, test,
                                                         load_size=256, input_size=224, batch_size=24)
    print(f"Device: {device}")

    model = load_pretrained(args.weights)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(labels[train]), label_smoothing=0.1)
    history = []
    best = {"val_macro_f1": -1}
    start = time.time()

    print("Stage 1: training only the new last layer")
    for p in model.parameters():
        p.requires_grad = False
    for p in model._fc.parameters():
        p.requires_grad = True
    optimizer = torch.optim.AdamW(model._fc.parameters(), lr=2e-3, weight_decay=1e-4)
    train_epochs(model, train_loader, val_loader, loss_fn, optimizer, None, args.head_epochs, history, best)

    print("Stage 2: fine-tuning the whole network")
    for p in model.parameters():
        p.requires_grad = True
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.finetune_epochs * len(train_loader))
    train_epochs(model, train_loader, val_loader, loss_fn, optimizer, scheduler, args.finetune_epochs, history, best)
    train_time = time.time() - start

    model.load_state_dict(best["weights"])
    torch.save(model.state_dict(), os.path.join(out_dir, "effnet.pt"))
    y_true, y_pred, predict_ms = test_model(model, test_loader)

    save_results(out_dir, "effnet", y_true, y_pred, {
        "params": sum(p.numel() for p in model.parameters()),
        "best_epoch": best["epoch"],
        "val_accuracy": best["val_accuracy"],
        "val_macro_f1": best["val_macro_f1"],
        "train_minutes": train_time / 60,
        "predict_ms": predict_ms,
        "history": history,
    })
    
if __name__ == "__main__":
    main()
