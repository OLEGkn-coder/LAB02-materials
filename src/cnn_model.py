import argparse
import os
import time
import torch
from torch import nn
from data import CLASSES, SEED, prepare, save_results
from nn_utils import class_weights, device, make_loaders, test_model, train_epochs

def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )
    
class SmallCNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(3, 16),
            conv_block(16, 32),
            conv_block(32, 64),
            conv_block(64, 128),
            conv_block(128, 200),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(200, n_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.fc(self.dropout(x))
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/dataset-resized")
    parser.add_argument("--split", default="random", choices=["random", "block"])
    parser.add_argument("--out", default="results")
    parser.add_argument("--epochs", type=int, default=40)
    args = parser.parse_args()

    torch.manual_seed(SEED)
    out_dir = os.path.join(args.out, args.split)
    paths, labels, train, val, test = prepare(args.data, args.split, out_dir)
    train_loader, val_loader, test_loader = make_loaders(paths, labels, train, val, test,
                                                         load_size=150, input_size=128, batch_size=32)
    print(f"Device: {device}")

    model = SmallCNN(len(CLASSES)).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(labels[train]), label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=3e-3, epochs=args.epochs,
                                                    steps_per_epoch=len(train_loader))

    history = []
    best = {"val_macro_f1": -1}
    start = time.time()
    train_epochs(model, train_loader, val_loader, loss_fn, optimizer, scheduler, args.epochs, history, best)
    train_time = time.time() - start

    model.load_state_dict(best["weights"])
    torch.save(model.state_dict(), os.path.join(out_dir, "cnn.pt"))
    y_true, y_pred, predict_ms = test_model(model, test_loader)

    save_results(out_dir, "cnn", y_true, y_pred, {
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
