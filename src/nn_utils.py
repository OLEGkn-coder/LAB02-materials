import copy
import time
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from data import CLASSES, SEED, load_image

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TrashDataset(Dataset):
    def __init__(self, paths, labels, load_size, transform):
        self.images = [load_image(p, load_size) for p in paths]
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        return self.transform(self.images[i]), int(self.labels[i])
    
def make_test_transform(input_size):
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    
def make_loaders(paths, labels, train, val, test, load_size, input_size, batch_size):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(input_size, scale=(0.65, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15, fill=200),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    test_transform = make_test_transform(input_size)

    def subset(idx, transform):
        return TrashDataset([paths[i] for i in idx], labels[idx], load_size, transform)

    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(subset(train, train_transform), batch_size, shuffle=True, drop_last=True, generator=generator)
    val_loader = DataLoader(subset(val, test_transform), 64)
    test_loader = DataLoader(subset(test, test_transform), 64)
    return train_loader, val_loader, test_loader

def class_weights(train_labels):
    counts = np.bincount(train_labels, minlength=len(CLASSES))
    weights = counts.sum() / (len(CLASSES) * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)

def predict(model, loader):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, targets in loader:
            outputs = model(images.to(device))
            y_pred += outputs.argmax(1).cpu().tolist()
            y_true += targets.tolist()
    return np.array(y_true), np.array(y_pred)

def train_epochs(model, train_loader, val_loader, loss_fn, optimizer, scheduler, epochs, history, best):
    for epoch in range(epochs):
        start = time.time()
        model.train()
        losses = []
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(images), targets)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            losses.append(loss.item())

        y_true, y_pred = predict(model, val_loader)
        val_acc = accuracy_score(y_true, y_pred)
        val_f1 = f1_score(y_true, y_pred, average="macro")
        history.append({"loss": float(np.mean(losses)), "val_accuracy": val_acc, "val_macro_f1": val_f1})
        print(f"epoch {len(history)}: loss {np.mean(losses):.3f}, val acc {val_acc:.3f}, "
              f"val F1 {val_f1:.3f} ({time.time() - start:.0f} s)")

        if val_f1 > best["val_macro_f1"]:
            best["val_macro_f1"] = val_f1
            best["val_accuracy"] = val_acc
            best["epoch"] = len(history)
            best["weights"] = copy.deepcopy(model.state_dict())
            
def test_model(model, test_loader):
    start = time.time()
    y_true, y_pred = predict(model, test_loader)
    predict_ms = (time.time() - start) / len(y_true) * 1000
    return y_true, y_pred, predict_ms
