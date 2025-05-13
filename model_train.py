# model_train.py
import os, time, copy, torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

__all__ = ["train_model", "make_optimizer", "get_loss"]

def get_loss():
    return nn.CrossEntropyLoss()

def make_optimizer(model, lr):
    return optim.Adam(model.parameters(), lr=lr)

def train_model(
    model,
    dataloaders,
    criterion,
    optimizer,
    device,
    save_dir,
    num_epochs=25,
    save_all_epochs=False,
):
    os.makedirs(save_dir, exist_ok=True)
    since = time.time()

    best_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    val_acc_history, train_acc_history = [], []

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}\n" + "-"*10)
        for phase in ["train", "val"]:
            model.train() if phase == "train" else model.eval()
            running_loss, running_corrects = 0.0, 0

            for inputs, labels in tqdm(dataloaders[phase], leave=False):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += (preds == labels).sum()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            print(f"{phase} Loss {epoch_loss:.4f} Acc {epoch_acc:.4f}")

            if phase == "val":
                val_acc_history.append(epoch_acc)
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_wts = copy.deepcopy(model.state_dict())
            else:
                train_acc_history.append(epoch_acc)

            if save_all_epochs:
                torch.save(model.state_dict(), os.path.join(save_dir, f"weights_e{epoch+1}.pt"))

    elapsed = time.time() - since
    print(f"\nTraining finished in {elapsed//60:.0f}m {elapsed%60:.0f}s")
    print(f"Best val acc {best_acc:.4f}")

    torch.save(best_wts, os.path.join(save_dir, "weights_best.pt"))
    torch.save(model.state_dict(), os.path.join(save_dir, "weights_last.pt"))
    model.load_state_dict(best_wts)
    return model, val_acc_history, train_acc_history
