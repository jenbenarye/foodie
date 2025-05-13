# eval.py
import torch
from tqdm import tqdm
import numpy as np
import cv2

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_top1 = 0
    correct_top5 = 0

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Top-1
            _, preds_top1 = torch.topk(outputs, k=1, dim=1)
            correct_top1 += (preds_top1[:, 0] == labels).sum()

            # Top-5
            _, preds_top5 = torch.topk(outputs, k=5, dim=1)
            correct_top5 += (preds_top5 == labels.unsqueeze(1)).sum()

            running_loss += loss.item() * inputs.size(0)

    n_samples = len(dataloader.dataset)
    loss = running_loss / n_samples
    acc_top1 = correct_top1.double() / n_samples
    acc_top5 = correct_top5.double() / n_samples

    return loss, acc_top1, acc_top5



def evaluate_blur(model, dataloader, criterion, device, blur_sigma):
    model.eval()
    running_loss = 0.0
    correct_top1 = 0
    correct_top5 = 0

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, leave=False):
            inputs, labels = inputs.to(device), labels.to(device)

            if blur_sigma > 0:
                imgs = inputs.cpu().numpy().transpose(0, 2, 3, 1)
                imgs = (imgs * 255).astype(np.uint8)
                imgs = [cv2.GaussianBlur(img, (0, 0), blur_sigma, blur_sigma) for img in imgs]
                imgs = np.stack(imgs).astype(np.float32) / 255.
                inputs = torch.tensor(imgs).permute(0, 3, 1, 2).to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, pred1 = outputs.topk(1, dim=1)
            _, pred5 = outputs.topk(5, dim=1)

            correct_top1 += (pred1[:, 0] == labels).sum()
            correct_top5 += (pred5 == labels.unsqueeze(1)).sum()
            running_loss += loss.item() * inputs.size(0)

    n = len(dataloader.dataset)
    return running_loss / n, correct_top1.double() / n, correct_top5.double() / n
