# run_eval_segmented.py
import torch
from data_preprocess_segmented import get_dataloaders
from model_init import initialize_model
from eval import evaluate, evaluate_blur
from model_train import get_loss
import pandas as pd


weights = "./trained_models/segmented_resnet50/weights_best.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataloaders = get_dataloaders(batch_size=64, use_augmentation=False)

model, _ = initialize_model("resnet50", num_classes=101, resume_from=None, use_pretrained=False)
model.load_state_dict(torch.load(weights, map_location=device))
model = model.to(device)

criterion = get_loss()
val_loss, acc1, acc5 = evaluate(model, dataloaders["val"], criterion, device)
print(f"Segmented Val Loss {val_loss:.4f} Top-1 Acc {acc1:.4f} Top-5 Acc {acc5:.4f}")


sigmas = [0, 0.1, 0.2, 0.3, 0.5]
results = []

for sigma in sigmas:
    loss, acc1, acc5 = evaluate_blur(model, dataloaders["val"], criterion, device, blur_sigma=sigma)
    print(f"sigma={sigma:<4} loss={loss:.4f} top1={acc1:.4f} top5={acc5:.4f}")
    results.append((sigma, loss, acc1.item(), acc5.item()))

df = pd.DataFrame(results, columns=["sigma", "val_loss", "top1_acc", "top5_acc"])
df.to_csv("blur_robustness_results_segmented.csv", index=False)
