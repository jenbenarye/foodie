# run_train_segmented.py
import torch
from data_preprocess_segmented import get_dataloaders
from model_init import initialize_model
from model_train import train_model, make_optimizer, get_loss

model_name = "resnet50"
num_classes = 101
batch_size = 64
epochs = 15
lr = 3e-4
use_pretrained = True
resume_from = None
save_dir = "./trained_models/segmented_resnet50"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataloaders = get_dataloaders(batch_size=batch_size, use_augmentation=True)

model, _ = initialize_model(model_name, num_classes, resume_from, use_pretrained)
model = model.to(device)
print("Model running on:", next(model.parameters()).device)

criterion = get_loss()
optimizer = make_optimizer(model, lr)

_, val_hist, train_hist = train_model(
    model, dataloaders, criterion, optimizer,
    device, save_dir, num_epochs=epochs, save_all_epochs=False
)
