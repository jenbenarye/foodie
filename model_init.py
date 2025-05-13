# model_init.py
import torch
from torchvision import models

def initialize_model(
    model_name="resnet50",
    num_classes=101,
    resume_from=None,
    use_pretrained=True,
):
    """
    Init CNN with new classifier head.

    Args:
        model_name: Backbone. Only "resnet50" for now.
        num_classes: Output classes. 101 for Food-101.
        resume_from: Path to .pt file to load weights from.
        use_pretrained: If True, load ImageNet weights.

    Returns:
        model: The modified model.
        input_size: Expected image size (224 for ResNet-50).
    """
    if model_name != "resnet50":
        raise ValueError("Only resnet50 supported.")

    model = models.resnet50(pretrained=use_pretrained)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    input_size = 224

    if resume_from:
        model.load_state_dict(torch.load(resume_from, map_location="cpu"))

    return model, input_size
