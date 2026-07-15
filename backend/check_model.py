import torch
from model import LipNet

# Load model
model = LipNet()

model.load_state_dict(
    torch.load(
        "best_lipnet.pth",
        map_location="cpu"
    )
)

model.eval()

print("Model Loaded Successfully\n")

# Print average weight values
for name, param in model.named_parameters():
    print(f"{name}: {param.abs().mean().item()}")