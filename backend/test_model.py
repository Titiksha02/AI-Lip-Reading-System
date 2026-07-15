import torch
from model import LipNet

model = LipNet()

model.load_state_dict(
    torch.load(
        "best_lipnet.pth",
        map_location="cpu"
    )
)

print("Model Loaded Successfully")