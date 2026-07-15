import torch
import torch.nn as nn

class LipNet(nn.Module):

    def __init__(self, vocab_size=28):

        super().__init__()

        self.conv3d = nn.Sequential(

            nn.Conv3d(
                in_channels=3,
                out_channels=32,
                kernel_size=(3,3,3),
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool3d(
                kernel_size=(1,2,2)
            ),

            nn.Conv3d(
                32,
                64,
                kernel_size=(3,3,3),
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool3d(
                kernel_size=(1,2,2)
            )
        )

        self.gru = nn.GRU(
            input_size=64*28*28,
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )

        self.fc = nn.Linear(
            512,
            vocab_size
        )

    def forward(self, x):

        # (B,T,H,W,C) → (B,C,T,H,W)
        x = x.permute(0,4,1,2,3)

        x = self.conv3d(x)

        B,C,T,H,W = x.shape

        x = x.permute(0,2,1,3,4)

        x = x.reshape(
            B,
            T,
            C*H*W
        )

        x,_ = self.gru(x)

        x = self.fc(x)

        return x