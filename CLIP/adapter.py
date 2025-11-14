import os
import argparse
import random
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image
import torchvision.models as models

class ClipAdapter(nn.Module):
    def __init__(self, c_in, bottleneck=768):
        super(ClipAdapter, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(c_in, bottleneck, bias=False),
            nn.LeakyReLU(inplace=False)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(bottleneck, c_in, bias=False),
            nn.LeakyReLU(inplace=False)
        )

    def forward(self, x):
        x = self.fc1(x)
        y = self.fc2(x)
        return x, y

class CLIP_Inplanted(nn.Module):
    def __init__(self, c_in, device):
        super().__init__()
        self.device = device
        self.cls_token_adapter = nn.ModuleList([ClipAdapter(c_in=c_in) for _ in range(4)])
        self.prompt_adapter = nn.ModuleList([ClipAdapter(c_in=768) for _ in range(2)])
        self.patch_token_adapter = nn.ModuleList([ClipAdapter(c_in=c_in) for _ in range(4)])

    def forward(self,):
        return
