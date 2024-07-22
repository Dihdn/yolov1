import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np


class Yolo_V1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=192, stride=2, kernel_size=7, padding=1),
                                    nn.MaxPool2d(kernel_size=2, stride=2, padding=1))
        self.conv_2 = nn.Sequential(nn.Conv2d(in_channels=192, out_channels=256, stride=1, kernel_size=3, padding=1),
                                    nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv_3 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=128, stride=1, kernel_size=1),
                                    nn.Conv2d(in_channels=128, out_channels=256, stride=1, kernel_size=3, padding=1),
                                    nn.Conv2d(in_channels=256, out_channels=256, stride=1, kernel_size=1),
                                    nn.Conv2d(in_channels=256, out_channels=512, stride=1, kernel_size=3, padding=1),
                                    nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv_4_min = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=256, stride=1, kernel_size=1),
                                        nn.Conv2d(in_channels=256, out_channels=512, stride=1, kernel_size=3, padding=1))
        self.conv_4 = nn.Sequential(self.conv_4_min,
                                   self.conv_4_min,
                                   self.conv_4_min,
                                   self.conv_4_min,
                                   nn.Conv2d(in_channels=512, out_channels=512, stride=1, kernel_size=1),
                                   nn.Conv2d(in_channels=512, out_channels=1024, stride=1, kernel_size=3, padding=1),
                                   nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv_5_min = nn.Sequential(nn.Conv2d(in_channels=1024, out_channels=512, stride=1, kernel_size=1),
                                       nn.Conv2d(in_channels=512, out_channels=1024, stride=1, kernel_size=3, padding=1))
        self.conv_5 = nn.Sequential(self.conv_5_min,
                                   self.conv_5_min,
                                   nn.Conv2d(in_channels=1024, out_channels=1024, stride=1, kernel_size=3, padding=1),
                                   nn.Conv2d(in_channels=1024, out_channels=1024, stride=2, kernel_size=3, padding=1))
        self.conv_6 = nn.Sequential(nn.Conv2d(in_channels=1024, out_channels=1024, stride=1, kernel_size=3, padding=1),
                                   nn.Conv2d(in_channels=1024, out_channels=1024, stride=1, kernel_size=3, padding=1))
        self.conn = nn.Sequential(nn.Flatten(),
                                  nn.Linear(in_features=7*7*1024, out_features=4096),
                                  nn.Linear(in_features=4096, out_features=7*7*8))

    def forward(self, x):
        x = F.leaky_relu(self.conv_1(x))
        x = F.leaky_relu(self.conv_2(x))
        x = F.leaky_relu(self.conv_3(x))
        x = F.leaky_relu(self.conv_4(x))
        x = F.leaky_relu(self.conv_5(x))
        x = F.leaky_relu(self.conv_6(x))
        x = self.conn(x)
        return x

if __name__ == "__main__":
    net = Yolo_V1()
    x = torch.ones(size=(1, 3, 448, 448))
    y = net(x)




