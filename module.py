import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np



class YOLO_V1(nn.Module):
    #训练网络
    def __init__(self):
        super(YOLO_V1, self).__init__()
        resnet = tvmodel

