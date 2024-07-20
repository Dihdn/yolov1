import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import pandas as pd
import numpy as np
import os
import utilities


data_path = "E:\\yolov1\\new_data\\train"

# 制作dataset
class new_data_dataset(Dataset):
    def __init__(self, data_path, resize=(448, 448)):
        self.data_path = data_path
        self.resize = resize
        #获取路径中所有图片名称
        self.img_path_names = os.listdir(path=data_path+"\\train_img")
        self.xml_path_names = os.listdir(path=data_path+"\\train_xml")

    def __len__(self):
        # 返回数图片数量
        return len(self.img_path_names)

    def __getitem__(self, index):
        img_path = self.data_path + "\\train_img\\" + self.img_path_names[index]
        xml_path = self.data_path + "\\train_xml\\" + self.img_path_names[index]
        # 将图片尺寸调整为448*448
        img = utilities.resize_img(img_path, self.resize)
        return img


if __name__ == "__main__":
    dataset = new_data_dataset(data_path)
    a = dataset[0]
    print("a")
