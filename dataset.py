import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import pandas as pd
import numpy as np
import PIL.Image as Image
import os
import utilities

CLASS_LIST = ["apple", "banana", "orange"]

data_path = "E:\\yolov1\\new_data\\train"

# 制作dataset
class new_data_dataset(Dataset):
    def __init__(self, data_path, resize=(448, 448), map_size=(7, 7)):
        self.data_path = data_path
        self.resize = resize
        self.map_size = map_size
        #获取路径中所有图片名称
        img_path = data_path+"\\train_img"
        xml_path = data_path+"\\train_xml"
        self.img_path_names = os.listdir(path=img_path)
        self.xml_path_names = os.listdir(path=xml_path)

    def __len__(self):
        # 返回数图片数量
        return len(self.img_path_names)

    def __getitem__(self, index):
        img_path = self.data_path + "\\train_img\\" + self.img_path_names[index]
        xml_path = self.data_path + "\\train_xml\\" + self.xml_path_names[index]
        # 将图片尺寸调整为448*448
        img = utilities.resize_img(img_path, self.resize)
        # 得到标签
        label = utilities.get_label(xml_path=xml_path, img_size=img.size, resize=self.resize, map_size=self.map_size, class_list=CLASS_LIST)
        return img, label


if __name__ == "__main__":
    dataset = new_data_dataset(data_path)
    img, label = dataset[1]
    Image._show(image=img)
    print(label[3][2])
    print(label[1][2])
    print(label[2][1])
    print("a")
