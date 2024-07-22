import os
import string
import torch
import PIL.Image as Image
import xml.etree.ElementTree as ET#zyz


# 得到文件类型
def get_file_type(path:str) -> str:
    # path:文件名或路径
    # return 文件类型
    path_split = path.split(".")
    return path_split[-1]


# 得到图片名字
def get_img_name(path):
    # path: 图片的路径
    # return 图片名字
    path_split = path.split(".")
    return path_split[-2]


# 调整图像尺寸
def resize_img(img_path, new_size:tuple):
    # img_path:文件路径  new_size:调整后的图片尺寸
    # return 调整尺寸后的图片
    origin_img = Image.open(img_path)
    img = origin_img.resize(new_size)
    return img


# 将原图坐标映射到(s*s)图像中
def get_map_coordinate(origin_coord:tuple, origin_size:tuple, map_size:tuple) -> tuple:
    # origin_coord:原图中的坐标 元组表示(x, y)  # origin_size:原始图片尺寸  元组表示(w, h)  # map_size:要映射到图片的层次 元组表示(w, h)
    # return s*s中的坐标(x, y)
    origin_x, origin_y = origin_coord
    origin_w, origin_h = origin_size
    map_w, map_h = map_size
    scale_w, scale_h = map_w/origin_w, map_h/origin_h
    map_x, map_y = scale_w*origin_x, scale_h*origin_y
    return (map_x, map_y)


# 将原图中的框映射到s*s图像中
def get_map_size(origin_size:tuple, map_size:tuple, box_size:tuple):
    # origin_size:原始图像尺寸 元组表示(w, h)  map_size:要映射的图像尺寸 元组表示(w, h)  box_size:原始图像中框的尺寸 元组表示(w, h)
    # return map_size中图像的尺寸 元组表示(w, h)
    scale_w, scale_h = map_size[0]/origin_size[0], map_size[1]/origin_size[1]
    map_box_size = (box_size[0]*scale_w, box_size[1]*scale_h)
    return map_box_size


# 将类别转成one-hot类
def convert_class_to_one_hot(class_name:str, class_list:list):
    # class_name:类别名称 class_list：所有类别组成的列表
    # return 类别的one-hot类型
    index = class_list.index(class_name)
    one_hot = torch.zeros(size=(len(class_list), ))
    one_hot[index] = 1
    one_hot = one_hot.detach().tolist()
    return one_hot


# 分离小数与整数部分
def separate_int_float(num):
    # num:要分离的数
    # return 整数和小数部分 元组表示(整数， 小数部分)
    return (int(num), num%1)

# 得到图像的标签
def get_label(xml_path, img_size:tuple, resize, map_size:tuple, class_list:list):
    # xml_path:xml文件路径  img_size:原始图片尺寸 元组表示(w, h)  map_size:映射图片尺寸 元组表示(w, h)
    # 得到图片相关信息
    img_size, objects = get_size_box(xml_path=xml_path)
    # 创建一个与map_size一样大的空白布
    size = (map_size[0], map_size[1], 5+len(class_list))
    blank_canvas = torch.zeros(size=size)
    for obj in objects:
        x_min, y_min, x_max, y_max, class_name = obj
        # 框的宽高
        bndbox_w, bndbox_h = x_max-x_min, y_max-y_min
        # 将框的宽高映射到7*7图像中
        box_w, box_h = get_map_size(origin_size=img_size, map_size=map_size, box_size=(bndbox_w, bndbox_h))
        # 框的中心坐标映射到(448*448)图像中
        central_x, central_y = get_map_coordinate(origin_coord=((x_min+x_max)/2, (y_min+y_max)/2), origin_size=img_size, map_size=resize)
        # 计算中心点在7*7中的偏移量
        map_x, map_y = central_x/(resize[0]/map_size[0]), central_y/(resize[1]/map_size[1])
        map_x_int, map_x_float = separate_int_float(map_x)
        map_y_int, map_y_float = separate_int_float(map_y)
        # 将类别转成one-hot类型
        one_hot = convert_class_to_one_hot(class_name=class_name, class_list=class_list)
        confidence = 1
        # 将标签数据填入对应的布中
        label_obj = []
        label_obj.append(map_x_float)
        label_obj.append(map_y_float)
        label_obj.append(box_w)
        label_obj.append(box_h)
        label_obj.append(confidence)
        label_obj += one_hot
        blank_canvas[map_x_int][map_y_int] = torch.tensor(label_obj)
    return blank_canvas



###zyz

def convert(size, box):
    #size格式{w,h},box格式[xmin,ymin,xmax,ymax]
    #利用448*448,转换到448*448的相对大小
    #得到方框在，448*448位置
    original_width, original_height = size[0], size[1]
    target_size=[448,448]
    # 计算宽和高的缩放比例
    scale_width = target_size[0] / original_width
    scale_height = target_size[1] / original_height
    scale = min(scale_width, scale_height)
    new_size = (int(original_width * scale), int(original_height * scale))
    # 选择较小的缩放比例，保持宽高比

    left = (target_size[0] - new_size[0]) // 2
    top = (target_size[1] - new_size[1]) // 2
    box_coordinates = [
        (int(left + x * scale), int(top + y * scale)) for x, y in box
    ]
    return box_coordinates

def convert2(box):
    #将位置转化为（中心点，方框宽度，高度）并归一化
    #box=[xmin,ymin,xmax,ymax]
    x,y=(box[2]+box[0])/2,(box[1]+box[3])/3
    x=1./448*x
    y=1./448*y
    h=box[3]-box[1]
    w=box[2]-box[0]
    return (x,y,h,w)

#从一个xml文件获取size（图片大小，多个方框,位置坐标）
def get_size_box(xml_path):
    #返回对应图片的信息(with,hight)[xmin,ymin,xmax,ymax,种类】
    in_file = open(xml_path, 'r')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    boxs = []
    for obj in root.iter('object'):
        cls = obj.find('name').text
        cls2=cls
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        box = [xmin, ymin, xmax, ymax,cls2]
        boxs.append(box)
    return (w,h),boxs



if __name__ == '__main__':
    path = "E:\\yolov1\\new_data\\train\\train_img\\apple_2.jpg"
    a = resize_img(path, (448, 448))
    print("a")
