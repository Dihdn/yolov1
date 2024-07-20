import os
import string
import PIL.Image as Image
import xml.etree.ElementTree as ET#zyz
# 得到文件类型
def get_file_type(path:str) -> str:
    path_split = path.split(".")
    return path_split[-1]

# 得到图片名字
def get_img_name(path):
    path_split = path.split(".")
    return path_split[-2]

# 将原图坐标映射到(s*s)图像中
def get_map_coordinate(origin_coord:tuple, origin_size:tuple, map_size:tuple) -> tuple:
    origin_x, origin_y = origin_coord
    origin_w, origin_h = origin_size
    map_w, map_h = map_size
    scale_w, scale_h = map_w/origin_w, map_h/origin_h
    map_x, map_y = scale_w*origin_x, scale_h*origin_y
    return (map_x, map_y)

# 调整图像尺寸
def resize_img(img_path, new_size:tuple):
    origin_img = Image.open(img_path)
    img = origin_img.resize(new_size)
    return img

# 得到图像的标签
def get_label(xml_path, img_size:tuple, map_size:tuple):
    


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

def get_size_box(file):
    #从xml文件获取size（图片大小，多个方框,位置坐标）

    xml_file = './new_data/train_xml/banana_9.xml'  # 替换为你的XML文件路径
    in_file = open(xml_file, 'r')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    boxs = []
    p = 0
    for obj in root.iter('object'):
        cls = obj.find('name').text
        print(cls + str(p))
        p = p + 1
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        box = [xmin, ymin, xmax, ymax]
        boxs.append(box)
    return (w,h),boxs

def resize_and_draw_box2(input_image_path, output_image_path):
    # 打开图片
    #output_image_path，转化大小后的图片保存路径
    image = Image.open(input_image_path)
    # 定义目标尺寸
    target_size = (448, 448)
    # 获取原始图片尺寸
    original_width, original_height = image.size
    # 计算宽和高的缩放比例
    scale_width = target_size[0] / original_width
    scale_height = target_size[1] / original_height
    # 选择较小的缩放比例，保持宽高比
    scale = min(scale_width, scale_height)
    # 计算新的图片尺寸
    new_size = (int(original_width * scale), int(original_height * scale))
    # 缩小图片，保持宽高比
    image = image.resize(new_size, Image.Resampling.LANCZOS)

    # 创建一个新的黑色背景的图片
    background = Image.new('RGB', target_size, (0, 0, 0))  # 黑色背景

    # 计算粘贴位置使图片居中
    left = (target_size[0] - new_size[0]) // 2
    top = (target_size[1] - new_size[1]) // 2

    # 将缩小后的图片粘贴到黑色背景图片上
    background.paste(image, (left, top))
    background.save(output_image_path)

if __name__ == '__main__':
    path = "E:\\yolov1\\new_data\\train\\train_img\\apple_1.jpg"
    a = resize_img(path, (448, 448))
    print("a")
