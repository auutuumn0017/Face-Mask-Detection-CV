import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
import random
from tqdm import tqdm

# 路径配置
BASE_DIR = Path(r"D:\autumn\CS_Experiment\机器视觉_Exp\大作业")
XML_DIR = BASE_DIR / "excluded_files" / "annotations"
IMG_DIR = BASE_DIR / "excluded_files" / "images"
YOLO_DIR = BASE_DIR / "excluded_files" / "dataset_yolo"

# 类别映射 (非常重要：YOLO是用数字代替类别的)
CLASSES = ['with_mask', 'without_mask', 'mask_weared_incorrect']
CLASS_DICT = {c: i for i, c in enumerate(CLASSES)}

def convert_box(size, box):
    # YOLO格式归一化: [x_center, y_center, width, height]
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    return (x * dw, y * dh, w * dw, h * dh)

def setup_directories():
    # 创建YOLO格式需要的目录树
    dirs = [
        YOLO_DIR / "images" / "train", YOLO_DIR / "images" / "val",
        YOLO_DIR / "labels" / "train", YOLO_DIR / "labels" / "val"
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

def process_dataset(val_ratio=0.2):
    setup_directories()
    
    # 获取所有图片列表
    xml_files = list(XML_DIR.glob("*.xml"))
    random.shuffle(xml_files)
    
    val_size = int(len(xml_files) * val_ratio)
    val_files = xml_files[:val_size]
    train_files = xml_files[val_size:]
    
    print(f"总文件数: {len(xml_files)} | 训练集: {len(train_files)} | 验证集: {len(val_files)}")
    
    def parse_and_copy(files, split_type):
        for xml_file in tqdm(files, desc=f"Processing {split_type}"):
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            size = root.find('size')
            w = int(size.find('width').text)
            h = int(size.find('height').text)
            
            img_name = xml_file.stem + ".png"
            src_img = IMG_DIR / img_name
            
            if not src_img.exists():
                continue
                
            # 写入 YOLO 标签文件
            label_file = YOLO_DIR / "labels" / split_type / (xml_file.stem + ".txt")
            with open(label_file, 'w', encoding='utf-8') as out_file:
                for obj in root.iter('object'):
                    difficult = obj.find('difficult').text if obj.find('difficult') is not None else '0'
                    cls_name = obj.find('name').text
                    if cls_name not in CLASSES or int(difficult) == 1:
                        continue
                        
                    cls_id = CLASS_DICT[cls_name]
                    xmlbox = obj.find('bndbox')
                    b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), 
                         float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
                    
                    bb = convert_box((w, h), b)
                    out_file.write(f"{cls_id} {' '.join([str(a) for a in bb])}\n")
            
            # 复制图片到对应目录
            dst_img = YOLO_DIR / "images" / split_type / img_name
            shutil.copy(src_img, dst_img)

    parse_and_copy(train_files, "train")
    parse_and_copy(val_files, "val")

def create_yaml():
    yaml_content = f"""path: {YOLO_DIR}
train: images/train
val: images/val

names:
  0: with_mask
  1: without_mask
  2: mask_weared_incorrect
"""
    with open(YOLO_DIR / "data.yaml", 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    print("\ndata.yaml 文件已生成！")

if __name__ == "__main__":
    process_dataset()
    create_yaml()
    print("数据集转换与划分完成，准备进行训练。")