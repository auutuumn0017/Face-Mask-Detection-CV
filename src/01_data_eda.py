import os
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter

# 设置支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 路径配置
BASE_DIR = Path(r"D:\autumn\CS_Experiment\机器视觉_Exp\大作业")
XML_DIR = BASE_DIR / "excluded_files" / "annotations"

def parse_xml_and_analyze():
    classes_count = Counter()
    bbox_areas = []
    
    # 遍历所有的xml文件
    for xml_file in XML_DIR.glob("*.xml"):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # 提取类别和边界框信息
        for obj in root.findall('object'):
            name = obj.find('name').text
            classes_count[name] += 1
            
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            
            # 计算面积 (宽 * 高)
            area = (xmax - xmin) * (ymax - ymin)
            bbox_areas.append(area)
            
    return classes_count, bbox_areas

def plot_eda(classes_count, bbox_areas):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 图1：类别分布柱状图
    classes = list(classes_count.keys())
    counts = list(classes_count.values())
    bars = ax1.bar(classes, counts, color=['#4C72B0', '#DD8452', '#55A868'])
    ax1.set_title("数据集类别数量分布", fontsize=14)
    ax1.set_ylabel("目标框数量")
    for bar in bars:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval + 10, int(yval), ha='center', va='bottom')

    # 图2：目标框大小分布散点/直方图
    ax2.hist(bbox_areas, bins=50, color='#64B5F6', edgecolor='black', alpha=0.7)
    ax2.set_title("目标框面积大小分布 (用于小目标分析)", fontsize=14)
    ax2.set_xlabel("面积 (Pixels)")
    ax2.set_ylabel("频数")
    
    plt.tight_layout()
    plt.savefig(BASE_DIR / "results" / "eda_results.png", dpi=300)
    print("EDA分析完成，图表已保存为 eda_results.png")
    plt.show()

if __name__ == "__main__":
    print("开始解析数据集进行探索性分析...")
    classes_count, bbox_areas = parse_xml_and_analyze()
    print("类别统计结果:", dict(classes_count))
    plot_eda(classes_count, bbox_areas)