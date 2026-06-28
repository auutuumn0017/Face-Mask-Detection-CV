import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

XML_DIR = Path(r"D:\autumn\CS_Experiment\机器视觉_Exp\大作业\excluded_files\annotations")

aspect_ratios = []
center_xs = []
center_ys = []

for xml_file in XML_DIR.glob("*.xml"):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    # 获取图像尺寸用于归一化位置
    size = root.find('size')
    img_w = float(size.find('width').text)
    img_h = float(size.find('height').text)
    
    for obj in root.findall('object'):
        bndbox = obj.find('bndbox')
        xmin, ymin = float(bndbox.find('xmin').text), float(bndbox.find('ymin').text)
        xmax, ymax = float(bndbox.find('xmax').text), float(bndbox.find('ymax').text)
        
        w, h = xmax - xmin, ymax - ymin
        if h > 0 and img_w > 0 and img_h > 0:
            aspect_ratios.append(w / h)
            # 记录中心点相对位置 (0~1)
            center_xs.append((xmin + w/2) / img_w)
            center_ys.append((ymin + h/2) / img_h)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# 图 3：宽高比分布直方图
ax1.hist(aspect_ratios, bins=50, range=(0, 3), color='#8172B2', edgecolor='black', alpha=0.7)
ax1.set_title("图3：目标框宽高比 (Aspect Ratio) 分布", fontsize=14)
ax1.set_xlabel("宽 / 高")
ax1.set_ylabel("频数")
ax1.axvline(x=1.0, color='r', linestyle='--', label='正方形 (1:1)')
ax1.legend()

# 图 4：中心点空间分布散点图
ax2.scatter(center_xs, center_ys, alpha=0.2, s=10, color='#C44E52')
ax2.set_title("图4：目标中心点空间位置分布 (归一化)", fontsize=14)
ax2.set_xlabel("X轴相对位置")
ax2.set_ylabel("Y轴相对位置")
ax2.invert_yaxis() # 图像坐标系Y轴向下

plt.tight_layout()
plt.savefig(XML_DIR.parent.parent / "results" / "eda_extra_results.png", dpi=300)
print("额外EDA图表生成完毕！加上之前的，现在你有4张硬核数据图表了！")
plt.show()