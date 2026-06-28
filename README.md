# 口罩佩戴目标检测大作业

## 1. 项目基本信息
- **项目名称**: 口罩佩戴目标检测与多模型对比分析
- **所选主题**: 题目二：目标检测任务——口罩佩戴目标检测
- **小组成员**: 艾洋 (235127) - 单人组
- **成员分工**: 单人独立完成全部任务，包括数据处理、传统算法实现、深度学习实现、真实数据采集与验证、实验报告撰写等。

## 2. 运行环境与依赖库
本项目可在本地 CPU 或带有 GPU 加速的环境（如 Kaggle、Colab 或本地 CUDA 环境）中运行。
- Python 3.8+
- OpenCV (`opencv-python`)
- Scikit-Learn (`scikit-learn`)
- Scikit-Image (`scikit-image`)
- Matplotlib, numpy, tqdm
- Ultralytics (YOLOv8): `pip install ultralytics`
- PyTorch & Torchvision: `pip install torch torchvision`
- TorchMetrics: `pip install torchmetrics`

## 3. 数据集说明
- **原数据集获取方式**: 原始数据集来自 Kaggle 面罩检测数据集。因报告要求不提交原数据集，详情见 `data/README.md` 中的下载链接。
- **真实数据采集说明**: 请参考 `real_world_data/README_real_data.md`。

## 4. 目录结构
- `data/`: 原数据集说明链接（已清空原始庞大数据防止扣分）。
- `excluded_files/`: (不提交的文件暂存区，包括 YOLO 格式数据和原 XML)。
- `real_world_data/`: 自行采集的真实场景验证图像及其说明。
- `results/`: 各个模型的训练指标、曲线图以及针对自拍图像的测试输出图片。
- `src/`: 核心源代码（数据处理、传统算法、YOLOv8、SSD-Lite）。

## 5. 主要代码运行步骤

### (1) 传统方法测试
```bash
python src/03_traditional_methods.py
```
运行后，将自动加载预训练好的特征提取与分类模型，并对真实数据进行画框预测，结果保存在 `results/runs/traditional_baseline/`。

### (2) YOLOv8 训练与推理
```bash
python src/04_yolov8_pipeline.py
```
代码中包含了完整的训练和验证功能。如仅需查看自采集图片测试效果，可只执行文件底部的 `infer_real_world()`，结果保存在 `results/runs/yolov8n_real_world_test/`。

### (3) SSD-Lite 训练与推理
```bash
python src/05_ssdlite_pipeline.py
```
结果及评价图表保存在 `results/runs/ssdlite_baseline/`。

## 6. 主要实验结果说明
- **YOLOv8n**: 在基础验证集上表现出色，mAP 很高。
- **SSD-Lite**: 通过手写训练流程实现了次佳性能，对小目标检测进行了专项评估。
- **传统方法**: HOG+SVM 和 随机森林 等方法在特征复杂时容易受到背景干扰，在自采集数据上的泛化能力弱于深度学习。
- **详细图文分析请参阅根目录下的 PDF 项目报告**。
