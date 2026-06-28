import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" # 解决多线程冲突

import torch
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

import torchvision
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torch.utils.data import Dataset, DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision

# ================= 1. 全局配置 =================
BASE_DIR = Path(r"D:\autumn\CS_Experiment\机器视觉_Exp\大作业")
YOLO_DIR = BASE_DIR / "excluded_files" / "dataset_yolo"
REAL_IMG_DIR = BASE_DIR / "real_world_data" / "images"
PROJECT_DIR = BASE_DIR / "results" / "runs" / "ssdlite_baseline"
PROJECT_DIR.mkdir(parents=True, exist_ok=True)
WEIGHT_PATH = PROJECT_DIR / "ssdlite_best.pth"

CLASSES = ['background', 'with_mask', 'without_mask', 'mask_weared_incorrect']
NUM_CLASSES = len(CLASSES)
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# ================= 2. 数据集构建 =================
class MaskDataset(Dataset):
    def __init__(self, split='train'):
        self.img_dir = YOLO_DIR / "images" / split
        self.label_dir = YOLO_DIR / "labels" / split
        self.img_files = list(self.img_dir.glob("*.png"))

    def __len__(self): return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        label_path = self.label_dir / (img_path.stem + ".txt")
        
        img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_h, img_w = img.shape[:2]
        img_tensor = torch.as_tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0

        boxes, labels = [], []
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    class_id, cx, cy, w, h = map(float, line.strip().split())
                    xmin, ymin = (cx - w / 2) * img_w, (cy - h / 2) * img_h
                    xmax, ymax = (cx + w / 2) * img_w, (cy + h / 2) * img_h
                    boxes.append([xmin, ymin, xmax, ymax])
                    labels.append(int(class_id) + 1) # SSD 需要背景类占位 0

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

        return img_tensor, {"boxes": boxes, "labels": labels}

def collate_fn(batch): return tuple(zip(*batch))

# ================= 3. 训练模块 =================
def train_ssdlite(epochs=30):
    print(f"\n[阶段 1/3] 🚀 开始训练 SSD-lite (MobileNetV3) 模型...")
    model = ssdlite320_mobilenet_v3_large(num_classes=NUM_CLASSES, weights_backbone='DEFAULT')
    model.to(DEVICE)
    
    dataset_train = MaskDataset('train')
    data_loader = DataLoader(dataset_train, batch_size=8, shuffle=True, collate_fn=collate_fn)
    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=0.0005, weight_decay=0.0005)
    loss_history = []
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, targets in progress_bar:
            images = list(image.to(DEVICE) for image in images)
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            epoch_loss += losses.item()
            progress_bar.set_postfix(loss=f"{losses.item():.4f}")
            
        avg_epoch_loss = epoch_loss / len(data_loader)
        loss_history.append(avg_epoch_loss)
        
    torch.save(model.state_dict(), WEIGHT_PATH)
    print("🎉 训练完成，模型权重已保存！")
    
    # 绘制并保存 Loss 曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), loss_history, marker='o', linestyle='-', color='#d62728', linewidth=2)
    plt.title('SSD-lite Training Loss Curve', fontsize=16, fontweight='bold')
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Average Loss', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    loss_fig_path = PROJECT_DIR / "ssdlite_loss_curve.png"
    plt.savefig(loss_fig_path, dpi=300, bbox_inches='tight')
    print(f"📈 训练 Loss 曲线已保存至: {loss_fig_path}")

# ================= 4. 评估模块 =================
def evaluate_and_plot():
    print(f"\n[阶段 2/3] 📊 正在验证集上进行推理并计算评估指标...")
    model = ssdlite320_mobilenet_v3_large(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(WEIGHT_PATH, weights_only=True))
    model.to(DEVICE)
    model.eval()

    dataset_val = MaskDataset('val')
    val_loader = DataLoader(dataset_val, batch_size=8, shuffle=False, collate_fn=collate_fn)
    metric = MeanAveragePrecision(iou_type="bbox")

    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="评估进度"):
            images = list(img.to(DEVICE) for img in images)
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
            preds = model(images)
            metric.update(preds, targets)

    results = metric.compute()
    metrics_to_plot = {
        'mAP (IoU=0.5:0.95)': results['map'].item(),
        'mAP@0.5': results['map_50'].item(),
        'mAP@0.75': results['map_75'].item(),
        'mAP (小目标)': results['map_small'].item(),
        'Recall (MaxDet=100)': results['mar_100'].item(),
        'Recall (小目标)': results['mar_small'].item()
    }

    # 绘制指标柱状图
    plt.rcParams['font.sans-serif'] = ['SimHei'] 
    plt.rcParams['axes.unicode_minus'] = False
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(metrics_to_plot.keys(), metrics_to_plot.values(), color=['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974', '#64B5F6'])
    ax.set_title('SSD-lite 验证集精度量化分析', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('Score (0.0 - 1.0)', fontsize=14)
    ax.set_ylim(0, 1.1)
    
    for bar in bars:
        yval = bar.get_height()
        display_val = f"{yval:.3f}" if yval >= 0 else "N/A"
        ax.text(bar.get_x() + bar.get_width()/2, max(yval, 0) + 0.02, display_val, ha='center', va='bottom', fontsize=12, fontweight='bold')
        
    plt.xticks(rotation=15, fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    save_path = PROJECT_DIR / "ssdlite_metrics_analysis.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"📈 评估分析图表已成功保存至: {save_path}")

# ================= 5. 真实数据测试模块 =================
def infer_real_world():
    print(f"\n[阶段 3/3] 📸 正在对自采集图片进行泛化性测试...")
    model = ssdlite320_mobilenet_v3_large(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(WEIGHT_PATH, weights_only=True))
    model.to(DEVICE)
    model.eval() 
    
    colors = {1: (0, 255, 0), 2: (255, 0, 0), 3: (0, 165, 255)}
    test_files = list(REAL_IMG_DIR.glob("*.png")) + list(REAL_IMG_DIR.glob("*.jpg"))
    
    for img_path in tqdm(test_files, desc="推理自采集图片"):
        img_bgr = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img_bgr is None: continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        img_tensor = torch.as_tensor(img_rgb, dtype=torch.float32).permute(2, 0, 1) / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            predictions = model(img_tensor)[0]
            
        boxes = predictions['boxes'].cpu().numpy()
        labels = predictions['labels'].cpu().numpy()
        scores = predictions['scores'].cpu().numpy()
        
        for i, box in enumerate(boxes):
            if scores[i] > 0.4:
                xmin, ymin, xmax, ymax = map(int, box)
                label_id = labels[i]
                label_name = CLASSES[label_id]
                color = colors.get(label_id, (255, 255, 255))
                cv2.rectangle(img_bgr, (xmin, ymin), (xmax, ymax), color, 3)
                cv2.putText(img_bgr, f"{label_name} {scores[i]:.2f}", (xmin, max(ymin-10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
        save_path = PROJECT_DIR / f"ssdlite_res_{img_path.name}"
        cv2.imencode('.jpg', img_bgr)[1].tofile(str(save_path))
        
    print(f"✅ 所有真实场景检测图已保存至: {PROJECT_DIR}")
    print("\n🎉 大作业 SSD-lite 全套对比流水线执行完毕！")

# ================= 主程序执行 =================
if __name__ == "__main__":
    # 【小贴士】：等你这三步都跑完一次后，如果以后只想调整图表或者测试图片，
    # 可以用 # 把 train_ssdlite() 注释掉，这样就不用每次都等它重新训练 30 轮了！
    
    # train_ssdlite(epochs=30)     # 阶段 1：训练并画 Loss
    evaluate_and_plot()          # 阶段 2：计算指标并画图
    infer_real_world()           # 阶段 3：真实场景框图可视化