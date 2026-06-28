import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from pathlib import Path
from ultralytics import YOLO


# ================= 路径配置 =================
BASE_DIR = Path(r"D:\autumn\CS_Experiment\机器视觉_Exp\大作业")
YAML_PATH = BASE_DIR / "excluded_files" / "dataset_yolo" / "data.yaml"
REAL_IMG_DIR = BASE_DIR / "real_world_data" / "images"  # 你自己拍的那20张照片

# YOLO训练结果默认保存在 runs/detect/ 目录下
PROJECT_DIR = BASE_DIR / "results" / "runs"

def train_yolov8():
    """
    第一阶段：模型训练
    """
    print("🚀 开始加载 YOLOv8n 预训练模型...")
    # 加载官方极轻量级预训练模型 (会自动下载 yolov8n.pt)
    model = YOLO("yolov8n.pt") 
    
    print("🔥 开始训练 YOLOv8n (大作业基础实验)...")
    # 开始训练
    results = model.train(
        data=str(YAML_PATH),
        epochs=30,             # 训练轮数：对于简单口罩检测，30-50轮足够收敛
        imgsz=640,             # 图像输入尺寸
        batch=16,              # 批次大小 (如果显存报错，可改为 8 或 4)
        project=str(PROJECT_DIR), # 保存路径
        name="yolov8n_baseline",  # 实验名称
        device="0",            # 如果有显卡写 "0"，纯CPU写 "cpu"
        workers=0              # Windows系统建议设为0防止多线程报错
    )
    print("✅ 训练完成！结果已保存在:", PROJECT_DIR / "yolov8n_baseline")

def eval_and_compare_yolov8():
    """
    第二阶段：指标评估与参数对比实验 (大作业加分项)
    """
    # 加载我们刚刚训练好的最佳模型权重
    best_weight_path = PROJECT_DIR / "yolov8n_baseline" / "weights" / "best.pt"
    model = YOLO(str(best_weight_path))
    
    print("\n📊 正在测试集上评估各项指标 (Precision, Recall, mAP)...")
    # 运行验证，它会自动计算并在控制台打印出各个类别的 P, R, mAP@0.5, mAP@0.5:0.95
    metrics = model.val(project=str(PROJECT_DIR), name="yolov8n_eval")
    
    # ------------------ 大作业要求：对比实验 ------------------
    print("\n🔬 正在进行【对比实验】：降低推理置信度阈值...")
    # 对比实验：改变置信度阈值 (conf=0.1)，观察它如何提高 Recall 但降低 Precision (误检增加)
    model.val(conf=0.1, project=str(PROJECT_DIR), name="yolov8n_eval_low_conf")

def infer_real_world():
    """
    第三阶段：真实自采集数据泛化性验证
    """
    best_weight_path = PROJECT_DIR / "yolov8n_baseline" / "weights" / "best.pt"
    if not best_weight_path.exists():
        print("未找到训练好的模型，请先运行 train_yolov8()")
        return
        
    model = YOLO(str(best_weight_path))
    
    print(f"\n📸 正在对你自采集的真实场景图片进行推理...")
    print(f"数据目录: {REAL_IMG_DIR}")
    
    # 对 myself_images 文件夹下的所有图片进行推理
    # save=True 会自动把画好检测框的图片保存下来
    results = model.predict(
        source=str(REAL_IMG_DIR),
        conf=0.4,             # 置信度阈值
        iou=0.45,             # NMS IoU 阈值
        save=True,            # 保存结果图
        project=str(PROJECT_DIR),
        name="yolov8n_real_world_test"
    )
    print("\n🎉 全部流程结束！请前往 runs 目录查看可视化结果和训练指标图表。")

if __name__ == "__main__":
    # # 第一次运行，请解除 train_yolov8() 的注释
    # train_yolov8()
    
    # 训练完成后，可以注释掉 train，单独运行下面两个函数
    eval_and_compare_yolov8()
    infer_real_world()