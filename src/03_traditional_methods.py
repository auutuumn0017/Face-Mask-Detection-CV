import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from skimage.feature import hog
import joblib
import matplotlib.pyplot as plt
from tqdm import tqdm

# ================= 路径配置 =================
BASE_DIR = Path(r"D:\autumn\CS_Experiment\机器视觉_Exp\大作业")
XML_DIR = BASE_DIR / "excluded_files" / "annotations"
IMG_DIR = BASE_DIR / "excluded_files" / "images"
REAL_IMG_DIR = BASE_DIR / "real_world_data" / "images"
MODEL_DIR = BASE_DIR / "src" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
PROJECT_DIR = BASE_DIR / "results" / "runs" / "traditional_baseline"
PROJECT_DIR.mkdir(parents=True, exist_ok=True)

class TraditionalDetectors:
    def __init__(self):
        # 初始化 Haar 级联分类器
        face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(face_cascade_path)
        
        self.svm_model = None
        self.rf_model = None
        
        # HOG 参数：保持高维度特征提取
        self.hog_params = {'orientations': 9, 'pixels_per_cell': (8, 8), 'cells_per_block': (2, 2), 'channel_axis': -1}

    # ========================== 训练阶段 ==========================
    def extract_features_from_gt(self):
        print("正在从全量数据集中提取特征 (这可能需要几分钟)...")
        X_hog, X_color, y = [], [], []
        xml_files = list(XML_DIR.glob("*.xml"))
        
        for xml_file in tqdm(xml_files):
            tree = ET.parse(xml_file)
            root = tree.getroot()
            img_path = IMG_DIR / (xml_file.stem + ".png")
            
            # 使用 imdecode 完美解决中文路径问题
            img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None: continue
            
            for obj in root.findall('object'):
                name = obj.find('name').text
                label = 1 if name == 'with_mask' else 0 
                
                bndbox = obj.find('bndbox')
                xmin, ymin = int(bndbox.find('xmin').text), int(bndbox.find('ymin').text)
                xmax, ymax = int(bndbox.find('xmax').text), int(bndbox.find('ymax').text)
                
                crop = img[ymin:ymax, xmin:xmax]
                if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10: continue
                
                crop_resized = cv2.resize(crop, (64, 64))
                
                # 特征 1：HOG
                hog_feat = hog(crop_resized, **self.hog_params)
                X_hog.append(hog_feat)
                
                # 特征 2：HSV 直方图
                hsv_crop = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2HSV)
                hist = cv2.calcHist([hsv_crop], [0, 1], None, [16, 16], [0, 180, 0, 256]).flatten()
                X_color.append(hist)
                
                y.append(label)
                
        return np.array(X_hog), np.array(X_color), np.array(y)

    def train_classifiers(self):
        X_hog, X_color, y = self.extract_features_from_gt()
        
        print("正在训练 HOG + SVM 分类器...")
        self.svm_model = SVC(kernel='linear', probability=True)
        self.svm_model.fit(X_hog, y)
        joblib.dump(self.svm_model, MODEL_DIR / 'svm_hog.pkl')
        
        print("正在训练 特征融合 + 随机森林...")
        self.rf_model = RandomForestClassifier(n_estimators=150, max_depth=15, random_state=42)
        X_combined = np.hstack((X_hog, X_color)) 
        self.rf_model.fit(X_combined, y)
        joblib.dump(self.rf_model, MODEL_DIR / 'rf_combined.pkl')
        print("模型训练完毕！\n")

    def load_models(self):
        try:
            self.svm_model = joblib.load(MODEL_DIR / 'svm_hog.pkl')
            self.rf_model = joblib.load(MODEL_DIR / 'rf_combined.pkl')
            print("已成功加载预训练的传统分类器模型。")
        except:
            print("未找到模型，自动启动全量训练流程...")
            self.train_classifiers()

    # ========================== 推理阶段 ==========================
    
    # 策略 1: Haar 定位 + 区域方差分析 (抛弃死板的颜色)
    def detect_method_1(self, img):
        results = []
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 优化：增大 minNeighbors 过滤背景误检
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=6, minSize=(40, 40))
        
        for (x, y, w, h) in faces:
            # 截取人脸下半部分 (口鼻区域)
            lower_half = img[y + int(h*0.5) : y + h, x : x + w]
            if lower_half.size == 0: continue
            
            # 使用灰度方差判断纹理。口罩表面通常平滑(方差小)，而嘴巴鼻子纹理复杂(方差大)
            gray_lower = cv2.cvtColor(lower_half, cv2.COLOR_BGR2GRAY)
            variance = np.var(gray_lower)
            
            label = 'with_mask' if variance < 1200 else 'without_mask'
            results.append((x, y, w, h, label))
        return results

    # 策略 2: HOG + 滑动窗口 + SVM + NMS (非极大值抑制)
    def detect_method_2(self, img):
        boxes, scores = [], []
        window_size = (64, 64)
        step_size = 24  # 缩小步长提升精度
        
        # 为提高速度和精度，将图像适度缩小进行滑窗
        scale = 0.5
        small_img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
        h, w = small_img.shape[:2]
        
        for y in range(0, h - window_size[1], step_size):
            for x in range(0, w - window_size[0], step_size):
                window = small_img[y:y+window_size[1], x:x+window_size[0]]
                hog_feat = hog(window, **self.hog_params).reshape(1, -1)
                
                prob = self.svm_model.predict_proba(hog_feat)[0][1]
                # 严格阈值
                if prob > 0.90: 
                    # 映射回原图坐标
                    orig_x, orig_y = int(x / scale), int(y / scale)
                    orig_w, orig_h = int(window_size[0] / scale), int(window_size[1] / scale)
                    boxes.append([orig_x, orig_y, orig_w, orig_h])
                    scores.append(float(prob))
                    
        # === 核心优化：使用 NMS 剔除重叠框 ===
        results = []
        if len(boxes) > 0:
            indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.90, nms_threshold=0.2)
            if len(indices) > 0:
                for i in indices.flatten():
                    x, y, w, h = boxes[i]
                    results.append((x, y, w, h, 'with_mask'))
        return results

    # 策略 3: Haar 提取 RoI + 随机森林
    def detect_method_3(self, img):
        results = []
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=6, minSize=(40, 40))
        
        for (x, y, w, h) in faces:
            crop = img[y:y+h, x:x+w]
            crop_resized = cv2.resize(crop, (64, 64))
            
            hog_feat = hog(crop_resized, **self.hog_params)
            hsv_crop = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv_crop], [0, 1], None, [16, 16], [0, 180, 0, 256]).flatten()
            
            X_test = np.hstack((hog_feat, hist)).reshape(1, -1)
            pred = self.rf_model.predict(X_test)[0]
            label = 'with_mask' if pred == 1 else 'without_mask'
            results.append((x, y, w, h, label))
        return results

# ========================== 可视化模块 ==========================
def visualize_results(img_path, detector, save_path=None, show=False):
    img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None: return
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    print(f"正在分析图像: {img_path.name}")
    res1 = detector.detect_method_1(img)
    res2 = detector.detect_method_2(img)
    res3 = detector.detect_method_3(img)
    
    # 改为 1行3列 的宽幅排版，避免图片被压缩
    fig, axs = plt.subplots(1, 3, figsize=(24, 8))
    methods = [(res1, "Method 1: Haar + Texture Variance"), 
               (res2, "Method 2: HOG + SVM + NMS"), 
               (res3, "Method 3: Haar + Random Forest")]
    
    for ax, (res, title) in zip(axs, methods):
        img_draw = img_rgb.copy()
        for (x, y, w, h, label) in res:
            color = (0, 255, 0) if label == 'with_mask' else (255, 0, 0)
            cv2.rectangle(img_draw, (x, y), (x+w, y+h), color, 4)
            cv2.putText(img_draw, label, (x, max(y-10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
            
        ax.imshow(img_draw)
        ax.set_title(title, fontsize=20, fontweight='bold', pad=20)
        ax.axis('off')
        
    plt.tight_layout(w_pad=3.0)
    
    if save_path is None:
        save_path = BASE_DIR / "results" / "traditional_methods_optimized.png"
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"对比图已保存至: {save_path}")
    if show:
        plt.show()
    plt.close(fig) # 防止内存泄漏

def infer_real_world(detector):
    print(f"\n📸 正在对自采集图片进行泛化性测试 (传统方法)...")
    test_files = list(REAL_IMG_DIR.glob("*.png")) + list(REAL_IMG_DIR.glob("*.jpg"))
    for img_path in test_files:
        save_path = PROJECT_DIR / f"trad_res_{img_path.name}"
        visualize_results(img_path, detector, save_path=save_path, show=False)
    print(f"✅ 所有传统方法真实场景检测图已保存至: {PROJECT_DIR}")

if __name__ == "__main__":
    detector = TraditionalDetectors()
    detector.load_models() 
    
    # 1. 找一张公开数据集的图测试并显示
    test_images = list(IMG_DIR.glob("*.png"))
    if test_images:
        visualize_results(test_images[0], detector, show=True)
        
    # 2. 对自采集的真实图片进行批量推理，不弹窗显示
    infer_real_world(detector)