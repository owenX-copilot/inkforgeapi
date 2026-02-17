#coding=utf-8
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageOps, ImageFilter
import os
import sys
import argparse
import json

# 添加当前目录到 path 以便能找到 model 包
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model.hanzi_tiny import HanziTiny

# ================= 配置 =================
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints", "best_hanzi_tiny.pth")
IMG_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_MAPPING_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints", "classes.json")

def load_classes():
    """加载类别映射并解码 unicode"""
    if os.path.exists(CLASS_MAPPING_FILE):
        try:
            with open(CLASS_MAPPING_FILE, 'r', encoding='utf-8') as f:
                classes = json.load(f)
            
            # 尝试解码 #Uxxxx 格式
            decoded_classes = []
            for c in classes:
                if c.startswith("#U"):
                    try:
                        decoded_classes.append(chr(int(c[2:], 16)))
                    except:
                        decoded_classes.append(c)
                else:
                    decoded_classes.append(c)
            return decoded_classes
        except Exception as e:
            print(f"⚠️ 读取 {CLASS_MAPPING_FILE} 失败: {e}")
            return []
    else:
        print(f"❌ 找不到类别文件: {CLASS_MAPPING_FILE}")
        return []

TARGET_CLASSES = load_classes()

def preprocess_image(img_path):
    """
    预处理图片，与 GUI 和训练保持一致的逻辑：
    1. 前景背景分离/反色
    2. 自动裁剪 (Auto-Crop)
    3. 模拟模糊 (如果是清晰的电脑字体或画板图)
    4. Resize + Normalize
    """
    try:
        img = Image.open(img_path).convert('L') # 转为灰度
    except Exception as e:
        print(f"❌ 无法打开图片: {img_path} ({e})")
        return None

    # 1. 自动判断这是否是一张“黑底白字”还是“白底黑字”的图
    # 简单的判断方法：看角落像素。通常背景占据边角。
    # 或者直接统计像素均值。
    # 这里我们假设由于我们在 GUI 里处理的是白底黑字，我们尽量把图转成白底黑字来做 crop，然后再看情况。
    # 实际上 HanziTiny 训练时大多是黑底白字 (MNIST风格) 或白底黑字均可，关键是 transform 里的处理。
    # GUI 中使用的是 ToTensor (0~1) -> Normalize((0.5), (0.5)) -> (-1~1)
    
    # 策略：我们先把图统一转成 "白底黑字" (背景255, 字0) 进行裁剪，这是 crop 逻辑最擅长的
    # 如果图片本身是黑底白字 (背景0)，invert 之后就是白底黑字。
    # 检查左上角像素
    pixel_00 = img.getpixel((0, 0))
    if pixel_00 < 128:
        # 假设是黑底白字 -> 先反转成白底黑字以便 crop
        img = ImageOps.invert(img)
    
    # 2. 自动裁剪 (Auto-Crop)
    # 对于白底黑字图，invert 后变为黑底白字，用 getbbox 找非零区域
    inverted = ImageOps.invert(img)
    bbox = inverted.getbbox()
    
    if bbox:
        left, upper, right, lower = bbox
        # 加一点 padding
        p = 10
        left = max(0, left - p)
        upper = max(0, upper - p)
        right = min(img.width, right + p)
        lower = min(img.height, lower + p)
        img = img.crop((left, upper, right, lower))
    
    # 3. 如果这一步得到的图是白底黑字，我们需要考虑模型训练时的输入。
    # 训练时使用了 0.5 mean/std。
    # 关键点：GUI 里在 transform 之前做了一次 filter(GaussianBlur)。
    img = img.filter(ImageFilter.GaussianBlur(radius=0.8))

    # 4. Transform
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # 注意：如果图片是白底(255)，ToTensor后是1.0，Normalize后是(1.0-0.5)/0.5 = 1.0
    # 如果图片是黑字(0)，ToTensor后是0.0，Normalize后是(0.0-0.5)/0.5 = -1.0
    # 只要模型是在这样的分布下训练的即可。
    
    return transform(img).unsqueeze(0).to(DEVICE) # Add batch dim

def predict(img_path):
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 找不到模型文件: {MODEL_PATH}")
        return

    num_classes = len(TARGET_CLASSES)
    if num_classes == 0:
        print("❌ 类别列表为空，无法进行预测。")
        return

    # 加载模型
    print(f"正在加载模型 (Classes: {num_classes})...")
    model = HanziTiny(num_classes=num_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    # 处理图片
    input_tensor = preprocess_image(img_path)
    if input_tensor is None:
        return

    # 预测
    print(f"正在识别: {img_path} ...")
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        
        # Top K
        k = 5
        conf, pred_idx = torch.topk(probs, k)
        
        print("\n=== 识别结果 ===")
        print(f"Top 1: 【 {TARGET_CLASSES[pred_idx[0][0]]} 】 (置信度: {conf[0][0].item()*100:.2f}%)")
        print("-" * 30)
        print("其他候选:")
        for i in range(1, k):
            idx = pred_idx[0][i].item()
            c = TARGET_CLASSES[idx]
            p = conf[0][i].item()
            print(f"{i+1}. {c} \t({p*100:.2f}%)")
        print("=" * 30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='HanziTiny 本地图片识别工具')
    parser.add_argument('-d', '--data', type=str, required=True, help='Path to the image file (jpg/png/bmp)')
    args = parser.parse_args()
    
    predict(args.data)
