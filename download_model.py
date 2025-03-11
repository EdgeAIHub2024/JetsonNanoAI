import urllib.request
import os

def download_file(url, filename):
    print(f"下载 {filename}...")
    urllib.request.urlretrieve(url, filename)
    print(f"{filename} 下载完成")

# 下载模型文件
model_url = "https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip"
zip_file = "mobilenet_ssd.zip"

# 下载并解压模型
download_file(model_url, zip_file)

# 解压文件
import zipfile
with zipfile.ZipFile(zip_file, 'r') as zip_ref:
    zip_ref.extractall('.')

# 重命名模型文件
os.rename('detect.tflite', 'mobilenet_ssd.tflite')

# 创建标签文件
labels = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
]

with open('labels.txt', 'w') as f:
    for label in labels:
        f.write(f"{label}\n")

print("模型和标签文件设置完成！")

# 清理zip文件
os.remove(zip_file)
print("清理完成！") 