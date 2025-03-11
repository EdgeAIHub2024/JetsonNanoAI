# 实时目标检测系统

基于TensorFlow Lite和OpenCV的实时目标检测系统，使用MobileNet SSD模型进行物体检测。

## 功能特点

- 实时摄像头目标检测
- 支持80种常见物体识别
- 显示检测置信度和FPS
- 自动记录检测日志

## 环境要求

- Python 3.8+
- OpenCV
- TensorFlow 2.x
- NumPy

## 安装步骤

1. 克隆仓库：
```bash
git clone https://github.com/你的用户名/你的仓库名.git
cd 你的仓库名
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 下载模型：
```bash
python download_model.py
```

## 使用方法

运行主程序：
```bash
python main.py
```

- 按'q'键退出程序
- 检测结果将保存在detection_log.txt中

## 文件说明

- `main.py`: 主程序
- `download_model.py`: 模型下载脚本
- `requirements.txt`: 依赖包列表
- `labels.txt`: 标签文件
- `mobilenet_ssd.tflite`: 预训练模型

## 注意事项

- 确保摄像头权限正确设置
- 首次运行需要下载模型文件
- 日志文件会持续增长，注意定期清理

## License

MIT License
