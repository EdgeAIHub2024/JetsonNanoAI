import cv2
import numpy as np
import tensorflow as tf
import time
import logging

# 配置日志
logging.basicConfig(filename='detection_log.txt', level=logging.INFO,
                    format='%(asctime)s - %(message)s')

# 加载TensorFlow Lite模型和标签
interpreter = tf.lite.Interpreter(model_path="mobilenet_ssd.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
labels = [line.strip() for line in open("labels.txt").readlines()]

# 初始化摄像头
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("错误: 无法打开摄像头")
    exit()

# 设置窗口
cv2.namedWindow("目标检测", cv2.WINDOW_NORMAL)
cv2.resizeWindow("目标检测", 800, 600)

# 主循环
prev_time = time.time()
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 预处理帧
    input_frame = cv2.resize(frame, (300, 300))
    input_frame = np.expand_dims(input_frame, axis=0).astype(np.uint8)

    # 运行推理
    interpreter.set_tensor(input_details[0]['index'], input_frame)
    interpreter.invoke()
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]

    # 处理检测结果
    for i in range(len(scores)):
        if scores[i] > 0.5:  # 置信度阈值
            class_id = int(classes[i])
            if class_id < 0 or class_id >= len(labels):
                continue
                
            ymin, xmin, ymax, xmax = boxes[i]
            h, w = frame.shape[:2]
            left = max(0, min(int(xmin * w), w-1))
            right = max(0, min(int(xmax * w), w-1))
            top = max(0, min(int(ymin * h), h-1))
            bottom = max(0, min(int(ymax * h), h-1))
            
            label = f"{labels[class_id]}: {scores[i]:.2f}"
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(frame, (left, top - label_size[1] - 10), (left + label_size[0], top), (0, 255, 0), -1)
            cv2.putText(frame, label, (left, top-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            # 记录日志
            logging.info(f"Detected: {label}, Box: ({left}, {top}, {right}, {bottom})")

    # 显示FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # 显示结果
    cv2.imshow("目标检测", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()