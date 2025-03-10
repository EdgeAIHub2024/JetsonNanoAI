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
labels = open("labels.txt").read().strip().split("\n")

# 初始化摄像头
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# 主循环
prev_time = time.time()
while True:
    # 捕获视频帧
    ret, frame = cap.read()
    if not ret:
        break

    # 预处理帧（调整大小为模型输入）
    input_frame = cv2.resize(frame, (300, 300))
    input_frame = np.expand_dims(input_frame, axis=0).astype(np.uint8)

    # 运行推理
    interpreter.set_tensor(input_details[0]['index'], input_frame)
    interpreter.invoke()
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # 框
    classes = interpreter.get_tensor(output_details[1]['index'])[0]  # 类别
    scores = interpreter.get_tensor(output_details[2]['index'])[0]  # 置信度

    # 可视化检测结果
    for i in range(len(scores)):
        if scores[i] > 0.5:  # 置信度阈值
            ymin, xmin, ymax, xmax = boxes[i]
            (h, w) = frame.shape[:2]
            left, right, top, bottom = int(xmin * w), int(xmax * w), int(ymin * h), int(ymax * h)
            label = f"{labels[int(classes[i])]}: {scores[i]:.2f}"
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, label, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # 记录日志
            logging.info(f"Detected: {label}, Box: ({left}, {top}, {right}, {bottom})")

    # 计算并显示FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # 显示结果
    cv2.imshow("EdgeAIHub Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()