from ultralytics import YOLO

def train_yolov8_pose():
    # 加载预训练模型
    model = YOLO("yolov8n-pose.pt")   # 可换成 yolov8s-pose.pt 等

    # 开始训练
    model.train(
        data="datasets/train.yaml",  # 数据集配置文件路径
        epochs=100,                    # 训练轮数
        imgsz=640,                     # 输入图片大小
        batch=16,                      # 批次大小
        device=0,                      # 训练设备（0 表示 GPU0）
        workers=8,                     # 线程数
        name="pose_finetune",          # 保存文件夹名 runs/pose_finetune/
        cache=True,                    # 提前缓存图片
        pretrained=True                # 保留预训练权重参数
    )

if __name__ == "__main__":
    train_yolov8_pose()
