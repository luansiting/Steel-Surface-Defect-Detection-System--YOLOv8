import os
import cv2
from ultralytics import YOLO


class DefectDetector:
    def __init__(self, weights_path=None):
        # 优先使用传入的权重路径，否则使用默认路径
        self.model_path = weights_path or os.path.join(
            "runs/train",
            "yolov8n_steel_defect",
            "weights",
            "best.pt"
        )
        self.output_dir = "runs/detect"
        os.makedirs(self.output_dir, exist_ok=True)

        # 验证模型文件是否存在
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型权重文件不存在: {self.model_path}，请先完成训练")

        self.model = YOLO(self.model_path)

    def detect(self, image_path):
        """对单张图像进行缺陷检测"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"待检测图像不存在: {image_path}")

        # 执行检测
        results = self.model(image_path)

        # 保存检测结果
        output_path = os.path.join(self.output_dir, f"result_{os.path.basename(image_path)}")
        results[0].save(output_path)

        print(f"检测完成，结果保存至: {output_path}")
        return output_path


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("使用方式: python src/detector.py <待检测图像路径>")
        sys.exit(1)

    detector = DefectDetector()
    detector.detect(sys.argv[1])
