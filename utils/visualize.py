import os
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO


class ResultVisualizer:
    def __init__(self, weights_path=None):
        self.model_path = weights_path or os.path.join(
            "runs/train",
            "yolov8n_steel_defect",
            "weights",
            "best.pt"
        )
        self.visualize_dir = "runs/visualize"
        os.makedirs(self.visualize_dir, exist_ok=True)

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型权重文件不存在: {self.model_path}，请先完成训练")

        self.model = YOLO(self.model_path)

    def visualize(self, image_path):
        """可视化带预测框的检测结果"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图像文件不存在: {image_path}")

        # 执行检测并获取结果
        results = self.model(image_path)[0]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 绘制预测框和标签
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls)
            conf = float(box.conf)

            # 绘制边界框
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 绘制类别和置信度
            label = f"{self.model.names[cls_id]}: {conf:.2f}"
            cv2.putText(image, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 保存可视化结果
        output_path = os.path.join(self.visualize_dir, f"vis_{os.path.basename(image_path)}")
        plt.figure(figsize=(10, 8))
        plt.imshow(image)
        plt.axis("off")
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()

        print(f"可视化结果保存至: {output_path}")
        return output_path


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("使用方式: python utils/visualize.py <待可视化图像路径>")
        sys.exit(1)

    visualizer = ResultVisualizer()
    visualizer.visualize(sys.argv[1])
