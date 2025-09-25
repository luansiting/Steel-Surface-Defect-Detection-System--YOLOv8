import os
from ultralytics import YOLO


class ModelEvaluator:
    def __init__(self, config_path=None, weights_path=None):
        self.config_path = config_path or os.path.join("config", "defect.yaml")
        self.model_path = weights_path or os.path.join(
            "runs/train",
            "yolov8n_steel_defect",
            "weights",
            "best.pt"
        )

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型权重文件不存在: {self.model_path}，请先完成训练")

        self.model = YOLO(self.model_path)

    def evaluate(self):
        """在验证集上计算评估指标"""
        print("开始模型评估...")
        results = self.model.val(data=self.config_path, verbose=False)

        # 提取关键指标
        metrics = {
            "mAP@0.5": results.box.map50,
            "mAP@0.5-0.95": results.box.map50_95,
            "Precision": results.box.precision.mean(),
            "Recall": results.box.recall.mean()
        }

        # 打印评估结果
        print("\n模型评估指标:")
        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")

        return metrics


if __name__ == "__main__":
    evaluator = ModelEvaluator()
    evaluator.evaluate()
