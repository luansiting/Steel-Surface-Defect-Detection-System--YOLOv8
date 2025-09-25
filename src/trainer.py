from ultralytics import YOLO
import yaml
import os


class ModelTrainer:
    def __init__(self, config_path=None, weights_path=None, data_root=None):
        # 定义 config_path 属性，根据传入参数赋值
        self.config_path = config_path
        self.weights_path = weights_path
        self.data_root = data_root
        self.model_type = "yolov8n.yaml"
        self.project_dir = "runs/train"
        self.experiment_name = "yolov8n_steel_defect"
        self.device = "cpu"

        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        if config_path is None:
            self.config_path = os.path.join(current_file_dir, "..", "config", "defect.yaml")
        else:
            self.config_path = config_path

        # 验证配置文件存在
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")

        # 新增：打印配置文件里的数据集路径（验证是否正确）
        print("训练集实际路径:", os.path.abspath(os.path.join(os.path.dirname(self.config_path), "data/raw/train/images")))
        print("验证集实际路径:", os.path.abspath(os.path.join(os.path.dirname(self.config_path), "data/raw/valid/images")))

    def train(self):
        """执行模型训练"""
        # 1. 初始化模型（定义model变量）
        model = YOLO(self.model_type)  # 确保在使用model前完成初始化

        # 2. 训练参数配置（data使用配置文件路径字符串）
        train_args = {
            "data": self.config_path,  # 直接传递路径，而非字典
            "epochs": 30,
            "batch": 10,
            "lr0": 0.001,
            "project": self.project_dir,
            "name": self.experiment_name,
            "device": self.device,
            "cache": False,
            "verbose": True
        }

        # 3. 启动训练并使用results变量
        print("开始模型训练...")
        results = model.train(**train_args)

        # 4. 处理训练结果
        result_path = os.path.join(self.project_dir, self.experiment_name)
        print(f"训练完成，结果保存至: {result_path}")
        print(f"最佳模型权重: {os.path.join(result_path, 'weights', 'best.pt')}")

        return results


if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.train()
