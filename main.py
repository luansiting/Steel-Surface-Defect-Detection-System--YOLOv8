pip install PyQt5==5.13 PyQtWebEngine==5.13 watchdog==2.0.0import argparse
import sys
import os
import traceback

# 修复导入路径问题（针对根目录结构）
current_dir = os.path.dirname(os.path.abspath(__file__))

# 添加所有必要的根目录模块到系统路径
sys.path.append(current_dir)  # 当前目录
sys.path.append(os.path.join(current_dir, 'src'))  # src目录
sys.path.append(os.path.join(current_dir, 'utils'))  # utils目录（与src同级）

# 修正导入语句（直接导入根目录下的模块）
try:
    from data_prep import DataPreprocessor
    from trainer import ModelTrainer
    from detector import DefectDetector
    from utils.metrics import ModelEvaluator  # utils目录下的metrics模块
    from utils.visualize import ResultVisualizer  # utils目录下的visualize模块
except ImportError as e:
    print(f"导入错误: {str(e)}", file=sys.stderr)
    print("当前系统路径:", sys.path, file=sys.stderr)
    print("请确保项目结构正确：src/, utils/ 目录存在且包含相应模块", file=sys.stderr)
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="钢材缺陷检测系统主程序")
    parser.add_argument('--action', required=True,
                        choices=['preprocess', 'train', 'evaluate', 'detect', 'visualize'],
                        help="要执行的操作：preprocess(数据预处理)、train(模型训练)、evaluate(模型评估)、detect(缺陷检测)、visualize(结果可视化)")
    parser.add_argument('--image', help="当action为detect或visualize时，指定待处理图像路径")

    # 重要参数（所有操作都需要）
    parser.add_argument('--weights', default='weights/best.pt',
                        help="模型权重文件路径 (默认: weights/best.pt)")
    parser.add_argument('--config', default='config/defect.yaml',
                        help="配置文件路径 (默认: config/defect.yaml)")
    parser.add_argument('--data-root', default='data', help="数据根目录路径")

    args = parser.parse_args()

    # 路径验证（增强健壮性）
    if not os.path.exists(args.config):
        print(f"错误：配置文件不存在 {args.config}", file=sys.stderr)
        sys.exit(1)

    if args.action != 'train' and not os.path.exists(args.weights):
        print(f"错误：权重文件不存在 {args.weights}", file=sys.stderr)
        print("请先运行训练操作或指定正确的权重路径", file=sys.stderr)
        sys.exit(1)

    # 图像操作的特殊检查
    if args.action in ['detect', 'visualize']:
        if not args.image:
            print("错误：执行detect或visualize时必须使用--image指定图像路径", file=sys.stderr)
            sys.exit(1)
        if not os.path.exists(args.image):
            print(f"错误：图像文件不存在 {args.image}", file=sys.stderr)
            sys.exit(1)

    try:
        # 各操作处理器（传递所有必要参数）
        if args.action == 'preprocess':
            print("===== 开始数据预处理 =====")
            preprocessor = DataPreprocessor(
                data_root=args.data_root,
                config_path=args.config
            )
            preprocessor.process_all_splits()
            print("===== 数据预处理完成 =====")

        elif args.action == 'train':
            print("===== 开始模型训练 =====")
            trainer = ModelTrainer(
                data_root=args.data_root,
                config_path=args.config,
                weights_path=args.weights
            )
            trainer.train()
            print("===== 模型训练完成 =====")

        elif args.action == 'evaluate':
            print("===== 开始模型评估 =====")
            evaluator = ModelEvaluator(
                data_root=args.data_root,
                weights_path=args.weights,
                config_path=args.config
            )
            metrics = evaluator.evaluate()
            print(f"===== 模型评估完成 =====")
            print(f"评估结果: mAP={metrics['map']:.4f}, Recall={metrics['recall']:.4f}")

        elif args.action == 'detect':
            print(f"===== 开始对图像 {args.image} 进行缺陷检测 =====")
            detector = DefectDetector(
                weights_path=args.weights,
                config_path=args.config
            )
            result_path = detector.detect(args.image)
            print(f"===== 缺陷检测完成，结果保存至 {result_path} =====")

        elif args.action == 'visualize':
            print(f"===== 开始对图像 {args.image} 进行结果可视化 =====")
            visualizer = ResultVisualizer(
                config_path=args.config
            )
            vis_path = visualizer.visualize(args.image)
            print(f"===== 结果可视化完成，保存至 {vis_path} =====")

    except Exception as e:
        # 详细错误报告（方便调试）
        print(f"\n{'=' * 50}", file=sys.stderr)
        print(f"操作失败: {str(e)}", file=sys.stderr)
        traceback.print_exc()
        print(f"{'=' * 50}\n", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()