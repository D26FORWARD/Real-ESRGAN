# flake8: noqa  指示flake8忽略此文件的代码风格检查 (因为主要是导入和简单调用)

import os.path as osp  # 导入os.path模块，并使用osp作为别名，用于路径操作
from basicsr.train import train_pipeline  # 从basicsr框架导入核心的训练流程函数

# 导入Real-ESRGAN项目自定义的模块。
# 关键点：仅仅导入这些包就会执行它们各自的 __init__.py 文件。
# 这些 __init__.py 文件负责动态扫描并导入其子目录中具体的实现文件（如网络架构、数据集、模型逻辑等）。
# 在那些具体实现文件中，通常会使用 @ARCH_REGISTRY.register(), @DATASET_REGISTRY.register(), @MODEL_REGISTRY.register()
# 等装饰器将其组件注册到 basicsr 的全局注册表中。
# 因此，这三行导入是确保 Real-ESRGAN 的自定义组件能被 basicsr 框架识别和使用的前提。
import realesrgan.archs  # 导入自定义网络架构模块，使其注册到 ARCH_REGISTRY
import realesrgan.data   # 导入自定义数据处理模块，使其注册到 DATASET_REGISTRY
import realesrgan.models # 导入自定义模型逻辑模块，使其注册到 MODEL_REGISTRY

if __name__ == '__main__':
    # 当此脚本作为主程序执行时运行以下代码

    # 计算项目的根目录路径。
    # __file__ 是当前脚本文件 (realesrgan/train.py) 的路径。
    # osp.pardir 代表上一级目录。
    # osp.join(__file__, osp.pardir, osp.pardir) 表示从当前文件向上回溯两级目录，
    # 即从 realesrgan/train.py -> realesrgan/ -> project_root/。
    # osp.abspath() 将其转换为绝对路径。
    # 这个 root_path 通常会传递给 basicsr 的训练流程，用于定位配置文件、保存模型等。
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))

    # 调用 basicsr 的训练流程函数，并传入项目根路径。
    # basicsr 的 train_pipeline 会处理命令行参数解析 (例如 -opt 指定的配置文件路径)，
    # 并根据配置文件初始化数据加载器、模型、优化器、损失函数等，然后开始训练循环。
    train_pipeline(root_path)
