import importlib  # 导入 importlib 模块，用于动态导入模块
from basicsr.utils import scandir  # 从 basicsr.utils 导入 scandir 函数，用于扫描目录内容
from os import path as osp  # 从 os.path 导入 path，并重命名为 osp，用于路径操作

# 自动扫描并导入模型模块以进行注册
# 扫描 models 文件夹下所有以 '_model.py' 结尾的文件

# 获取当前文件（__init__.py）所在的目录的绝对路径，即 models 文件夹的路径
model_folder = osp.dirname(osp.abspath(__file__))

# 使用 scandir 遍历 model_folder 中的所有文件和目录
# v 是遍历到的每个文件/目录的完整路径
# osp.basename(v) 获取文件名（例如 'realesrgan_model.py'）
# osp.splitext(osp.basename(v))[0] 获取文件名去除扩展名后的部分（例如 'realesrgan_model'）
# if v.endswith('_model.py') 确保只处理以 '_model.py' 结尾的文件
model_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(model_folder) if v.endswith('_model.py')]

# 动态导入所有找到的模型模块
# file_name 是上一步中获取的无扩展名的文件名，如 'realesrgan_model'
# f'realesrgan.models.{file_name}' 构建模块的完整导入路径，如 'realesrgan.models.realesrgan_model'
# importlib.import_module() 执行实际的导入操作
# _model_modules 列表存储了所有被导入的模块对象。
# 导入模块本身会执行模块内的代码，这对于注册模型类（如果这些类使用了 @MODEL_REGISTRY.register() 装饰器）至关重要。
# 在 basicsr 框架中，模型类通常封装了训练循环、损失函数定义、优化器设置、以及与网络 (archs) 和数据加载器 (data) 的交互逻辑。
_model_modules = [importlib.import_module(f'realesrgan.models.{file_name}') for file_name in model_filenames]
