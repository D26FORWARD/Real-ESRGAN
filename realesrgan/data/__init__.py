import importlib  # 导入 importlib 模块，用于动态导入模块
from basicsr.utils import scandir  # 从 basicsr.utils 导入 scandir 函数，用于扫描目录内容
from os import path as osp  # 从 os.path 导入 path，并重命名为 osp，用于路径操作

# 自动扫描并导入数据集模块以进行注册
# 扫描 data 文件夹下所有以 '_dataset.py' 结尾的文件

# 获取当前文件（__init__.py）所在的目录的绝对路径，即 data 文件夹的路径
data_folder = osp.dirname(osp.abspath(__file__))

# 使用 scandir 遍历 data_folder 中的所有文件和目录
# v 是遍历到的每个文件/目录的完整路径
# osp.basename(v) 获取文件名（例如 'realesrgan_dataset.py'）
# osp.splitext(osp.basename(v))[0] 获取文件名去除扩展名后的部分（例如 'realesrgan_dataset'）
# if v.endswith('_dataset.py') 确保只处理以 '_dataset.py' 结尾的文件
dataset_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(data_folder) if v.endswith('_dataset.py')]

# 动态导入所有找到的数据集模块
# file_name 是上一步中获取的无扩展名的文件名，如 'realesrgan_dataset'
# f'realesrgan.data.{file_name}' 构建模块的完整导入路径，如 'realesrgan.data.realesrgan_dataset'
# importlib.import_module() 执行实际的导入操作
# _dataset_modules 列表存储了所有被导入的模块对象。
# 导入模块本身会执行模块内的代码，这对于注册 PyTorch Dataset 类（如果这些类使用了 @DATASET_REGISTRY.register() 装饰器）至关重要。
_dataset_modules = [importlib.import_module(f'realesrgan.data.{file_name}') for file_name in dataset_filenames]
