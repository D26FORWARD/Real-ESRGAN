import importlib  # 导入 importlib 模块，用于动态导入模块
from basicsr.utils import scandir  # 从 basicsr.utils 导入 scandir 函数，用于扫描目录内容
from os import path as osp  # 从 os.path 导入 path，并重命名为 osp，用于路径操作

# 自动扫描并导入架构模块以进行注册
# 扫描 archs 文件夹下所有以 '_arch.py' 结尾的文件

# 获取当前文件（__init__.py）所在的目录的绝对路径，即 archs 文件夹的路径
arch_folder = osp.dirname(osp.abspath(__file__))

# 使用 scandir 遍历 arch_folder 中的所有文件和目录
# v 是遍历到的每个文件/目录的完整路径
# osp.basename(v) 获取文件名（例如 'srvgg_arch.py'）
# osp.splitext(osp.basename(v))[0] 获取文件名去除扩展名后的部分（例如 'srvgg_arch'）
# if v.endswith('_arch.py')确保只处理以 '_arch.py' 结尾的文件
arch_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(arch_folder) if v.endswith('_arch.py')]

# 动态导入所有找到的架构模块
# file_name 是上一步中获取的无扩展名的文件名，如 'srvgg_arch'
# f'realesrgan.archs.{file_name}' 构建模块的完整导入路径，如 'realesrgan.archs.srvgg_arch'
# importlib.import_module() 执行实际的导入操作
# _arch_modules 列表存储了所有被导入的模块对象，尽管在这个 __init__.py 中这些对象后续未被显式使用，
# 但导入模块本身会执行模块内的代码，这对于注册类（如果架构文件中有注册机制）至关重要。
_arch_modules = [importlib.import_module(f'realesrgan.archs.{file_name}') for file_name in arch_filenames]
