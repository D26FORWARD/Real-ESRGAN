import os # 操作系统接口，用于路径拼接等
from basicsr.data.data_util import paired_paths_from_folder, paired_paths_from_lmdb # 从basicsr导入用于成对图像路径处理的工具函数
from basicsr.data.transforms import augment, paired_random_crop # 从basicsr导入数据增强函数：通用增强、成对随机裁剪
from basicsr.utils import FileClient, imfrombytes, img2tensor # 从basicsr导入工具：文件客户端、从字节解码图像、图像转张量
from basicsr.utils.registry import DATASET_REGISTRY # 从basicsr导入数据集注册表
from torch.utils import data as data # PyTorch的数据加载工具
from torchvision.transforms.functional import normalize # 从torchvision导入归一化函数


@DATASET_REGISTRY.register() # 装饰器，将此类注册到DATASET_REGISTRY
class RealESRGANPairedDataset(data.Dataset):
    """用于图像恢复的成对图像数据集。

    读取LQ (Low Quality, 例如LR 低分辨率, 模糊, 噪声等) 和 GT (Ground-Truth 高质量) 图像对。

    共有三种模式:
    1. 'lmdb': 使用lmdb文件。
        如果 opt['io_backend'] == 'lmdb'。
    2. 'meta_info': 使用元信息文件生成路径。
        如果 opt['io_backend'] != 'lmdb' 且 opt['meta_info'] 不为 None。
    3. 'folder': 扫描文件夹生成路径。
        其余情况。

    参数:
        opt (dict): 训练数据集的配置字典。包含以下键:
            dataroot_gt (str): GT图像的数据根路径。
            dataroot_lq (str): LQ图像的数据根路径。
            meta_info (str): 元信息文件的路径。
            io_backend (dict): IO后端类型及其他参数。
            filename_tmpl (str): 每个文件名的模板。注意模板不包括文件扩展名。默认: '{}'。
            gt_size (int): GT图像块的裁剪尺寸。
            use_hflip (bool): 是否使用水平翻转。
            use_rot (bool): 是否使用旋转 (通过垂直翻转和行列转置实现)。
            scale (int): 缩放因子，将自动添加。 (通常由框架或模型设置)
            phase (str): 'train' 或 'val'，指示数据集的用途。
    """

    def __init__(self, opt):
        super(RealESRGANPairedDataset, self).__init__() # 调用父类构造函数
        self.opt = opt # 保存配置选项
        self.file_client = None # 文件客户端，延迟初始化
        self.io_backend_opt = opt['io_backend'] # IO后端配置
        # 用于归一化输入图像的均值和标准差
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq'] # GT和LQ图像文件夹路径
        # 文件名模板，用于在'folder'模式下匹配LQ和GT图像
        self.filename_tmpl = opt.get('filename_tmpl', '{}') # 使用get方法提供默认值

        # 根据IO后端类型确定加载路径的方式
        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder] # LMDB数据库路径列表
            self.io_backend_opt['client_keys'] = ['lq', 'gt'] # 客户端键名
            # 从LMDB获取成对的路径信息
            self.paths = paired_paths_from_lmdb([self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info' in self.opt and self.opt['meta_info'] is not None:
            # 使用元信息文件 (通常是txt文件，每行是 "gt_path_relative, lq_path_relative")
            with open(self.opt['meta_info']) as fin:
                paths_info = [line.strip() for line in fin]
            self.paths = []
            for path_info in paths_info:
                # 假设路径以逗号和空格分隔，例如 "subfolder/gt_img.png, subfolder/lq_img.png"
                gt_path_rel, lq_path_rel = path_info.split(', ')
                # 拼接成绝对路径
                self.paths.append({
                    'gt_path': os.path.join(self.gt_folder, gt_path_rel),
                    'lq_path': os.path.join(self.lq_folder, lq_path_rel)
                })
        else:
            # 默认使用文件夹扫描模式
            # 注意：如果文件夹内文件过多，此操作可能非常耗时，建议使用meta_info文件
            self.paths = paired_paths_from_folder([self.lq_folder, self.gt_folder], ['lq', 'gt'], self.filename_tmpl)

    def __getitem__(self, index):
        # 惰性初始化文件客户端
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale'] # 获取缩放因子

        # 加载GT和LQ图像。维度顺序: HWC (高宽通道); 通道顺序: BGR;
        # 图像范围: [0, 1], float32.
        gt_path = self.paths[index]['gt_path'] # 获取GT图像路径
        img_bytes = self.file_client.get(gt_path, 'gt') # 读取GT图像字节流
        img_gt = imfrombytes(img_bytes, float32=True) # 解码GT图像

        lq_path = self.paths[index]['lq_path'] # 获取LQ图像路径
        img_bytes = self.file_client.get(lq_path, 'lq') # 读取LQ图像字节流
        img_lq = imfrombytes(img_bytes, float32=True) # 解码LQ图像

        # 训练阶段的数据增强
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size'] # GT图像裁剪尺寸
            # 成对随机裁剪：从GT图像中随机裁剪一块，并从LQ图像中裁剪出对应位置、按scale缩放的区域
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
            # 翻转、旋转：对图像对应用相同的随机翻转和旋转操作，以保持对应关系
            img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_hflip'], self.opt['use_rot'])

        # BGR转RGB, HWC转CHW, numpy数组转tensor
        # img2tensor函数会处理这些转换
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)

        # 图像归一化 (如果提供了均值和标准差)
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True) # 对LQ图像进行归一化
            normalize(img_gt, self.mean, self.std, inplace=True) # 对GT图像进行归一化

        # 返回包含LQ图像、GT图像及其路径的字典
        return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path}

    def __len__(self):
        # 返回数据集中图像对的总数
        return len(self.paths)
