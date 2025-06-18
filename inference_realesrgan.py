import argparse # 用于解析命令行参数
import cv2 # OpenCV库，用于图像读写
import glob # 用于查找文件路径模式
import os # 操作系统接口，用于路径操作

# 从basicsr库导入RRDBNet网络结构 (Real-ESRGAN的常用骨干网络)
from basicsr.archs.rrdbnet_arch import RRDBNet
# 从basicsr库导入从URL下载文件的工具
from basicsr.utils.download_util import load_file_from_url

# 从本项目导入RealESRGANer推理器和SRVGGNetCompact网络结构
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact


def main():
    """Real-ESRGAN的推理演示脚本。"""
    parser = argparse.ArgumentParser() # 创建参数解析器
    # 定义命令行参数
    parser.add_argument('-i', '--input', type=str, default='inputs', help='输入图像或文件夹路径')
    parser.add_argument(
        '-n',
        '--model_name',
        type=str,
        default='RealESRGAN_x4plus', # 默认模型名称
        help=('可选模型: RealESRGAN_x4plus | RealESRNet_x4plus | RealESRGAN_x4plus_anime_6B | RealESRGAN_x2plus | '
              'realesr-animevideov3 | realesr-general-x4v3'))
    parser.add_argument('-o', '--output', type=str, default='results', help='输出文件夹路径')
    parser.add_argument(
        '-dn',
        '--denoise_strength', # 去噪强度参数
        type=float,
        default=0.5,
        help=('去噪强度。0表示弱去噪(保留噪声)，1表示强去噪能力。'
              '仅用于realesr-general-x4v3模型。'))
    parser.add_argument('-s', '--outscale', type=float, default=4, help='图像最终的上采样倍数')
    parser.add_argument(
        '--model_path', type=str, default=None, help='[可选] 模型路径。通常不需要指定，脚本会自动下载或查找。')
    parser.add_argument('--suffix', type=str, default='out', help='修复后图像的文件名后缀')
    parser.add_argument('-t', '--tile', type=int, default=0, help='瓦片(tile)大小，0表示测试时不使用瓦片处理')
    parser.add_argument('--tile_pad', type=int, default=10, help='瓦片之间的填充大小')
    parser.add_argument('--pre_pad', type=int, default=0, help='每个边界的预填充大小')
    parser.add_argument('--face_enhance', action='store_true', help='使用GFPGAN增强面部')
    parser.add_argument(
        '--fp32', action='store_true', help='推理时使用fp32精度。默认: fp16 (半精度)。')
    parser.add_argument(
        '--alpha_upsampler', # alpha通道的上采样器
        type=str,
        default='realesrgan',
        help='alpha通道的上采样器。可选: realesrgan | bicubic')
    parser.add_argument(
        '--ext', # 输出图像扩展名
        type=str,
        default='auto',
        help='图像扩展名。可选: auto | jpg | png, auto表示使用与输入相同的扩展名')
    parser.add_argument(
        '-g', '--gpu-id', type=int, default=None, help='指定使用的GPU设备ID (默认为None，自动选择)')

    args = parser.parse_args() # 解析命令行参数

    # 根据模型名称确定模型结构和权重下载URL
    args.model_name = args.model_name.split('.')[0] # 去除可能的扩展名
    if args.model_name == 'RealESRGAN_x4plus':  # x4 RRDBNet 模型
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4 # 网络自身的放大倍数
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']
    elif args.model_name == 'RealESRNet_x4plus':  # x4 RRDBNet 模型 (无GAN版本)
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth']
    elif args.model_name == 'RealESRGAN_x4plus_anime_6B':  # x4 RRDBNet 动漫模型 (6个block)
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth']
    elif args.model_name == 'RealESRGAN_x2plus':  # x2 RRDBNet 模型
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        netscale = 2
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth']
    elif args.model_name == 'realesr-animevideov3':  # x4 VGG风格模型 (轻量级, 用于动漫视频)
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth']
    elif args.model_name == 'realesr-general-x4v3':  # x4 VGG风格通用模型
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
        netscale = 4
        # 此模型有两个权重文件，一个是带去噪的(wdn)，一个是不带的。通过DNI进行插值控制去噪强度。
        file_url = [
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth', # 带去噪权重
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth'  # 不带去噪权重
        ]
    else: # 如果模型名称不在预设列表中，则抛出错误
        raise ValueError(f"未知的模型名称: {args.model_name}")


    # 确定模型权重文件的最终路径
    if args.model_path is not None: # 如果用户显式指定了模型路径
        model_path = args.model_path
    else: # 用户未指定路径，则在 'weights' 目录下查找或下载
        model_path = os.path.join('weights', args.model_name + '.pth')
        if not os.path.isfile(model_path): # 如果本地不存在该权重文件
            # ROOT_DIR 在 realesrgan.utils 中定义，指向项目根目录
            # 此处重新获取脚本所在目录的父目录作为下载根目录的参考 (通常是项目根目录下的 'realesrgan' 文件夹)
            # 但更稳妥的做法是从 import realesrgan.utils import ROOT_DIR
            # (假设此脚本在项目根目录下，或者 'weights' 目录在脚本同级)
            current_script_dir = os.path.dirname(os.path.abspath(__file__))
            weights_dir = os.path.join(current_script_dir, 'weights') # 假设weights目录在脚本同级
            os.makedirs(weights_dir, exist_ok=True) # 确保weights目录存在

            for url_item in file_url: # 遍历下载链接列表 (主要为realesr-general-x4v3模型)
                # load_file_from_url 会下载并保存文件，然后返回其本地路径
                # 对于realesr-general-x4v3，它会下载两个模型，model_path会是最后一个下载的模型的路径
                # 这在DNI逻辑中会通过替换名称来找到另一个模型
                model_path = load_file_from_url(
                    url=url_item, model_dir=weights_dir, progress=True, file_name=None)

    # 使用DNI (Deep Network Interpolation) 控制去噪强度 (仅针对realesr-general-x4v3)
    dni_weight = None
    if args.model_name == 'realesr-general-x4v3' and args.denoise_strength != 1:
        # 如果模型是realesr-general-x4v3且用户指定了非默认的denoise_strength
        # wdn_model_path 指向带去噪的权重文件
        # model_path 当前可能指向 realesr-general-x4v3.pth 或 realesr-general-wdn-x4v3.pth
        # 需要确保 model_path 指向不带 wdn 的版本，wdn_model_path 指向带 wdn 的版本
        if 'wdn' in model_path: # 如果当前 model_path 是 wdn 版本
            wdn_model_path = model_path
            model_path = model_path.replace('realesr-general-wdn-x4v3', 'realesr-general-x4v3')
        else: # 当前 model_path 是不带 wdn 的版本
            wdn_model_path = model_path.replace('realesr-general-x4v3', 'realesr-general-wdn-x4v3')

        model_path = [model_path, wdn_model_path] # model_path 变为一个列表，包含不带去噪和带去噪的两个模型路径
        # dni_weight 控制两个模型的插值比例，从而控制去噪效果
        dni_weight = [args.denoise_strength, 1 - args.denoise_strength]

    # 初始化 RealESRGANer 推理器
    upsampler = RealESRGANer(
        scale=netscale, # 网络自身的放大倍数
        model_path=model_path, # 模型路径 (单个或列表)
        dni_weight=dni_weight, # DNI权重 (如果model_path是列表)
        model=model, # 实例化的网络模型
        tile=args.tile, # 瓦片大小
        tile_pad=args.tile_pad, # 瓦片填充
        pre_pad=args.pre_pad, # 图像预填充
        half=not args.fp32, # 是否使用半精度 (fp16)，默认是 (因 not args.fp32)
        gpu_id=args.gpu_id) # GPU ID

    if args.face_enhance:  # 如果启用了面部增强
        from gfpgan import GFPGANer # 动态导入GFPGANer
        # 初始化GFPGANer，用于面部修复和增强
        # bg_upsampler=upsampler 表示GFPGAN会使用当前的RealESRGANer实例来放大背景区域
        face_enhancer = GFPGANer(
            model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth', # GFPGAN模型路径
            upscale=args.outscale, # 最终输出的放大倍数
            arch='clean', # GFPGAN架构
            channel_multiplier=2, # GFPGAN通道倍增因子
            bg_upsampler=upsampler) # 背景上采样器指定为当前的RealESRGANer实例

    os.makedirs(args.output, exist_ok=True) # 创建输出文件夹 (如果不存在)

    # 判断输入是单个文件还是文件夹
    if os.path.isfile(args.input):
        paths = [args.input] # 单个文件
    else:
        paths = sorted(glob.glob(os.path.join(args.input, '*'))) # 文件夹，获取所有文件路径并排序

    # 遍历所有输入图像路径
    for idx, path in enumerate(paths):
        imgname, extension = os.path.splitext(os.path.basename(path)) # 获取文件名和原始扩展名
        print('测试中', idx, imgname)

        img = cv2.imread(path, cv2.IMREAD_UNCHANGED) # 读取图像，cv2.IMREAD_UNCHANGED会保留alpha通道
        if img is None: # 检查图像是否成功读取
            print(f"错误: 无法读取图像 {path}")
            continue # 跳过这个图像

        if len(img.shape) == 3 and img.shape[2] == 4: # 判断是否为RGBA图像
            img_mode = 'RGBA'
        else:
            img_mode = None # 其他情况（RGB, 灰度图）

        try:
            if args.face_enhance: # 如果启用面部增强
                # 使用GFPGANer进行增强，它会先识别人脸，修复人脸，然后将修复后的人脸粘贴回原图背景
                #背景由bg_upsampler(即RealESRGANer)处理
                _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
            else: # 仅使用RealESRGAN进行超分
                output, _ = upsampler.enhance(img, outscale=args.outscale)
        except RuntimeError as error: # 捕获运行时错误，例如CUDA显存不足
            print('错误', error)
            print('如果遇到CUDA显存不足 (CUDA out of memory) 的问题，请尝试使用 --tile 参数并设置一个较小的值。')
        except Exception as error: # 捕获其他可能的错误
            print(f'处理图像 {imgname} 时发生错误: {error}')
        else: # 如果没有错误，则保存结果
            if args.ext == 'auto': # 如果输出扩展名为'auto'
                extension = extension[1:] # 使用原始扩展名 (去除'.')
                if not extension: # 如果原始文件没有扩展名
                    extension = 'png' # 默认为png
            else:
                extension = args.ext # 使用用户指定的扩展名
            if img_mode == 'RGBA':  # RGBA图像应保存为png格式以保留透明度
                extension = 'png'

            if args.suffix == '': # 如果后缀为空
                save_path = os.path.join(args.output, f'{imgname}.{extension}')
            else: # 添加后缀
                save_path = os.path.join(args.output, f'{imgname}_{args.suffix}.{extension}')
            cv2.imwrite(save_path, output) # 保存图像


if __name__ == '__main__':
    main() # 执行主函数
