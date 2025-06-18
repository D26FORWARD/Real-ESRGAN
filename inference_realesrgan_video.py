import argparse # 用于解析命令行参数
import cv2 # OpenCV库
import glob # 文件路径匹配
import mimetypes # 用于猜测文件类型
import numpy as np
import os # 操作系统接口
import shutil # 高级文件操作，如删除目录树
import subprocess # 用于执行子进程 (例如ffmpeg命令)
import torch # PyTorch
from basicsr.archs.rrdbnet_arch import RRDBNet # RRDBNet网络结构
from basicsr.utils.download_util import load_file_from_url # 从URL下载文件
from os import path as osp # 路径操作别名
from tqdm import tqdm # 进度条库

from realesrgan import RealESRGANer # RealESRGAN推理器
from realesrgan.archs.srvgg_arch import SRVGGNetCompact # SRVGG网络结构

try:
    import ffmpeg # ffmpeg-python绑定库
except ImportError: # 如果导入失败
    import pip
    pip.main(['install', '--user', 'ffmpeg-python']) # 尝试使用pip安装
    import ffmpeg


def get_video_meta_info(video_path):
    """获取视频元信息 (宽度, 高度, FPS, 音频流, 总帧数)。"""
    ret = {}
    try:
        probe = ffmpeg.probe(video_path) # 使用ffmpeg.probe获取视频信息
    except ffmpeg.Error as e:
        print(f'ffmpeg.probe错误: {e.stderr}')
        raise # 重新抛出异常，让调用者处理
    video_streams = [stream for stream in probe['streams'] if stream['codec_type'] == 'video']
    if not video_streams:
        raise ValueError(f"在 {video_path} 中未找到视频流。")
    has_audio = any(stream['codec_type'] == 'audio' for stream in probe['streams'])
    ret['width'] = video_streams[0]['width']
    ret['height'] = video_streams[0]['height']
    ret['fps'] = eval(video_streams[0]['avg_frame_rate']) # 使用eval转换分数形式的FPS
    ret['audio'] = ffmpeg.input(video_path).audio if has_audio else None # 如果有音频，则准备音频流对象
    ret['nb_frames'] = int(video_streams[0]['nb_frames']) # 总帧数
    return ret


def get_sub_video(args, num_process, process_idx):
    """如果num_process > 1, 使用ffmpeg将输入视频分割成子视频片段。"""
    if num_process == 1: # 如果只有一个进程，则直接返回原始输入路径
        return args.input
    meta = get_video_meta_info(args.input) # 获取完整视频的元信息
    duration = int(meta['nb_frames'] / meta['fps']) # 计算总时长 (秒)
    part_time = duration // num_process # 每个子视频片段的时长
    print(f'总时长: {duration}, 子片段时长: {part_time}')
    # 创建临时子视频文件夹
    tmp_video_dir = osp.join(args.output, f'{args.video_name}_inp_tmp_videos')
    os.makedirs(tmp_video_dir, exist_ok=True)
    out_path = osp.join(tmp_video_dir, f'{process_idx:03d}.mp4') # 子视频输出路径

    # 构建ffmpeg命令行
    # -ss: 起始时间; -to: 结束时间 (最后一个片段可能不到part_time)
    # -async 1: 音频同步; -y: 覆盖输出
    cmd = [
        args.ffmpeg_bin, '-i', args.input, '-ss', str(part_time * process_idx),
    ]
    if process_idx != num_process - 1:
        cmd.extend(['-to', str(part_time * (process_idx + 1))])
    cmd.extend(['-async', '1', out_path, '-y'])

    print(' '.join(cmd))
    subprocess.run(cmd, check=True) # 执行ffmpeg命令，check=True会在出错时抛异常
    return out_path


class Reader:
    """读取器类，用于从视频文件或图像文件夹中读取帧。"""
    def __init__(self, args, total_workers=1, worker_idx=0):
        self.args = args
        input_type_guess = mimetypes.guess_type(args.input)[0]
        self.input_type = 'folder' if input_type_guess is None and osp.isdir(args.input) else input_type_guess
        self.paths = []  # 用于图像和文件夹类型
        self.audio = None
        self.input_fps = None
        if self.input_type is not None and self.input_type.startswith('video'):
            # 如果是多进程处理视频，则获取分配给当前worker的子视频路径
            video_path = get_sub_video(args, total_workers, worker_idx)
            # 启动ffmpeg进程，以管道方式输出原始BGR24视频帧
            self.stream_reader = (
                ffmpeg.input(video_path).output('pipe:', format='rawvideo', pix_fmt='bgr24',
                                                loglevel='error').run_async(
                                                    pipe_stdin=True, pipe_stdout=True, cmd=args.ffmpeg_bin))
            meta = get_video_meta_info(video_path) # 获取子视频的元信息
            self.width = meta['width']
            self.height = meta['height']
            self.input_fps = meta['fps']
            self.audio = meta['audio'] # 注意：这里的audio是子视频的，合并时可能需要特殊处理或使用原视频音频
            self.nb_frames = meta['nb_frames']
        else: # 处理图像文件夹或单张图像
            if self.input_type is not None and self.input_type.startswith('image'): # 单张图像
                self.paths = [args.input]
            else: # 图像文件夹
                paths = sorted(glob.glob(os.path.join(args.input, '*')))
                # 多进程时，均分图像文件列表给各个worker
                tot_frames = len(paths)
                num_frame_per_worker = tot_frames // total_workers + (1 if tot_frames % total_workers else 0)
                self.paths = paths[num_frame_per_worker * worker_idx : num_frame_per_worker * (worker_idx + 1)]

            self.nb_frames = len(self.paths)
            assert self.nb_frames > 0, '文件夹为空或无法识别图像'
            from PIL import Image # 用于获取图像尺寸
            try:
                tmp_img = Image.open(self.paths[0])
                self.width, self.height = tmp_img.size
            except Exception as e:
                print(f"错误：无法打开图像 {self.paths[0]} 来获取尺寸: {e}")
                # 可以设置默认值或抛出更具体的异常
                self.width, self.height = 0,0 # 或者更合适的默认值

        self.idx = 0 # 用于图像列表的索引

    def get_resolution(self):
        """返回视频/图像帧的高度和宽度。"""
        return self.height, self.width

    def get_fps(self):
        """获取输出视频的FPS。优先使用用户指定的FPS，其次是输入视频的FPS，默认为24。"""
        if self.args.fps is not None:
            return self.args.fps
        elif self.input_fps is not None:
            return self.input_fps
        return 24

    def get_audio(self):
        """返回音频流对象 (如果存在)。"""
        return self.audio

    def __len__(self):
        """返回总帧数。"""
        return self.nb_frames

    def get_frame_from_stream(self):
        """从ffmpeg管道读取一帧。"""
        img_bytes = self.stream_reader.stdout.read(self.width * self.height * 3)  # BGR24格式，每像素3字节
        if not img_bytes: # 如果没有字节流，表示视频结束
            return None
        img = np.frombuffer(img_bytes, np.uint8).reshape([self.height, self.width, 3]) # 转换为NumPy数组
        return img

    def get_frame_from_list(self):
        """从图像文件列表读取一帧。"""
        if self.idx >= self.nb_frames: # 如果已读完所有图像
            return None
        img = cv2.imread(self.paths[self.idx]) # 使用cv2读取图像
        self.idx += 1
        return img

    def get_frame(self):
        """获取下一帧 (根据输入类型调用相应方法)。"""
        if self.input_type is not None and self.input_type.startswith('video'):
            return self.get_frame_from_stream()
        else:
            return self.get_frame_from_list()

    def close(self):
        """关闭ffmpeg读取流 (如果已打开)。"""
        if self.input_type is not None and self.input_type.startswith('video'):
            self.stream_reader.stdin.close() # 关闭ffmpeg子进程的stdin
            self.stream_reader.wait() # 等待子进程结束


class Writer:
    """写入器类，用于将处理后的帧写入输出视频文件。"""
    def __init__(self, args, audio, height, width, video_save_path, fps):
        out_width, out_height = int(width * args.outscale), int(height * args.outscale) # 计算输出分辨率
        if out_height > 2160: # 对超大分辨率输出进行提示
            print('您正在生成大于4K的视频，由于IO速度限制，处理会非常缓慢。强烈建议减小输出倍数(-s参数)。')

        # 启动ffmpeg进程，以管道方式接收原始BGR24视频帧并编码保存
        if audio is not None: # 如果存在音频流
            self.stream_writer = (
                ffmpeg.input('pipe:', format='rawvideo', pix_fmt='bgr24', s=f'{out_width}x{out_height}',
                             framerate=fps) # 从管道输入视频帧
                .output(
                    audio, # 添加音频流 (从Reader获取)
                    video_save_path, # 输出文件路径
                    pix_fmt='yuv420p', # 常用的像素格式，兼容性好
                    vcodec='libx264', # 使用H.264视频编码器
                    loglevel='error',
                    acodec='copy') # 直接复制音频流，不重新编码
                .overwrite_output() # 覆盖已存在文件
                .run_async(pipe_stdin=True, pipe_stdout=True, cmd=args.ffmpeg_bin))
        else: # 无音频流
            self.stream_writer = (
                ffmpeg.input('pipe:', format='rawvideo', pix_fmt='bgr24', s=f'{out_width}x{out_height}',
                             framerate=fps)
                .output(video_save_path, pix_fmt='yuv420p', vcodec='libx264', loglevel='error')
                .overwrite_output()
                .run_async(pipe_stdin=True, pipe_stdout=True, cmd=args.ffmpeg_bin))

    def write_frame(self, frame):
        """将一帧图像数据写入ffmpeg管道。"""
        frame = frame.astype(np.uint8).tobytes() # 转换为uint8并转为字节流
        self.stream_writer.stdin.write(frame) # 写入ffmpeg子进程的stdin

    def close(self):
        """关闭ffmpeg写入流。"""
        self.stream_writer.stdin.close() # 关闭stdin，通知ffmpeg没有更多帧了
        self.stream_writer.wait() # 等待ffmpeg子进程完成编码和写入


def inference_video(args, video_save_path, device=None, total_workers=1, worker_idx=0):
    """核心视频推理函数，会被每个worker调用(如果是多进程)。"""
    # ---------------------- 根据模型名称确定模型和权重 ---------------------- #
    # (这部分逻辑与 inference_realesrgan.py 中的模型选择逻辑几乎完全相同)
    args.model_name = args.model_name.split('.pth')[0]
    if args.model_name == 'RealESRGAN_x4plus':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']
    # ... (其他模型的elif分支，与inference_realesrgan.py中类似) ...
    elif args.model_name == 'RealESRNet_x4plus':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth']
    elif args.model_name == 'RealESRGAN_x4plus_anime_6B':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth']
    elif args.model_name == 'RealESRGAN_x2plus':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        netscale = 2
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth']
    elif args.model_name == 'realesr-animevideov3':
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth']
    elif args.model_name == 'realesr-general-x4v3':
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
        netscale = 4
        file_url = [
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth',
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth'
        ]
    else:
        raise ValueError(f"未知的模型名称: {args.model_name}")


    # ---------------------- 确定模型路径 (本地查找或下载) ---------------------- #
    # (这部分逻辑与 inference_realesrgan.py 中的模型路径确定逻辑几乎完全相同)
    model_path = os.path.join('weights', args.model_name + '.pth') # 默认在 'weights' 目录
    if not os.path.isfile(model_path):
        # 获取脚本所在目录，并假设 'weights' 文件夹在此目录下或项目根目录下
        # ROOT_DIR = osp.dirname(osp.abspath(__file__)) # 这是指当前文件(utils.py)的目录
        # inference_realesrgan_video.py 的 weights 目录通常在其同级或项目的根目录
        # 为了简单，假设 weights 文件夹在当前工作目录的子目录中，或者用户已将其放在正确位置
        # 更健壮的方式是使用固定的相对路径或绝对路径
        weights_dir = 'weights'
        os.makedirs(weights_dir, exist_ok=True)
        for url_item in file_url:
            model_path = load_file_from_url(
                url=url_item, model_dir=weights_dir, progress=True, file_name=None)

    # ---------------------- DNI 设置 (去噪强度控制) ---------------------- #
    # (这部分逻辑与 inference_realesrgan.py 中的DNI设置逻辑几乎完全相同)
    dni_weight = None
    if args.model_name == 'realesr-general-x4v3' and args.denoise_strength != 1:
        # 确保model_path指向正确的非wdn版本，wdn_model_path指向wdn版本
        if 'wdn' in model_path: # 如果当前 model_path 是 wdn 版本
            wdn_model_path = model_path
            model_path = model_path.replace('realesr-general-wdn-x4v3', 'realesr-general-x4v3')
        else: # 当前 model_path 是不带 wdn 的版本
            wdn_model_path = model_path.replace('realesr-general-x4v3', 'realesr-general-wdn-x4v3')
        # 检查wdn模型是否存在，如果不存在则尝试下载（如果file_url包含它）
        if not osp.isfile(wdn_model_path) and len(file_url)>1:
             wdn_url = next((url for url in file_url if 'wdn' in url), None)
             if wdn_url:
                load_file_from_url(url=wdn_url, model_dir=osp.dirname(model_path), progress=True, file_name=None)

        model_path = [model_path, wdn_model_path]
        dni_weight = [args.denoise_strength, 1 - args.denoise_strength]

    # ---------------------- 初始化RealESRGANer推理器 ---------------------- #
    upsampler = RealESRGANer(
        scale=netscale, model_path=model_path, dni_weight=dni_weight, model=model,
        tile=args.tile, tile_pad=args.tile_pad, pre_pad=args.pre_pad,
        half=not args.fp32, device=device) # 注意这里传入了device参数

    # ---------------------- 面部增强设置 (可选) ---------------------- #
    if 'anime' in args.model_name and args.face_enhance: # 动漫模型通常不适用GFPGAN面部增强
        print('面部增强不支持动漫模型，此选项已为您关闭。如果坚持开启，请手动修改代码。')
        args.face_enhance = False

    if args.face_enhance:
        from gfpgan import GFPGANer # 动态导入GFPGAN
        face_enhancer = GFPGANer(
            model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
            upscale=args.outscale, arch='clean', channel_multiplier=2,
            bg_upsampler=upsampler) # GFPGAN使用RealESRGANer作为背景放大器
    else:
        face_enhancer = None

    # ---------------------- 初始化Reader和Writer ---------------------- #
    reader = Reader(args, total_workers, worker_idx) # 初始化帧读取器
    audio = reader.get_audio() # 获取音频流 (如果是主进程且有音频)
    height, width = reader.get_resolution() # 获取视频帧原始分辨率
    fps = reader.get_fps() # 获取目标FPS
    writer = Writer(args, audio if worker_idx == 0 else None, height, width, video_save_path, fps) # 初始化帧写入器, 只有主worker写入音频

    # ---------------------- 逐帧处理循环 ---------------------- #
    pbar = tqdm(total=len(reader), unit='frame', desc=f'推理 worker {worker_idx}') # 初始化进度条
    while True:
        img = reader.get_frame() # 读取一帧
        if img is None: # 如果没有帧了 (视频结束或图像列表读完)
            break

        try:
            if args.face_enhance and face_enhancer: # 如果启用面部增强
                _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
            else: # 仅使用RealESRGAN
                output, _ = upsampler.enhance(img, outscale=args.outscale)
        except RuntimeError as error: # 捕获运行时错误 (如OOM)
            print('错误:', error)
            print('如果遇到CUDA显存不足，请尝试使用 --tile 参数并设置一个较小的值。')
            # 可选：跳过此帧或采取其他措施
            continue # 跳过出错的帧
        else:
            writer.write_frame(output) # 将处理后的帧写入输出视频

        if device: # 如果在GPU上运行，同步一下可能有助于显存管理或精确计时
             torch.cuda.synchronize(device)
        pbar.update(1) # 更新进度条

    # ---------------------- 清理和关闭 ---------------------- #
    reader.close()
    writer.close()
    pbar.close()


def run(args):
    """多进程运行的主协调函数。"""
    args.video_name = osp.splitext(os.path.basename(args.input))[0] # 获取视频名 (不含扩展名)
    video_save_path = osp.join(args.output, f'{args.video_name}_{args.suffix}.mp4') # 最终输出视频路径

    if args.extract_frame_first: # 如果设置了先提取所有帧为图像文件
        print('正在将视频提取为帧...')
        tmp_frames_folder = osp.join(args.output, f'{args.video_name}_inp_tmp_frames')
        os.makedirs(tmp_frames_folder, exist_ok=True)
        # 使用ffmpeg命令行提取帧
        cmd = [args.ffmpeg_bin, '-i', args.input, '-qscale:v', '1', '-qmin', '1', '-qmax', '1', '-vsync', '0', f'{tmp_frames_folder}/frame%08d.png']
        subprocess.run(cmd, check=True)
        args.input = tmp_frames_folder # 将输入路径改为帧文件夹路径
        print('帧提取完成。')


    num_gpus = torch.cuda.device_count() # 获取可用GPU数量
    num_process = num_gpus * args.num_process_per_gpu # 计算总进程数

    if num_process == 1: # 如果只有一个进程，则直接调用inference_video
        inference_video(args, video_save_path, device=torch.device(0) if num_gpus > 0 else torch.device('cpu'))
        return

    # 多进程处理
    ctx = torch.multiprocessing.get_context('spawn') # 使用'spawn'上下文以避免CUDA相关问题
    pool = ctx.Pool(num_process) # 创建进程池
    # 创建临时输出子视频文件夹
    tmp_out_video_dir = osp.join(args.output, f'{args.video_name}_out_tmp_videos')
    os.makedirs(tmp_out_video_dir, exist_ok=True)

    pbar = tqdm(total=num_process, unit='sub_video', desc='多进程推理')
    for i in range(num_process): # 为每个进程分配任务
        sub_video_save_path = osp.join(tmp_out_video_dir, f'{i:03d}.mp4') # 子视频保存路径
        # 异步执行inference_video
        pool.apply_async(
            inference_video,
            args=(args, sub_video_save_path, torch.device(i % num_gpus), num_process, i), # 为每个worker分配GPU
            callback=lambda arg: pbar.update(1)) # 每完成一个子任务，更新进度条
    pool.close()
    pool.join() # 等待所有子进程完成
    pbar.close()

    # ---------------------- 合并子视频 ---------------------- #
    print('正在合并所有处理后的子视频...')
    # 准备一个包含所有子视频路径的txt文件 (ffmpeg concat demuxer需要)
    vidlist_path = osp.join(args.output, f'{args.video_name}_vidlist.txt')
    with open(vidlist_path, 'w') as f:
        for i in range(num_process):
            f.write(f'file \'{args.video_name}_out_tmp_videos/{i:03d}.mp4\'\n') # 注意路径格式

    # 使用ffmpeg concat合并视频
    cmd = [
        args.ffmpeg_bin, '-f', 'concat', '-safe', '0', '-i', vidlist_path, '-c',
        'copy', video_save_path # '-c copy' 表示直接复制流，不重新编码，速度快
    ]
    print(' '.join(cmd))
    subprocess.run(cmd, check=True)

    # ---------------------- 清理临时文件和文件夹 ---------------------- #
    print('清理临时文件...')
    shutil.rmtree(tmp_out_video_dir) # 删除输出子视频的临时文件夹
    # 如果输入视频被分割过，也删除输入的临时文件夹
    inp_tmp_video_dir = osp.join(args.output, f'{args.video_name}_inp_tmp_videos')
    if osp.exists(inp_tmp_video_dir):
        shutil.rmtree(inp_tmp_video_dir)
    os.remove(vidlist_path) # 删除vidlist.txt
    print('视频处理完成。')


def main(): # 主入口函数
    """Real-ESRGAN的推理演示。主要用于修复动漫视频。"""
    # (参数定义与 inference_realesrgan.py 中的类似，但有视频特有参数)
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='inputs', help='输入视频、图像或文件夹')
    parser.add_argument(
        '-n', '--model_name', type=str, default='realesr-animevideov3', # 视频版默认模型
        help=('模型名称... Default:realesr-animevideov3'))
    parser.add_argument('-o', '--output', type=str, default='results', help='输出文件夹')
    parser.add_argument('-dn', '--denoise_strength', type=float, default=0.5, help=('去噪强度...'))
    parser.add_argument('-s', '--outscale', type=float, default=4, help='最终上采样倍数')
    parser.add_argument('--suffix', type=str, default='out', help='修复后视频的后缀')
    parser.add_argument('-t', '--tile', type=int, default=0, help='瓦片大小...')
    parser.add_argument('--tile_pad', type=int, default=10, help='瓦片填充')
    parser.add_argument('--pre_pad', type=int, default=0, help='预填充')
    parser.add_argument('--face_enhance', action='store_true', help='使用GFPGAN增强面部')
    parser.add_argument('--fp32', action='store_true', help='使用fp32精度...')
    parser.add_argument('--fps', type=float, default=None, help='输出视频的FPS')
    parser.add_argument('--ffmpeg_bin', type=str, default='ffmpeg', help='ffmpeg程序路径')
    parser.add_argument('--extract_frame_first', action='store_true', help='是否先将视频提取为帧图像再处理')
    parser.add_argument('--num_process_per_gpu', type=int, default=1, help='每张GPU上运行的进程数')
    parser.add_argument('--alpha_upsampler', type=str, default='realesrgan', help='alpha通道上采样器...')
    parser.add_argument('--ext', type=str, default='auto', help='图像扩展名...(主要用于extract_frame_first模式)')
    args = parser.parse_args()

    args.input = args.input.rstrip('/').rstrip('\\') # 清理输入路径末尾的斜杠
    os.makedirs(args.output, exist_ok=True) # 创建输出目录

    # 判断输入是否为视频文件
    input_type_guess = mimetypes.guess_type(args.input)[0]
    if input_type_guess is not None and input_type_guess.startswith('video'):
        is_video = True
    # 如果mimetype未识别，但路径又不是一个已存在的目录，也可能是一个视频文件（或者其他单个文件）
    # 但这里主要依赖mimetype，如果不是目录，run()内部的Reader会尝试按单文件图像处理
    elif not osp.isdir(args.input):
        # 简单检查：如果不是目录，且没有明确的图像mimetype，也可能是一个视频。
        # 更鲁棒的检查可能需要ffprobe。但当前脚本结构下，如果不是明确的video mimetype，
        # 且extract_frame_first=False，它会尝试作为图像文件夹或单图像处理。
        # 如果extract_frame_first=True，则ffmpeg会尝试打开它。
        is_video = not osp.isdir(args.input) # 粗略判断，如果不是目录，则可能是视频或单图
    else:
        is_video = False


    if is_video and args.input.endswith('.flv'): # 特殊处理flv格式，先转为mp4
        mp4_path = args.input.replace('.flv', '.mp4')
        os.system(f'{args.ffmpeg_bin} -i {args.input} -codec copy {mp4_path}') # 使用系统ffmpeg命令转换
        args.input = mp4_path # 更新输入路径为转换后的mp4

    # 如果设置了先提取帧但输入不是视频，则禁用提取帧选项
    if args.extract_frame_first and not is_video:
        print('输入不是视频文件，已禁用 --extract_frame_first 选项。')
        args.extract_frame_first = False

    run(args) # 调用run函数开始处理

    # 如果之前提取了帧，则清理临时帧文件夹
    if args.extract_frame_first and is_video: # 确保是视频输入才执行清理
        tmp_frames_folder = osp.join(args.output, f'{args.video_name}_inp_tmp_frames')
        if osp.exists(tmp_frames_folder): # 再次检查以防万一
             shutil.rmtree(tmp_frames_folder)


if __name__ == '__main__':
    main() # 脚本入口
