#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2025 - 2025 Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam
# @Email  : sepinetam@gmail.com
# @File   : mumu.py

"""
视频自动字幕生成与翻译工具
- 自动处理文件夹中的所有视频
- 生成英文字幕并翻译为中文
- 内存优化设计
- 兼容M2芯片与最新版argostranslate
"""

import os
import sys
import time
import glob
import logging
import subprocess
import gc
import tempfile
from pathlib import Path
import argostranslate.package
import argostranslate.translate
from dotenv import load_dotenv

# 尝试导入所需的可选依赖
try:
    import humanize

    _HAS_HUMANIZE = True
except ImportError:
    _HAS_HUMANIZE = False

# 明确导入openai的whisper模块
try:
    import whisper

    # 测试是否有load_model方法
    if hasattr(whisper, 'load_model'):
        _WHISPER_SOURCE = "standard"
    else:
        raise ImportError("导入的whisper包没有load_model方法")
except ImportError as e:
    logger = logging.getLogger("subtitle_processor")
    if logger.handlers:
        logger.error(f"导入whisper失败: {e}")
        logger.error("请确保安装了正确的包: pip install openai-whisper")


    # 如果导入失败，定义一个假的whisper模块以便在运行时提供更好的错误消息
    class WhisperStub:
        def __init__(self):
            pass

        def load_model(self, *args, **kwargs):
            raise ImportError("无法导入whisper模块。请确保安装了正确的包: pip install openai-whisper")


    whisper = WhisperStub()
    _WHISPER_SOURCE = "stub"
from pathlib import Path
import gc
import shutil
import tempfile

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("subtitle_process.log")
    ]
)
logger = logging.getLogger("subtitle_processor")


def load_config():
    """从.env文件加载配置"""
    load_dotenv()

    config = {
        'INPUT_FOLDER': os.getenv('INPUT_FOLDER', './videos'),
        'OUTPUT_FOLDER': os.getenv('OUTPUT_FOLDER', './output'),
        'SUBTITLE_FOLDER': os.getenv('SUBTITLE_FOLDER', './subtitles'),
        'TEMP_FOLDER': os.getenv('TEMP_FOLDER', './temp'),
        'WHISPER_MODEL': os.getenv('WHISPER_MODEL', 'base'),
        'VIDEO_EXTENSIONS': os.getenv('VIDEO_EXTENSIONS', '.mp4,.mkv,.avi,.mov').split(','),
        'GENERATE_ENGLISH_SRT': os.getenv('GENERATE_ENGLISH_SRT', 'true').lower() == 'true',
        'GENERATE_CHINESE_SRT': os.getenv('GENERATE_CHINESE_SRT', 'true').lower() == 'true',
        'EMBED_SUBTITLES': os.getenv('EMBED_SUBTITLES', 'true').lower() == 'true',
        'CLEAN_TEMP_FILES': os.getenv('CLEAN_TEMP_FILES', 'true').lower() == 'true',
        'BATCH_SIZE': int(os.getenv('BATCH_SIZE', '5')),  # 每批处理的字幕数量，避免占用太多内存
        'COPY_SUBTITLES_TO_SEPARATE_FOLDER': os.getenv('COPY_SUBTITLES_TO_SEPARATE_FOLDER', 'true').lower() == 'true',
        'ORGANIZE_SUBTITLES_BY_LANGUAGE': os.getenv('ORGANIZE_SUBTITLES_BY_LANGUAGE', 'true').lower() == 'true',
    }

    return config


def ensure_directories(config):
    """确保所需目录存在"""
    for folder in [config['INPUT_FOLDER'], config['OUTPUT_FOLDER'], config['TEMP_FOLDER'], config['SUBTITLE_FOLDER']]:
        os.makedirs(folder, exist_ok=True)
        logger.info(f"确保目录存在: {folder}")

    # 如果需要按语言组织字幕，创建语言子文件夹
    if config['ORGANIZE_SUBTITLES_BY_LANGUAGE']:
        os.makedirs(os.path.join(config['SUBTITLE_FOLDER'], 'en'), exist_ok=True)
        os.makedirs(os.path.join(config['SUBTITLE_FOLDER'], 'zh'), exist_ok=True)
        logger.info(
            f"创建语言子文件夹: {os.path.join(config['SUBTITLE_FOLDER'], 'en')} 和 {os.path.join(config['SUBTITLE_FOLDER'], 'zh')}")


def find_videos(config):
    """查找所有需要处理的视频文件"""
    videos = []
    for ext in config['VIDEO_EXTENSIONS']:
        pattern = os.path.join(config['INPUT_FOLDER'], f"*{ext}")
        videos.extend(glob.glob(pattern))

    logger.info(f"找到 {len(videos)} 个视频文件需要处理")
    return videos


def prepare_translation_model():
    """准备翻译模型"""
    logger.info("准备翻译模型...")
    try:
        # 检查是否已经安装了英文到中文的翻译包
        from_code, to_code = "en", "zh"
        try:
            # 尝试获取翻译器，如果成功则说明已安装
            translator = argostranslate.translate.get_translation_from_codes(from_code, to_code)
            logger.info("翻译模型已安装")
        except Exception:
            # 如果无法获取翻译器，则安装翻译包
            logger.info("未找到英文到中文翻译包，开始安装...")
            argostranslate.package.update_package_index()
            available_packages = argostranslate.package.get_available_packages()
            package_to_install = next(
                (p for p in available_packages if p.from_code == from_code and p.to_code == to_code), None
            )
            if package_to_install:
                # 新版本的argostranslate不再使用is_installed方法
                argostranslate.package.install_from_path(package_to_install.download())
                logger.info(f"已安装{from_code}到{to_code}翻译包")
            else:
                logger.error(f"未找到{from_code}到{to_code}的翻译包")
                raise ValueError(f"无法找到{from_code}到{to_code}的翻译包")
        logger.info("翻译模型准备完成")
    except Exception as e:
        logger.error(f"准备翻译模型失败: {str(e)}")
        logger.info("将尝试使用默认英文到中文翻译")


def format_timestamp(seconds):
    """格式化时间戳为SRT格式"""
    hours = int(seconds / 3600)
    minutes = int((seconds % 3600) / 60)
    secs = seconds % 60
    millisecs = int((secs - int(secs)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{int(secs):02d},{millisecs:03d}"


def generate_english_subtitles(video_path, output_srt, whisper_model, temp_dir):
    """使用Whisper生成英文字幕"""
    logger.info(f"开始生成英文字幕: {os.path.basename(video_path)}")
    start_time = time.time()

    try:
        # 检查我们导入的whisper模块类型
        logger.info(f"正在使用的Whisper模块源: {_WHISPER_SOURCE}")
        if _WHISPER_SOURCE == "stub":
            logger.error("未正确安装Whisper模块，请先运行: pip install openai-whisper")
            return False

        # 如果音频文件已存在，则使用已提取的音频
        audio_path = os.path.join(temp_dir, f"{os.path.splitext(os.path.basename(video_path))[0]}.wav")

        if not os.path.exists(audio_path):
            # 提取音频 - 16kHz, 单声道
            logger.info(f"提取音频: {os.path.basename(video_path)}")
            try:
                extract_cmd = [
                    "ffmpeg", "-i", video_path,
                    "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
                    "-hide_banner", "-loglevel", "error",
                    audio_path
                ]
                subprocess.run(extract_cmd, check=True)
            except subprocess.CalledProcessError as e:
                logger.error(f"FFmpeg提取音频失败: {e}")
                # 尝试使用备用命令，兼容性更好
                try:
                    extract_cmd = [
                        "ffmpeg", "-i", video_path,
                        "-vn", "-ar", "16000", "-ac", "1",
                        "-f", "wav", audio_path
                    ]
                    subprocess.run(extract_cmd, check=True)
                except subprocess.CalledProcessError as e2:
                    logger.error(f"备用提取音频命令也失败: {e2}")
                    return False

        # 检查音频文件是否成功创建
        if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
            logger.error(f"音频提取失败或文件为空: {audio_path}")
            return False

        # 检查whisper模块是否有load_model方法
        if not hasattr(whisper, 'load_model'):
            logger.error("当前安装的whisper模块没有load_model方法，可能不是正确的OpenAI Whisper")
            logger.info("尝试安装正确的whisper包: pip install openai-whisper")
            return False

        # 为M2芯片优化模型加载
        logger.info(f"加载Whisper {whisper_model}模型...")
        try:
            import torch

            # 创建模型加载选项
            model_args = {}

            # 检测是否有MPS可用（M系列芯片）
            try:
                if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    logger.info("检测到Apple Silicon，使用MPS后端...")
                    device = torch.device("mps")
                    # 设置环境变量以优化MPS性能
                    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
                    model_args['device'] = device
                elif torch.cuda.is_available():
                    logger.info("使用CUDA后端...")
                    device = torch.device("cuda")
                    model_args['device'] = device
                else:
                    logger.info("使用CPU后端...")
                    device = torch.device("cpu")
                    model_args['device'] = device
            except Exception as e:
                logger.warning(f"设备检测失败: {e}，将使用默认设备")
                # 不指定device，让whisper自己选择

            # 加载模型 - 尝试不同的参数组合
            try:
                # 方法1: 使用设备参数
                if model_args:
                    model = whisper.load_model(whisper_model, **model_args)
                else:
                    # 方法2: 不使用设备参数
                    model = whisper.load_model(whisper_model)
            except Exception as load_error:
                logger.warning(f"加载模型失败: {load_error}，尝试不同参数...")
                # 方法3: 仅使用模型名称
                model = whisper.load_model(whisper_model)

            # 转录音频，使用较小的计算单元以降低内存需求
            logger.info("开始转录...")

            # 准备转录参数
            transcribe_args = {
                'language': 'en',
                'verbose': False,
            }

            # 根据Whisper版本添加适当的参数
            if _WHISPER_SOURCE == "standard":
                # 标准whisper支持这些参数
                transcribe_args['condition_on_previous_text'] = True
                transcribe_args['fp16'] = False  # M2上fp16可能有问题

            result = model.transcribe(audio_path, **transcribe_args)

            # 写入SRT格式
            logger.info("保存英文字幕...")
            with open(output_srt, "w", encoding="utf-8") as srt_file:
                for i, segment in enumerate(result["segments"], start=1):
                    start_time_fmt = format_timestamp(segment["start"])
                    end_time_fmt = format_timestamp(segment["end"])
                    text = segment["text"].strip()
                    srt_file.write(f"{i}\n{start_time_fmt} --> {end_time_fmt}\n{text}\n\n")

            # 释放模型内存
            del model
            if 'cuda' in str(device) if 'device' in locals() else False:
                torch.cuda.empty_cache()
            gc.collect()

            elapsed = time.time() - start_time
            logger.info(f"英文字幕生成完成，耗时: {elapsed:.2f}秒")
            return True

        except Exception as e:
            logger.error(f"Whisper模型处理失败: {str(e)}")
            logger.info("尝试使用最简单的方式加载Whisper...")
            try:
                # 最基本的方式尝试
                model = whisper.load_model("base")
                result = model.transcribe(audio_path)

                with open(output_srt, "w", encoding="utf-8") as srt_file:
                    for i, segment in enumerate(result["segments"], start=1):
                        start_time_fmt = format_timestamp(segment["start"])
                        end_time_fmt = format_timestamp(segment["end"])
                        text = segment["text"].strip()
                        srt_file.write(f"{i}\n{start_time_fmt} --> {end_time_fmt}\n{text}\n\n")

                del model
                gc.collect()

                elapsed = time.time() - start_time
                logger.info(f"使用基本模式完成字幕生成，耗时: {elapsed:.2f}秒")
                return True
            except Exception as e2:
                logger.error(f"基本模式也失败: {str(e2)}")
                logger.error("请确保安装了正确的OpenAI Whisper包: pip install openai-whisper")
                return False

    except Exception as e:
        logger.error(f"生成英文字幕失败: {str(e)}")
        return False


def translate_subtitles_batch(english_srt, chinese_srt, batch_size=5):
    """分批翻译字幕以减少内存占用"""
    logger.info(f"开始翻译字幕: {os.path.basename(english_srt)}")
    start_time = time.time()

    try:
        # 读取英文字幕
        with open(english_srt, "r", encoding="utf-8") as f:
            content = f.read()

        # 分离成字幕块
        srt_blocks = content.strip().split("\n\n")
        translated_blocks = []
        total_blocks = len(srt_blocks)

        # 获取翻译器
        try:
            translator = argostranslate.translate.get_translation_from_codes("en", "zh")
        except Exception as e:
            logger.warning(f"获取翻译器失败，将使用默认translate方法: {e}")
            translator = None

        # 分批处理
        for i in range(0, total_blocks, batch_size):
            batch = srt_blocks[i:i + batch_size]
            translated_batch = []

            for block in batch:
                lines = block.split("\n")
                if len(lines) >= 3:
                    subtitle_id = lines[0]
                    time_code = lines[1]
                    text = "\n".join(lines[2:])

                    # 翻译文本部分
                    try:
                        if translator:
                            translated_text = translator.translate(text)
                        else:
                            translated_text = argostranslate.translate.translate(text, "en", "zh")
                    except Exception as e:
                        logger.warning(f"翻译失败，使用原文: {e}")
                        translated_text = text

                    translated_batch.append(f"{subtitle_id}\n{time_code}\n{translated_text}")

            translated_blocks.extend(translated_batch)
            # 进度报告
            if (i + batch_size) % 50 == 0 or (i + batch_size) >= total_blocks:
                progress = min((i + batch_size), total_blocks) / total_blocks * 100
                logger.info(f"翻译进度: {progress:.1f}% ({min((i + batch_size), total_blocks)}/{total_blocks})")

        # 保存中文字幕
        with open(chinese_srt, "w", encoding="utf-8") as f:
            f.write("\n\n".join(translated_blocks))

        elapsed = time.time() - start_time
        logger.info(f"字幕翻译完成，耗时: {elapsed:.2f}秒")
        return True

    except Exception as e:
        logger.error(f"翻译字幕失败: {str(e)}")
        return False


def embed_subtitles(video_path, subtitle_path, output_video):
    """使用FFmpeg将字幕嵌入视频"""
    logger.info(f"开始嵌入字幕: {os.path.basename(video_path)}")
    start_time = time.time()

    try:
        cmd = [
            "ffmpeg", "-i", video_path,
            "-i", subtitle_path,
            "-c:v", "copy", "-c:a", "copy",
            "-c:s", "mov_text",
            "-metadata:s:s:0", "language=chi",
            "-hide_banner", "-loglevel", "error",
            output_video
        ]
        subprocess.run(cmd, check=True)

        elapsed = time.time() - start_time
        logger.info(f"字幕嵌入完成，耗时: {elapsed:.2f}秒")
        return True

    except Exception as e:
        logger.error(f"嵌入字幕失败: {str(e)}")
        return False


def process_video(video_path, config, temp_dir):
    """处理单个视频文件"""
    video_filename = os.path.basename(video_path)
    base_name = os.path.splitext(video_filename)[0]

    # 定义输出文件路径
    english_srt = os.path.join(temp_dir, f"{base_name}_en.srt")
    chinese_srt = os.path.join(temp_dir, f"{base_name}_zh.srt")
    output_video = os.path.join(config['OUTPUT_FOLDER'], f"{base_name}_with_cn_subtitles.mp4")

    # 同时保存一份SRT到输出文件夹
    output_en_srt = os.path.join(config['OUTPUT_FOLDER'], f"{base_name}_en.srt")
    output_zh_srt = os.path.join(config['OUTPUT_FOLDER'], f"{base_name}_zh.srt")

    # 字幕单独存放的路径
    if config['ORGANIZE_SUBTITLES_BY_LANGUAGE']:
        subtitle_en_path = os.path.join(config['SUBTITLE_FOLDER'], 'en', f"{base_name}.srt")
        subtitle_zh_path = os.path.join(config['SUBTITLE_FOLDER'], 'zh', f"{base_name}.srt")
    else:
        subtitle_en_path = os.path.join(config['SUBTITLE_FOLDER'], f"{base_name}_en.srt")
        subtitle_zh_path = os.path.join(config['SUBTITLE_FOLDER'], f"{base_name}_zh.srt")

    logger.info(f"===== 开始处理视频: {video_filename} =====")

    # 1. 生成英文字幕
    if config['GENERATE_ENGLISH_SRT']:
        if not generate_english_subtitles(video_path, english_srt, config['WHISPER_MODEL'], temp_dir):
            logger.error(f"无法为视频 {video_filename} 生成英文字幕，跳过后续步骤")
            return False

        # 复制英文字幕到输出文件夹
        shutil.copy(english_srt, output_en_srt)
        logger.info(f"英文字幕已保存: {output_en_srt}")

        # 复制到单独的字幕文件夹（如果启用）
        if config['COPY_SUBTITLES_TO_SEPARATE_FOLDER']:
            shutil.copy(english_srt, subtitle_en_path)
            logger.info(f"英文字幕已单独保存: {subtitle_en_path}")

    # 2. 翻译为中文字幕
    if config['GENERATE_CHINESE_SRT']:
        if not translate_subtitles_batch(english_srt, chinese_srt, config['BATCH_SIZE']):
            logger.error(f"无法为视频 {video_filename} 生成中文字幕，跳过后续步骤")
            return False

        # 复制中文字幕到输出文件夹
        shutil.copy(chinese_srt, output_zh_srt)
        logger.info(f"中文字幕已保存: {output_zh_srt}")

        # 复制到单独的字幕文件夹（如果启用）
        if config['COPY_SUBTITLES_TO_SEPARATE_FOLDER']:
            shutil.copy(chinese_srt, subtitle_zh_path)
            logger.info(f"中文字幕已单独保存: {subtitle_zh_path}")

    # 3. 嵌入中文字幕到视频
    if config['EMBED_SUBTITLES']:
        if not embed_subtitles(video_path, chinese_srt, output_video):
            logger.error(f"无法为视频 {video_filename} 嵌入中文字幕")
            return False
        logger.info(f"带字幕视频已保存: {output_video}")

    logger.info(f"===== 视频处理完成: {video_filename} =====")
    return True


def main():
    """主函数"""
    logger.info("=== 视频字幕生成与翻译工具启动 ===")

    # 加载配置
    config = load_config()
    logger.info(f"已加载配置: {config}")

    # 确保目录存在
    ensure_directories(config)

    # 查找视频
    videos = find_videos(config)
    if not videos:
        logger.warning(f"在 {config['INPUT_FOLDER']} 中没有找到视频文件")
        return

    # 准备翻译模型
    if config['GENERATE_CHINESE_SRT'] or config['EMBED_SUBTITLES']:
        prepare_translation_model()

    # 创建临时目录
    with tempfile.TemporaryDirectory(dir=config['TEMP_FOLDER']) as temp_dir:
        logger.info(f"创建临时工作目录: {temp_dir}")

        # 处理每个视频
        for i, video in enumerate(videos, 1):
            logger.info(f"处理视频 {i}/{len(videos)}: {os.path.basename(video)}")
            process_video(video, config, temp_dir)

            # 强制垃圾回收
            gc.collect()

    logger.info("=== 所有视频处理完成 ===")


if __name__ == "__main__":
    main()
