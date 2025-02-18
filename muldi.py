import os
import json
import requests
from tqdm import tqdm
import rich
import time
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import random

# 预定义的IP池
IP_POOL = [
    None  # 使用默认IP
]

def setup_logging():
    """设置日志配置"""
    logging.basicConfig(
        filename='download.log',
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def download_video(url, output_path, max_retries=3, timeout=30):
    # 检查文件是否已存在
    if os.path.exists(output_path):
        logging.info(f"文件已存在，跳过下载: {os.path.basename(output_path)}")
        return True

    # 随机选择一个IP
    proxy_ip = random.choice(IP_POOL)
    proxies = {'http': f'http://{proxy_ip}', 'https': f'http://{proxy_ip}'} if proxy_ip else None
    
    for attempt in range(max_retries):
        try:
            start_time = time.time()
            if attempt > 0:
                logging.info(f"第{attempt + 1}次尝试下载: {os.path.basename(output_path)}")
            else:
                logging.info(f"开始下载: {os.path.basename(output_path)}")
            
            response = requests.get(url, stream=True, proxies=proxies, timeout=timeout)
            response.raise_for_status()
            
            # 获取文件大小用于进度条
            total_size = int(response.headers.get('content-length', 0))
            
            # 使用tqdm显示下载进度
            with open(output_path, 'wb') as f, tqdm(
                desc=os.path.basename(output_path),
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for data in response.iter_content(chunk_size=1024):
                    size = f.write(data)
                    pbar.update(size)
            
            end_time = time.time()
            duration = end_time - start_time
            logging.info(f"下载完成: {os.path.basename(output_path)} - 用时: {duration:.2f}秒")
            return True

        except Exception as e:
            error_msg = f"下载失败 {output_path} (尝试 {attempt + 1}/{max_retries}): {str(e)}"
            print(error_msg)
            logging.error(error_msg)
            
            # 如果不是最后一次尝试，等待一段时间后重试
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 5  # 递增等待时间
                time.sleep(wait_time)
                # 重新选择代理
                proxy_ip = random.choice(IP_POOL)
                proxies = {'http': f'http://{proxy_ip}', 'https': f'http://{proxy_ip}'} if proxy_ip else None
            else:
                # 最后一次尝试也失败了
                if os.path.exists(output_path):
                    os.remove(output_path)  # 删除可能存在的不完整文件
                return False

def batch_download(json_file):
    """批量下载视频文件，使用队列管理确保同时下载数不超过4个"""
    try:
        # 设置日志
        setup_logging()
        logging.info(f"开始批量下载任务，使用文件: {json_file}")
        batch_start_time = time.time()
        
        # 读取JSON文件
        with open(json_file, 'r', encoding='utf-8') as f:
            video_urls = json.load(f)
        
        # 创建目标文件夹（使用JSON文件名，去掉.json后缀）
        folder_name = os.path.splitext(os.path.basename(json_file))[0]
        os.makedirs(folder_name, exist_ok=True)
        
        # 准备下载任务
        total_count = len(video_urls)
        success_count = 0
        active_downloads = 0
        max_concurrent = 4  # 最大并发下载数
        
        # 使用线程池进行并行下载
        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            # 创建下载任务队列
            pending_tasks = list(video_urls.items())
            active_futures = {}
            
            while pending_tasks or active_futures:
                # 提交新的下载任务，直到达到最大并发数
                while active_downloads < max_concurrent and pending_tasks:
                    key, url = pending_tasks.pop(0)
                    output_path = os.path.join(folder_name, f"{key}.mp4")
                    future = executor.submit(download_video, url, output_path)
                    active_futures[future] = key
                    active_downloads += 1
                    logging.info(f"开始新的下载任务: {key} (当前活跃下载数: {active_downloads})")
                
                # 等待任意一个任务完成
                if active_futures:
                    for done in as_completed(active_futures.keys()):
                        key = active_futures[done]
                        try:
                            if done.result():
                                success_count += 1
                                logging.info(f"下载成功: {key} (剩余任务: {len(pending_tasks)}")
                            else:
                                logging.error(f"下载失败: {key}")
                        except Exception as e:
                            print(f"\n下载失败 {key}: {str(e)}")
                            logging.error(f"下载失败 {key}: {str(e)}")
                        
                        # 更新活跃下载数
                        active_downloads -= 1
                        del active_futures[done]
                        break
        
        batch_end_time = time.time()
        batch_duration = batch_end_time - batch_start_time
        completion_msg = f"批量下载完成！成功: {success_count}/{total_count} - 总用时: {batch_duration:.2f}秒"
        print(f"\n{completion_msg}")
        logging.info(completion_msg)
        
    except json.JSONDecodeError:
        error_msg = f"错误：无法解析JSON文件 {json_file}"
        print(error_msg)
        logging.error(error_msg)
    except Exception as e:
        error_msg = f"错误：{str(e)}"
        print(error_msg)
        logging.error(error_msg)

def main():
    import sys
    if len(sys.argv) != 2:
        print("使用方法: python muldi.py <json_file>")
        sys.exit(1)
    
    json_file = sys.argv[1]
    if not os.path.exists(json_file):
        print(f"错误：文件不存在 {json_file}")
        sys.exit(1)
    
    batch_download(json_file)

if __name__ == '__main__':
    main()