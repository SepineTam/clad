import os
import sys
from t2j import txt_to_json
from muldi import batch_download

def main():
    if len(sys.argv) != 2:
        print("使用方法: python main.py <input_txt_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    if not os.path.exists(input_file):
        print(f"错误：文件不存在 {input_file}")
        sys.exit(1)
    
    try:
        # 第一步：将txt转换为json
        print("正在将txt文件转换为json格式...")
        json_file = txt_to_json(input_file)
        print(f"转换成功！生成文件: {json_file}")
        
        # 第二步：开始下载视频
        print("\n开始下载视频...")
        batch_download(json_file)
        
    except Exception as e:
        print(f"处理过程中出错: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()