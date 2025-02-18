import json
import sys

def txt_to_json(input_file):
    # 读取txt文件并移除空行
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    
    # 创建JSON数据结构
    result = {}
    for i, url in enumerate(lines, 1):
        # 生成编号（确保两位数格式）
        number = f"No{i:02d}"
        result[number] = url
    
    # 生成输出文件名（将.txt替换为.json）
    output_file = input_file.rsplit('.', 1)[0] + '.json'
    
    # 写入JSON文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
    
    return output_file

def main():
    if len(sys.argv) != 2:
        print("使用方法: python t2j.py <input_txt_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    try:
        output_file = txt_to_json(input_file)
        print(f"转换成功！输出文件: {output_file}")
    except Exception as e:
        print(f"转换失败: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()