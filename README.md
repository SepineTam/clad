# clad
一个批量下载视频的脚本

## 技术栈
- Python 3.11+
- requests：用于发送HTTP请求下载视频
- json：处理配置文件
- os：处理文件和目录操作

## 使用说明

### 安装依赖
```bash
pip install requests
```

### 使用方法
1. 准备视频链接文件
   - 创建一个文本文件（例如example.txt）
   - 将视频链接按行放入文件中

2. 运行程序
```bash
python main.py example.txt
```

3. 程序会自动：
   - 将文本文件转换为JSON格式
   - 创建对应的文件夹
   - 开始下载视频

### 示例
- 输入文件格式（example.txt）：
```
https://example.com/video1.mp4
https://example.com/video2.mp4
```

## 注意事项
- 确保有足够的磁盘空间
- 保持网络连接稳定
- 视频将按序号命名（No01.mp4, No02.mp4...）

## 常见问题
1. 下载失败
   - 检查网络连接
   - 确认视频链接是否有效
   - 检查磁盘空间是否充足

2. 文件格式错误
   - 确保输入文件中每行只包含一个视频链接
   - 确保链接格式正确
