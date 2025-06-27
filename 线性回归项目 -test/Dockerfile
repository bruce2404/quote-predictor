# 使用阿里云镜像源
FROM registry.cn-hangzhou.aliyuncs.com/library/python:3.11

# 设置工作目录
WORKDIR /app

# 复制requirements.txt
COPY requirements.txt /app/

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com

# 将当前目录内容复制到工作目录
COPY . /app

# 暴露端口
EXPOSE 5000

# 运行应用
CMD ["python", "app.py"]