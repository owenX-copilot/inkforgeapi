# InkForgeAPI

基于 PyTorch 的中文手写识别 API 服务，专为嵌入式设备优化。

## 项目概述

InkForgeAPI 是一个轻量级的手写识别 API 服务，支持中文手写识别（630个常用汉字）。项目设计考虑了嵌入式设备的资源限制（32G TF卡系统盘 + 256G eMMC数据盘），具有以下特点：

- **轻量高效**：基于 HanziTiny 模型，参数量极低，适合边缘部署
- **资源友好**：模型懒加载，避免启动时内存爆炸
- **易于扩展**：抽象架构支持多模型扩展
- **生产就绪**：完整的错误处理、日志记录和监控

## 功能特性

### 核心功能
- ✅ 中文手写识别（630个常用汉字）
- ✅ RESTful API 接口
- ✅ 支持文件上传和 base64 编码
- ✅ Top-K 预测结果返回
- ✅ 模型懒加载
- ✅ 健康检查端点

### 架构特性
- ✅ 抽象识别器接口（BaseRecognizer）
- ✅ 配置驱动的模型管理
- ✅ 预留多模型扩展接口
- ✅ 类型注解和完整文档
- ✅ 错误处理和输入验证

### 部署特性
- ✅ 嵌入式设备优化
- ✅ 内存敏感设计
- ✅ 日志写入 eMMC 数据盘
- ✅ 生产环境配置

## 快速开始

### 1. 环境准备

```bash
# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt

# 对于嵌入式设备（无 GPU），使用 CPU 版本的 PyTorch：
pip install torch==2.1.0+cpu torchvision==0.16.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
```

### 2. 目录结构

```
InkForgeAPI/
├── main.py                 # FastAPI 主应用
├── config.yaml            # 配置文件
├── requirements.txt       # 依赖包列表
├── test_client.py         # 测试客户端
├── recognizers/           # 识别器模块
│   ├── __init__.py
│   ├── base.py           # 抽象基类
│   └── chinese.py        # 中文手写识别器
├── utils/                 # 工具模块
│   ├── __init__.py
│   ├── preprocessing.py  # 图像预处理
│   └── model_loader.py   # 模型加载工具
├── model/                 # 模型定义
│   └── hanzi_tiny.py     # HanziTiny 模型
├── checkpoints/           # 模型检查点
│   ├── best_hanzi_tiny.pth  # 模型权重
│   ├── classes.json      # 字符映射
│   └── train_status.json # 训练状态
├── train/                 # 训练代码
│   └── train_hanzi_tiny.py
└── gui_hanzi_tiny.py     # GUI 应用程序
```

### 3. 启动服务

```bash
# 开发模式（自动重载）
python main.py

# 或使用 uvicorn 直接启动
uvicorn main:app --host 0.0.0.0 --port 20802 --reload

# 生产模式（无重载）
uvicorn main:app --host 0.0.0.0 --port 20802 --workers 1
```

服务启动后，访问以下地址：
- API 文档 (Swagger UI)：http://localhost:20802/docs
- API 文档 (ReDoc)：http://localhost:20802/redoc
- 健康检查：http://localhost:20802/health
- 模型列表：http://localhost:20802/models

### 4. 测试 API

```bash
# 使用测试客户端
python test_client.py --test

# 或使用 curl
curl -X POST -F "file=@test_image.jpg" http://localhost:20802/predict

# 健康检查
curl http://localhost:20802/health

# 查看模型列表
curl http://localhost:20802/models
```

## API 接口

### 1. 健康检查
```
GET /health
```
返回服务状态和模型加载情况。

### 2. 模型列表
```
GET /models
```
返回所有可用模型的信息。

### 3. 文件上传预测
```
POST /predict
```
参数：
- `file`: 图像文件（支持 jpg, png, bmp, tiff）
- `model_name`: 模型名称（默认：chinese_handwriting）
- `top_k`: 返回前K个预测结果（默认：5）

### 4. Base64 预测
```
POST /predict/base64
```
请求体：
```json
{
  "image": "base64编码的图像数据",
  "model_name": "chinese_handwriting",
  "top_k": 5
}
```

## 配置说明

### 配置文件 (config.yaml)

主要配置项：

```yaml
# 系统设置
system:
  log_dir: "/mnt/data/logs"  # 日志目录（eMMC数据盘）
  default_model: "chinese_handwriting"

# 模型配置
models:
  chinese_handwriting:
    type: "chinese"
    model_path: "checkpoints/best_hanzi_tiny.pth"
    config:
      class_mapping_path: "checkpoints/classes.json"
      img_size: 64  # 输入图像尺寸
      mean: 0.5     # 标准化均值
      std: 0.5      # 标准化标准差

# API 端点配置
endpoints:
  predict:
    max_top_k: 20
    max_upload_size: 10485760  # 10MB
```

### 模型配置

当前支持的模型：
- `chinese_handwriting`: 中文手写识别（630个汉字）

预留接口（未来扩展）：
- `mnist_digits`: MNIST 数字识别
- `emnist_alphanum`: EMNIST 字母数字识别
- `symbols`: 数学符号识别

## 模型信息

### HanziTiny 模型
- **输入尺寸**: 64×64 灰度图
- **输出类别**: 630个常用汉字
- **模型大小**: ~1.4MB
- **训练准确率**: 96.37%
- **架构特点**:
  - 深度可分离卷积（DSConv）
  - SE 注意力机制
  - ReLU6 激活函数
  - Global Average Pooling

### 字符映射
字符映射文件：`checkpoints/classes.json`
- 格式：Unicode 编码数组（如 `["#U4e00", "#U4e03", ...]`）
- 解码：`#U4e00` → `0x4e00` → `"一"`

## 部署指南

### 嵌入式设备部署

1. **环境准备**
```bash
# 安装最小化依赖
pip install fastapi uvicorn Pillow opencv-python numpy PyYAML
pip install torch==2.1.0+cpu torchvision==0.16.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
```

2. **目录准备**
```bash
# 创建日志目录（eMMC数据盘）
mkdir -p /mnt/data/logs

# 创建模型目录（如果需要从其他位置加载）
mkdir -p /mnt/data/models
```

3. **服务管理**
```bash
# 使用 systemd 管理服务
sudo nano /etc/systemd/system/inkforgeapi.service
```

服务配置文件示例：
```ini
[Unit]
Description=InkForgeAPI Handwriting Recognition Service
After=network.target

[Service]
User=owenx
WorkingDirectory=/data/aozaiweb/InkForgeAPI
ExecStart=/usr/bin/python3 main.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
Environment="PYTHONPATH=/data/aozaiweb/InkForgeAPI"

[Install]
WantedBy=multi-user.target
```

4. **启动服务**
```bash
sudo systemctl daemon-reload
sudo systemctl enable inkforge
sudo systemctl start inkforge
sudo systemctl status inkforge
```

### 性能优化

1. **内存优化**
   - 模型懒加载：首次请求时加载
   - 图像流式处理：避免一次性加载大图像
   - 定期 GC：配置 `gc_threshold`

2. **CPU 优化**
   - 单工作进程：CPU 密集型任务
   - 请求超时：避免长时间阻塞
   - 并发限制：控制最大并发数

3. **存储优化**
   - 日志写入 eMMC：避免 TF 卡频繁写入
   - 模型文件只读：减少磁盘访问

## 开发指南

### 添加新识别器

1. 创建新的识别器类（继承 `BaseRecognizer`）：
```python
from recognizers.base import BaseRecognizer

class NewRecognizer(BaseRecognizer):
    def load_model(self):
        # 实现模型加载
        pass

    def preprocess(self, image):
        # 实现图像预处理
        pass

    def predict(self, image, top_k=5):
        # 实现预测逻辑
        pass

    def get_class_mapping(self):
        # 返回类别映射
        pass
```

2. 在 `ModelRegistry` 中注册：
```python
# 在 main.py 的 startup_event 中添加
ModelRegistry.register("new_type", NewRecognizer)
```

3. 在 `config.yaml` 中配置：
```yaml
models:
  new_model:
    type: "new_type"
    model_path: "/path/to/model.pth"
    config:
      # 模型特定配置
```

### 测试新功能

```bash
# 运行单元测试（待实现）
python -m pytest tests/

# 运行集成测试
python test_client.py --test

# 性能测试
python test_client.py --test --url http://your-server:20802
```

## 故障排除

### 常见问题

1. **模型加载失败**
   - 检查模型文件路径是否正确
   - 确认 PyTorch 版本兼容性
   - 检查文件权限

2. **内存不足**
   - 启用模型懒加载
   - 减少工作进程数
   - 增加交换空间

3. **预测准确率低**
   - 确保输入图像预处理正确
   - 检查字符映射文件
   - 验证模型训练状态

4. **API 响应慢**
   - 检查网络连接
   - 优化图像大小
   - 调整并发设置

### 日志查看

```bash
# 查看服务日志
sudo journalctl -u inkforgeapi -f

# 查看应用日志
tail -f /mnt/data/logs/inkforge_api.log

# 查看错误日志
grep -i error /mnt/data/logs/inkforge_api.log
```

## 性能指标

### 硬件要求
- **CPU**: 最低 1 核心，推荐 2+ 核心
- **内存**: 最低 512MB，推荐 1GB+
- **存储**: 系统盘 32G，数据盘 256G eMMC
- **网络**: 100Mbps 以太网

### 性能基准
- **单次预测时间**: < 100ms (CPU)
- **并发处理**: 10-20 请求/秒
- **内存占用**: ~200MB（加载模型后）
- **启动时间**: < 5秒

## 安全考虑

1. **输入验证**
   - 图像格式验证
   - 文件大小限制
   - 路径遍历防护

2. **资源限制**
   - 最大文件上传大小
   - 最大并发请求数
   - 请求超时设置

3. **访问控制**
   - CORS 配置（生产环境应限制）
   - 速率限制（待实现）
   - API 密钥认证（待实现）

## 未来扩展

### 计划功能
- [ ] 批量推理支持
- [ ] 模型热更新
- [ ] GPU 加速支持
- [ ] 多语言识别
- [ ] 实时流识别

### 架构改进
- [ ] 分布式部署
- [ ] 负载均衡
- [ ] 缓存机制
- [ ] 监控仪表板

## 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 贡献指南

欢迎提交 Issue 和 Pull Request！

1. Fork 项目
2. 创建功能分支
3. 提交更改
4. 推送到分支
5. 创建 Pull Request

## 联系方式

- 项目主页：https://github.com/yourusername/InkForgeAPI
- 问题反馈：https://github.com/yourusername/InkForgeAPI/issues
- 文档更新：欢迎提交文档改进

---

**注意**: 本项目专为嵌入式设备优化，在生产环境部署前请充分测试。