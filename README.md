# InkForgeAPI

基于 PyTorch 的中文手写识别 API 服务，专为嵌入式设备优化。

## 项目概述

InkForgeAPI 是一个轻量级的手写识别 API 服务，支持中文手写识别（995个常用汉字）。项目设计考虑了嵌入式设备的资源限制，具有以下特点：

- **轻量高效**：基于 HanziTiny 模型，参数量极低，适合边缘部署
- **资源友好**：模型懒加载，避免启动时内存爆炸
- **易于扩展**：抽象架构支持多模型扩展
- **生产就绪**：完整的错误处理、日志记录和监控
- **多字识别**：支持 Atomic-DP 方法的多字符手写行识别

## 功能特性

### 核心功能
- ✅ 中文手写识别（995个常用汉字）
- ✅ RESTful API 接口
- ✅ 支持文件上传和 base64 编码
- ✅ 支持单行识别和单字识别
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
- ✅ 生产环境配置

## 快速开始

### 使用在线服务
本项目提供在线API服务，可直接调用：
- 单字识别：`POST https://api.aozai.top/InkForge/predict`
- 多字识别：`POST https://api.aozai.top/InkForge/multipredict`
- API文档：https://api.aozai.top/InkForge/docs

### 1. 环境准备（本地部署）

```bash
# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt

# 对于嵌入式设备（无 GPU），使用 CPU 版本的 PyTorch：
pip install torch==2.1.0+cpu torchvision==0.16.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

# 复制配置文件示例
cp config.example.yaml config.yaml
```

### 2. 目录结构

```
InkForgeAPI/
├── main.py                 # FastAPI 主应用
├── config.example.yaml     # 配置文件示例（复制为config.yaml使用）
├── requirements.txt       # 依赖包列表
├── recognizers/           # 识别器模块
│   ├── __init__.py
│   ├── base.py           # 抽象基类
│   ├── chinese.py        # 中文手写识别器
│   └── multi_char.py     # 多字符识别器（Atomic-DP方法）
├── utils/                 # 工具模块
│   ├── __init__.py
│   ├── preprocessing.py  # 图像预处理
│   ├── model_loader.py   # 模型加载工具
│   ├── atomic_segmentation.py  # 原子分割
│   ├── candidate_graph.py      # 候选图构建
│   └── dp_search.py            # 动态规划搜索
├── model/                 # 模型定义
│   └── hanzi_tiny.py     # HanziTiny 模型
├── checkpoints/           # 模型检查点
│   ├── best.pth          # 模型权重
│   ├── classes.json      # 字符映射（995个汉字）
│   └── train_status.json # 训练状态
└── *.py                  # 其他Python脚本
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

**在线API服务地址**：
- API 文档 (Swagger UI)：https://api.aozai.top/InkForge/docs
- API 文档 (ReDoc)：https://api.aozai.top/InkForge/redoc
- 健康检查：https://api.aozai.top/InkForge/health
- 模型列表：https://api.aozai.top/InkForge/models

如需本地部署，服务启动后可通过相应端口访问上述端点。


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
- `chinese_handwriting`: 中文手写识别（995个常用汉字）
- `multi_char_handwriting`: 多字符手写行识别（Atomic-DP方法，基于相同单字模型）

配置文件中的预留接口（注释状态）：
- `mnist_digits`: MNIST 数字识别
- `emnist_alphanum`: EMNIST 字母数字识别
- `symbols`: 数学符号识别

## 模型信息

### HanziTiny 模型
- **输入尺寸**: 64×64 灰度图
- **输出类别**: 995个常用汉字
- **模型大小**: ~1.4MB
- **训练准确率**: 96.37%
- **架构特点**:
  - 深度可分离卷积（DSConv）
  - SE 注意力机制
  - ReLU6 激活函数
  - Global Average Pooling

### 多字符识别（Atomic-DP方法）
- **方法**: 原子分割 + 动态规划路径搜索
- **支持**: 手写行识别，自动分割连续字符
- **特点**: 基于相同单字模型，无需额外训练
- **配置**: 可调整分割阈值、形状惩罚权重等参数

### 字符映射
字符映射文件：`checkpoints/classes.json`
- 格式：Unicode 编码数组（如 `["#U4e00", "#U4e03", ...]`）
- 解码：`#U4e00` → `0x4e00` → `"一"`

## 部署指南

### 嵌入式设备部署

1. **环境准备**
```bash
# 安装最小化依赖（参考requirements.txt）
pip install fastapi uvicorn Pillow opencv-python numpy PyYAML
pip install torch==2.6.0 torchvision  # 或使用CPU版本：torch==2.6.0+cpu
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
   - 模型文件只读：减少磁盘访问

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
```

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

### 已实现功能
- [x] 中文手写识别（995汉字）
- [x] RESTful API 服务
- [x] 多字符手写行识别（Atomic-DP）
- [x] 本地GUI应用程序
- [x] 完整的配置管理系统

### 计划功能
- [ ] 批量推理支持
- [ ] 模型热更新
- [ ] 多语言识别扩展
- [ ] 实时流识别

### 架构改进
- [ ] 分布式部署支持
- [ ] 负载均衡优化
- [ ] 高级缓存机制
- [ ] 监控仪表板

## 贡献指南

欢迎提交 Issue 和 Pull Request！

1. Fork 项目
2. 创建功能分支
3. 提交更改
4. 推送到分支
5. 创建 Pull Request

---

**注意**: 本项目专为嵌入式设备优化，在生产环境部署前请充分测试。