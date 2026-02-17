"""
InkForgeAPI 主应用
基于 FastAPI 的手写识别 API 服务
"""

import os
import re
# 限制为单线程，提高ARM设备稳定性
os.environ["OMP_NUM_THREADS"] = "1"

import sys
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np

# 添加当前目录到路径
sys.path.append(str(Path(__file__).parent))

from utils.preprocessing import decode_base64_image, validate_image, normalize_handwriting_image
from utils.model_loader import ModelLoader, ModelRegistry
from recognizers.chinese import ChineseHandwritingRecognizer
from recognizers.multi_char import MultiCharHandwritingRecognizer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 创建 FastAPI 应用
app = FastAPI(
    title="InkForgeAPI",
    description="中文手写识别 API 服务",
    version="1.0.0",
    docs_url="/InkForge/docs",
    redoc_url="/InkForge/redoc",
    openapi_url="/InkForge/openapi.json"
)

# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应限制来源
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 访问日志中间件
from fastapi import Request
import time
from starlette.middleware.base import BaseHTTPMiddleware

class AccessLogMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()

        # 处理请求
        response = await call_next(request)

        # 计算响应时间
        process_time = (time.time() - start_time) * 1000  # 毫秒

        # 记录访问日志
        if access_logger:
            access_logger.log_request(request, response, process_time)

        # 添加响应头
        response.headers["X-Process-Time"] = str(process_time)

        return response

# 添加访问日志中间件
app.add_middleware(AccessLogMiddleware)

# 全局模型加载器
model_loader: Optional[ModelLoader] = None


@app.on_event("startup")
async def startup_event():
    """应用启动事件"""
    global model_loader

    try:
        # 初始化访问日志记录器
        init_access_logger()

        # 注册识别器类型
        ModelRegistry.register("chinese", ChineseHandwritingRecognizer)
        ModelRegistry.register("multi_char", MultiCharHandwritingRecognizer)

        # 初始化模型加载器
        config_path = Path(__file__).parent / "config.yaml"
        model_loader = ModelLoader(str(config_path))

        logger.info("InkForgeAPI 启动成功")
        logger.info(f"配置文件: {config_path}")

        # 输出可用模型信息
        models_config = model_loader.config.get("models", {})
        logger.info(f"配置的模型: {list(models_config.keys())}")

        # 预加载所有模型（避免懒加载的并发问题）
        logger.info("开始预加载模型...")
        for model_name in models_config.keys():
            try:
                model_loader.get_model(model_name)
                logger.info(f"预加载模型成功: {model_name}")
            except Exception as e:
                logger.error(f"预加载模型失败 {model_name}: {e}")
        logger.info("所有模型预加载完成")
        
        # 测试模型推理是否正常
        logger.info("测试模型推理...")
        try:
            import torch
            import numpy as np
            test_model = model_loader.get_model("chinese_handwriting")
            test_input = torch.randn(1, 1, 64, 64)
            with torch.no_grad():
                test_output = test_model.model(test_input)
            logger.info(f"模型推理测试成功, 输出形状: {test_output.shape}")
        except Exception as e:
            logger.error(f"模型推理测试失败: {e}")

    except Exception as e:
        logger.error(f"应用启动失败: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭事件"""
    logger.info("InkForgeAPI 正在关闭...")


@app.get("/InkForge")
@app.get("/InkForge/")
async def root():
    """根端点"""
    return {
        "service": "InkForgeAPI",
        "version": "1.0.0",
        "description": "中文手写识别 API 服务",
        "endpoints": {
            "docs": "/InkForge/docs",
            "redoc": "/InkForge/redoc",
            "openapi": "/InkForge/openapi.json",
            "health": "/InkForge/health",
            "models": "/InkForge/models",
            "predict": "/InkForge/predict",
            "predict_base64": "/InkForge/predict/base64",
            "multipredict": "/InkForge/multipredict",
            "multipredict_base64": "/InkForge/multipredict/base64"
        }
    }




@app.get("/InkForge/health")
async def health_check():
    """健康检查端点"""
    global model_loader

    try:
        if model_loader is None:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={"status": "error", "message": "模型加载器未初始化"}
            )

        # 获取已加载模型的状态
        loaded_models = model_loader.get_loaded_models()
        models_status = {}

        for model_name, model_info in loaded_models.items():
            models_status[model_name] = {
                "loaded": model_info.get("is_loaded", False),
                "status": "healthy" if model_info.get("is_loaded") else "not_loaded"
            }

        return {
            "status": "healthy",
            "service": "InkForgeAPI",
            "models": models_status,
            "timestamp": get_current_timestamp()
        }

    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "error", "message": str(e)}
        )


@app.get("/InkForge/models")
async def list_models():
    """列出所有可用模型"""
    global model_loader

    if model_loader is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="模型加载器未初始化"
        )

    try:
        models_config = model_loader.config.get("models", {})
        loaded_models = model_loader.get_loaded_models()

        models_info = {}
        for model_name, config in models_config.items():
            model_info = {
                "name": model_name,
                "type": config.get("type", "unknown"),
                "description": config.get("description", ""),
                "config": config.get("config", {}),
                "loaded": model_name in loaded_models,
                "status": "loaded" if model_name in loaded_models else "not_loaded"
            }

            if model_name in loaded_models:
                model_info.update(loaded_models[model_name])

            models_info[model_name] = model_info

        return {
            "models": models_info,
            "count": len(models_info),
            "timestamp": get_current_timestamp()
        }

    except Exception as e:
        logger.error(f"获取模型列表失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取模型列表失败: {str(e)}"
        )


@app.post("/InkForge/predict")
async def predict(
    file: UploadFile = File(...),
    model_name: str = "chinese_handwriting",
    top_k: int = 5
):
    """
    手写识别预测端点

    Args:
        file: 上传的图像文件
        model_name: 模型名称 (默认为 chinese_handwriting)
        top_k: 返回前K个预测结果 (默认为5)

    Returns:
        预测结果
    """
    global model_loader

    if model_loader is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="模型加载器未初始化"
        )

    # 验证 top_k 参数
    if top_k < 1 or top_k > 20:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="top_k 必须在 1 到 20 之间"
        )

    # 验证模型名称
    if not validate_model_name(model_name):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="模型名称格式无效"
        )

    try:
        # 读取图像文件
        contents = await file.read()
        if not contents:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="上传的文件为空"
            )

        # 将字节转换为 numpy 数组
        import cv2
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

        if image is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="无法解码图像文件"
            )

        # 验证图像
        if not validate_image(image):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="图像格式无效"
            )

        # 获取模型
        try:
            recognizer = model_loader.get_model(model_name)
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"模型 '{model_name}' 未找到: {str(e)}"
            )

        # 执行预测
        predictions = recognizer.predict(image, top_k=top_k)

        # 构建响应
        response = {
            "success": True,
            "model": model_name,
            "predictions": predictions,
            "top_k": top_k,
            "timestamp": get_current_timestamp(),
            "metadata": {
                "file_name": sanitize_filename(file.filename),
                "file_size": len(contents),
                "image_shape": image.shape,
                "image_dtype": str(image.dtype)
            }
        }

        safe_filename = sanitize_filename(file.filename)
        logger.info(f"预测成功: {model_name}, 文件: {safe_filename}")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"预测失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"预测失败: {str(e)}"
        )


@app.post("/InkForge/predict/base64")
async def predict_base64(
    request: Dict[str, Any],
    model_name: str = "chinese_handwriting",
    top_k: int = 5
):
    """
    使用 base64 编码的图像数据进行预测

    Args:
        request: 包含 base64 图像数据的请求体
        model_name: 模型名称
        top_k: 返回前K个预测结果

    Returns:
        预测结果
    """
    global model_loader

    if model_loader is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="模型加载器未初始化"
        )

    # 验证请求体
    if "image" not in request:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="请求体必须包含 'image' 字段"
        )

    # 验证 top_k 参数
    if top_k < 1 or top_k > 20:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="top_k 必须在 1 到 20 之间"
        )

    # 验证模型名称
    if not validate_model_name(model_name):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="模型名称格式无效"
        )

    try:
        # 解码 base64 图像
        image_data = request["image"]
        image = decode_base64_image(image_data)

        # 验证图像
        if not validate_image(image):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="图像格式无效"
            )

        # 获取模型
        try:
            recognizer = model_loader.get_model(model_name)
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"模型 '{model_name}' 未找到: {str(e)}"
            )

        # 执行预测
        predictions = recognizer.predict(image, top_k=top_k)

        # 构建响应
        response = {
            "success": True,
            "model": model_name,
            "predictions": predictions,
            "top_k": top_k,
            "timestamp": get_current_timestamp(),
            "metadata": {
                "image_shape": image.shape,
                "image_dtype": str(image.dtype)
            }
        }

        logger.info(f"Base64 预测成功: {model_name}")
        return response

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"图像解码失败: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Base64 预测失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"预测失败: {str(e)}"
        )


# ================= 多字识别端点 =================

@app.post("/InkForge/multipredict")
async def multipredict(
    file: UploadFile = File(...),
    model_name: str = "multi_char_handwriting",
    top_k: int = 5
):
    """
    多字手写识别预测端点
    
    使用 Atomic-DP 方法识别手写行图像中的多个字符
    
    Args:
        file: 上传的图像文件（手写行图像）
        model_name: 模型名称 (默认为 multi_char_handwriting)
        top_k: 每个字符返回前K个预测结果 (默认为5)
        
    Returns:
        识别结果，包含:
        - text: 识别出的文本
        - characters: 每个字符的详细信息
        - num_characters: 识别出的字符数
        - num_blocks: 原子块数量
        - timing: 各阶段耗时
    """
    global model_loader

    if model_loader is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="模型加载器未初始化"
        )

    # 验证 top_k 参数
    if top_k < 1 or top_k > 20:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="top_k 必须在 1 到 20 之间"
        )

    # 验证模型名称
    if not validate_model_name(model_name):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="模型名称格式无效"
        )

    try:
        # 读取图像文件
        contents = await file.read()
        if not contents:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="上传的文件为空"
            )

        # 将字节转换为 numpy 数组
        import cv2
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

        if image is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="无法解码图像文件"
            )

        # 验证图像 - 多字识别允许更宽的图像
        if not isinstance(image, np.ndarray):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="图像格式无效：不是有效的numpy数组"
            )
        
        if len(image.shape) not in [2, 3]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"图像格式无效：维度不正确 {len(image.shape)}"
            )
        
        height, width = image.shape[:2]
        
        # 多字识别允许更宽的图像（最大4096宽度）
        if height < 16 or width < 16:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"图像尺寸过小：{width}x{height}，最小需要16x16"
            )
        
        if height > 1024 or width > 4096:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"图像尺寸过大：{width}x{height}，最大支持4096x1024"
            )
        
        if image.dtype not in [np.uint8, np.float32, np.float64]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"图像数据类型无效：{image.dtype}"
            )

        # 标准化手写图像格式（确保灰度图，uint8类型）
        # 不改变颜色，假设前端已经确保是白底灰字
        image = normalize_handwriting_image(image)

        # 获取模型
        try:
            recognizer = model_loader.get_model(model_name)
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"模型 '{model_name}' 未找到: {str(e)}"
            )

        # 执行多字预测
        result = recognizer.predict(image, top_k=top_k)

        # 构建响应
        response = {
            "success": result.get("success", True),
            "model": model_name,
            "text": result.get("text", ""),
            "characters": result.get("characters", []),
            "num_characters": result.get("num_characters", 0),
            "num_blocks": result.get("num_blocks", 0),
            "total_cost": result.get("total_cost", 0),
            "timing": result.get("timing", {}),
            "timestamp": get_current_timestamp(),
            "metadata": {
                "file_name": sanitize_filename(file.filename),
                "file_size": len(contents),
                "image_shape": list(image.shape),
                "image_dtype": str(image.dtype)
            }
        }

        if "error" in result:
            response["error"] = result["error"]

        safe_filename = sanitize_filename(file.filename)
        logger.info(f"多字预测成功: {model_name}, 文件: {safe_filename}, 识别结果: {result.get('text', '')}")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"多字预测失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"多字预测失败: {str(e)}"
        )


@app.post("/InkForge/multipredict/base64")
async def multipredict_base64(
    request: Dict[str, Any],
    model_name: str = "multi_char_handwriting",
    top_k: int = 5
):
    """
    使用 base64 编码的图像数据进行多字识别
    
    Args:
        request: 包含 base64 图像数据的请求体
        model_name: 模型名称
        top_k: 每个字符返回前K个预测结果
        
    Returns:
        识别结果
    """
    global model_loader

    if model_loader is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="模型加载器未初始化"
        )

    # 验证请求体
    if "image" not in request:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="请求体必须包含 'image' 字段"
        )

    # 验证 top_k 参数
    if top_k < 1 or top_k > 20:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="top_k 必须在 1 到 20 之间"
        )

    # 验证模型名称
    if not validate_model_name(model_name):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="模型名称格式无效"
        )

    try:
        # 解码 base64 图像
        image_data = request["image"]
        image = decode_base64_image(image_data)

        # 验证图像 - 多字识别允许更宽的图像
        if not isinstance(image, np.ndarray):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="图像格式无效：不是有效的numpy数组"
            )
        
        if len(image.shape) not in [2, 3]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"图像格式无效：维度不正确 {len(image.shape)}"
            )
        
        height, width = image.shape[:2]
        
        # 多字识别允许更宽的图像（最大4096宽度）
        if height < 16 or width < 16:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"图像尺寸过小：{width}x{height}，最小需要16x16"
            )
        
        if height > 1024 or width > 4096:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"图像尺寸过大：{width}x{height}，最大支持4096x1024"
            )
        
        if image.dtype not in [np.uint8, np.float32, np.float64]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"图像数据类型无效：{image.dtype}"
            )

        # 标准化手写图像格式（确保灰度图，uint8类型）
        # 不改变颜色，假设前端已经确保是白底灰字
        image = normalize_handwriting_image(image)

        # 获取模型
        try:
            recognizer = model_loader.get_model(model_name)
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"模型 '{model_name}' 未找到: {str(e)}"
            )

        # 执行多字预测
        result = recognizer.predict(image, top_k=top_k)

        # 构建响应
        response = {
            "success": result.get("success", True),
            "model": model_name,
            "text": result.get("text", ""),
            "characters": result.get("characters", []),
            "num_characters": result.get("num_characters", 0),
            "num_blocks": result.get("num_blocks", 0),
            "total_cost": result.get("total_cost", 0),
            "timing": result.get("timing", {}),
            "timestamp": get_current_timestamp(),
            "metadata": {
                "image_shape": list(image.shape),
                "image_dtype": str(image.dtype)
            }
        }

        if "error" in result:
            response["error"] = result["error"]

        logger.info(f"Base64 多字预测成功: {model_name}, 识别结果: {result.get('text', '')}")
        return response

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"图像解码失败: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Base64 多字预测失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"多字预测失败: {str(e)}"
        )


def get_current_timestamp() -> str:
    """获取当前时间戳"""
    from datetime import datetime
    return datetime.now().isoformat()


# ================= 访问日志记录 =================

class AccessLogger:
    """访问日志记录器（CSV格式）"""

    def __init__(self, log_dir: str = "/mnt/data/logs"):
        import os
        from pathlib import Path
        import csv

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # 每天一个日志文件
        from datetime import datetime
        date_str = datetime.now().strftime("%Y-%m-%d")
        self.log_file = self.log_dir / f"access_{date_str}.csv"

        # CSV文件头
        self.headers = [
            "timestamp", "client_ip", "method", "path",
            "status_code", "response_time_ms", "user_agent", "referer"
        ]

        # 如果文件不存在，创建并写入表头
        if not self.log_file.exists():
            with open(self.log_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(self.headers)

    def log_request(self, request, response, response_time_ms: float):
        """记录请求日志"""
        import csv
        from datetime import datetime

        try:
            # 获取客户端IP
            client_ip = request.client.host if request.client else "unknown"

            # 获取User-Agent
            user_agent = request.headers.get("user-agent", "")

            # 获取Referer
            referer = request.headers.get("referer", "")

            # 构建日志记录
            record = [
                datetime.now().isoformat(),
                client_ip,
                request.method,
                str(request.url.path),
                response.status_code,
                f"{response_time_ms:.2f}",
                user_agent[:200],  # 限制长度
                referer[:200]      # 限制长度
            ]

            # 写入CSV
            with open(self.log_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(record)

        except Exception as e:
            logger.error(f"访问日志记录失败: {e}")


# 全局访问日志记录器
access_logger = None

def init_access_logger():
    """初始化访问日志记录器"""
    global access_logger
    try:
        import yaml
        with open("config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        log_dir = config.get('system', {}).get('log_dir', '/mnt/data/logs')
        access_logger = AccessLogger(log_dir)
        logger.info(f"访问日志记录器初始化完成，目录: {log_dir}")
    except Exception as e:
        logger.warning(f"访问日志记录器初始化失败，使用默认目录: {e}")
        access_logger = AccessLogger()


# ================= 安全辅助函数 =================

def sanitize_filename(filename: str) -> str:
    """清理文件名，移除潜在危险字符"""
    if not filename:
        return "unknown"
    # 只保留字母、数字、点、下划线、连字符
    sanitized = re.sub(r'[^\w\.\-]', '_', filename)
    # 限制长度
    return sanitized[:100]


def validate_model_name(model_name: str) -> bool:
    """验证模型名称是否合法"""
    # 只允许字母、数字、下划线、连字符
    if not re.match(r'^[a-zA-Z0-9_\-]+$', model_name):
        return False
    # 限制长度
    if len(model_name) > 50:
        return False
    return True


def validate_base64_data(image_data: str, max_size_mb: int = 10) -> bool:
    """验证Base64数据是否合法"""
    # 检查长度
    if len(image_data) > max_size_mb * 1024 * 1024:
        return False

    # 检查是否为有效的Base64（基本检查）
    if not re.match(r'^[A-Za-z0-9+/]*={0,2}$', image_data):
        return False

    return True


if __name__ == "__main__":
    # 当直接运行main.py时，仍然支持直接启动（开发时使用）
    # 生产环境由systemd通过uvicorn命令启动
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=20802,
        reload=False,  # 生产环境设为 False
        log_level="info"
    )