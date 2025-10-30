# 开发指南

本指南说明如何开发、扩展与测试 Orchestrator。

## 新增 AI 服务

以新增一个 TTS 服务为例，需要完成如下步骤：

### 1. 创建新的客户端类

在 `orchestrator/generation/text2speech/` 目录下创建新文件，如 `new_tts_client.py`：

```python
from .tts_adapter import TextToSpeechAdapter

class NewTTSClient(TextToSpeechAdapter):
    """New TTS client implementation"""

    AVAILABLE_FOR_STREAM = True  # 是否支持流式

    def __init__(self, name: str, **kwargs):
        super().__init__(name=name, **kwargs)
        # 初始化客户端特定参数

    async def _generate_tts(
        self,
        request_id: str,
        text: str,
        voice_name: str,
        voice_speed: float = 1.0,
        voice_style: Union[None, str] = None,
        language: str = "zh",
        start_time: float = 0.0,
    ) -> Dict[str, Any]:
        """实现 TTS 生成逻辑"""
        # 调用第三方 TTS API
        # 返回包含 audio、speech_text、speech_time、duration 的字典
        pass

    async def get_voice_names(self, **kwargs: Any) -> Dict[str, Any]:
        """返回可用音色列表"""
        return {"voice_id": "voice_name"}
```

### 2. 更新构建器

在 `orchestrator/generation/text2speech/builder.py` 中注册新客户端：

```python
from .new_tts_client import NewTTSClient

_TTS_ADAPTERS = dict(
    # Existing adapters...
    NewTTSClient=NewTTSClient,
)
```

### 3. 更新配置文件

在 `configs/local.py`、`configs/docker.py` 等配置中添加新 TTS 适配器配置：

```python
tts_adapters=dict(
    # Existing adapters...
    new_tts=dict(
        type="NewTTSClient",
        name="new_tts_client",
        # Client-specific parameters...
    ),
),
```

### 4. 添加测试

在 `tests/adapters/test_tts_adapters.py` 中添加测试函数：

```python
@pytest.mark.asyncio
async def test_new_tts_client_stream():
    """Test new TTS client streaming functionality"""
    # 检查环境变量
    api_key = os.environ.get("NEW_TTS_API_KEY")
    if not api_key:
        pytest.skip("NEW_TTS_API_KEY is not set")

    # 配置与测试逻辑
    tts_client_cfg = dict(
        type="NewTTSClient",
        name="new_tts_client",
    )
    # Test implementation...
```

## 测试

项目使用 pytest（支持异步）进行全面测试，覆盖适配器、聚合器与 IO 模块等组件。

### 测试分类

项目包含以下测试模块：

- 适配器测试（`tests/adapters/`）：覆盖各类服务适配器
  - `test_a2f_adapters.py` - Audio2Face 适配器
  - `test_asr_adapters.py` - 语音识别适配器
  - `test_audio_conversation_adapters.py` - 音频对话适配器
  - `test_classification_adapters.py` - 分类适配器
  - `test_conversation_adapter.py` - 文本对话适配器
  - `test_memory_adapters.py` - 记忆适配器
  - `test_reaction_adapters.py` - 反应适配器
  - `test_s2m_adapters.py` - 语音转动作适配器
  - `test_tts_adapters.py` - 语音合成适配器

- 聚合器测试（`tests/aggregator/`）：覆盖数据聚合器
  - `test_blendshapes_aggregator.py` - 面部表情聚合器
  - `test_conversation_aggregator.py` - 对话聚合器
  - `test_tts_reaction_aggregator.py` - TTS 反应聚合器

- IO 测试（`tests/io/`）：覆盖输入/输出模块
  - `test_config_client.py` - 配置客户端
  - `test_memory_client.py` - 记忆客户端

### 测试数据准备

下载所需测试文件并放置在正确的目录结构：

1. 创建测试输入目录：
   ```bash
   mkdir -p input
   ```

2. 下载测试文件：
   ```bash
   cd input
   # 下载不同采样率的测试音频
   curl -LjO https://github.com/LazyBusyYang/CatStream/releases/download/orchestrator_cicd_files/test_audio_16kHz.wav
   curl -LjO https://github.com/LazyBusyYang/CatStream/releases/download/orchestrator_cicd_files/test_audio_24kHz.wav
   # 生成默认测试音频
   cp test_audio_16kHz.wav test_audio.wav
   cd ..
   ```

### 运行测试

1. 运行全部测试：
   ```bash
   # 创建日志目录
   mkdir -p logs

   # 运行全部测试
   python -m pytest tests --log-cli-level=ERROR
   ```

2. 运行指定模块：
   ```bash
   # 适配器测试
   python -m pytest tests/adapters/

   # 聚合器测试
   python -m pytest tests/aggregator/

   # IO 测试
   python -m pytest tests/io/
   ```

3. 运行指定文件：
   ```bash
   # 对话适配器测试
   python -m pytest tests/adapters/test_conversation_adapter.py

   # 配置客户端测试
   python -m pytest tests/io/test_config_client.py
   ```

## 代码质量

项目通过以下方式保障代码质量：

- 代码检查：使用 pre-commit 进行代码风格与质量检查
- 类型标注：完善的类型注解支持
- CI/CD：自动化测试与部署流水线

## Lint & 格式化

使用 `setup.cfg` 中的项目配置：

```bash
python -m pip install flake8 black isort
flake8
black --check .
isort --check-only .
```

## 日志与调试

- 运行时日志：`logs/`
- 通过设置 `LOG_LEVEL=DEBUG` 提高日志详细程度
- 如需调试流式处理，可开启模块级调试日志并查看聚合器输出

## 部署说明

- 为一致性推荐 Docker
- 仅以环境变量注入必要的 API Key
- 多服务部署时，确保 MongoDB 与 3D 引擎（A2F/S2M）可达
