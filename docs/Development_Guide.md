# Development Guide

## Adding New AI Services

Taking adding a new TTS service as an example, you need to complete the following steps:

### 1. Create New Client Class

Create a new client file in the `orchestrator/generation/text2speech/` directory, for example `new_tts_client.py`:

```python
from .tts_adapter import TextToSpeechAdapter

class NewTTSClient(TextToSpeechAdapter):
    """New TTS client implementation"""

    AVAILABLE_FOR_STREAM = True  # Whether streaming is supported

    def __init__(self, name: str, **kwargs):
        super().__init__(name=name, **kwargs)
        # Initialize client-specific parameters

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
        """Implement TTS generation logic"""
        # Call third-party TTS API
        # Return dictionary containing audio, speech_text, speech_time, duration
        pass

    async def get_voice_names(self, **kwargs: Any) -> Dict[str, Any]:
        """Return available voice list"""
        return {"voice_id": "voice_name"}
```

### 2. Update Builder

Register the new client in `orchestrator/generation/text2speech/builder.py`:

```python
from .new_tts_client import NewTTSClient

_TTS_ADAPTERS = dict(
    # Existing adapters...
    NewTTSClient=NewTTSClient,
)
```

### 3. Update Configuration Files

Add new TTS adapter configuration in `configs/local.py`, `configs/docker.py`, etc.:

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

### 4. Add Tests

Add test functions in `tests/adapters/test_tts_adapters.py`:

```python
@pytest.mark.asyncio
async def test_new_tts_client_stream():
    """Test new TTS client streaming functionality"""
    # Check environment variables
    api_key = os.environ.get("NEW_TTS_API_KEY")
    if not api_key:
        pytest.skip("NEW_TTS_API_KEY is not set")

    # Configuration and test logic
    tts_client_cfg = dict(
        type="NewTTSClient",
        name="new_tts_client",
    )
    # Test implementation...
```

## Testing

The project includes comprehensive testing using pytest with async test support. Tests cover various components of adapters, aggregators, and IO modules.

### Test Categories

The project includes the following test modules:

- **Adapter Tests** (`tests/adapters/`): Test various service adapters
  - `test_a2f_adapters.py` - Audio2Face adapter tests
  - `test_asr_adapters.py` - Speech recognition adapter tests
  - `test_audio_conversation_adapters.py` - Audio conversation adapter tests
  - `test_classification_adapters.py` - Classification adapter tests
  - `test_conversation_adapter.py` - Conversation adapter tests
  - `test_memory_adapters.py` - Memory adapter tests
  - `test_reaction_adapters.py` - Reaction adapter tests
  - `test_s2m_adapters.py` - Speech-to-motion adapter tests
  - `test_tts_adapters.py` - Text-to-speech adapter tests

- **Aggregator Tests** (`tests/aggregator/`): Test data aggregators
  - `test_blendshapes_aggregator.py` - Blendshapes aggregator tests
  - `test_conversation_aggregator.py` - Conversation aggregator tests
  - `test_tts_reaction_aggregator.py` - TTS reaction aggregator tests

- **IO Tests** (`tests/io/`): Test input/output modules
  - `test_config_client.py` - Configuration client tests
  - `test_memory_client.py` - Memory client tests

### Test Data Preparation

Download required test files and organize them into the correct directory structure:

1. **Create test input directory:**
   ```bash
   mkdir -p input
   ```

2. **Download test files:**
   ```bash
   cd input
   # Download test audio files with different sample rates
   curl -LjO https://github.com/LazyBusyYang/CatStream/releases/download/orchestrator_cicd_files/test_audio_16kHz.wav
   curl -LjO https://github.com/LazyBusyYang/CatStream/releases/download/orchestrator_cicd_files/test_audio_24kHz.wav
   # Create default test audio file
   cp test_audio_16kHz.wav test_audio.wav
   cd ..
   ```

### Running Tests

1. **Run all tests:**
   ```bash
   # Create logs directory
   mkdir -p logs

   # Run all tests
   python -m pytest tests --log-cli-level=ERROR
   ```

2. **Run specific test modules:**
   ```bash
   # Run adapter tests
   python -m pytest tests/adapters/

   # Run aggregator tests
   python -m pytest tests/aggregator/

   # Run IO tests
   python -m pytest tests/io/
   ```

3. **Run specific test files:**
   ```bash
   # Run conversation adapter tests
   python -m pytest tests/adapters/test_conversation_adapter.py

   # Run configuration client tests
   python -m pytest tests/io/test_config_client.py
   ```

## Code Quality

The project maintains high code quality through:

- **Code Inspection**: Using pre-commit hooks for code style and quality checks
- **Type Hints**: Complete type annotation support
- **CI/CD**: Automated testing and deployment pipelines
