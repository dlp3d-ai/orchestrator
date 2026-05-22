from pathlib import Path


MIGRATED_ADAPTERS = [
    "orchestrator/classification/openai_classification_client.py",
    "orchestrator/classification/deepseek_classification_client.py",
    "orchestrator/classification/gemini_classification_client.py",
    "orchestrator/classification/xai_classification_client.py",
    "orchestrator/reaction/openai_reaction_client.py",
    "orchestrator/reaction/deepseek_reaction_client.py",
    "orchestrator/reaction/gemini_reaction_client.py",
    "orchestrator/reaction/xai_reaction_client.py",
    "orchestrator/memory/openai_memory_client.py",
    "orchestrator/memory/deepseek_memory_client.py",
    "orchestrator/memory/gemini_memory_client.py",
    "orchestrator/memory/xai_memory_client.py",
    "orchestrator/conversation/openai_conversation_client.py",
    "orchestrator/conversation/deepseek_conversation_client.py",
    "orchestrator/conversation/gemini_conversation_client.py",
    "orchestrator/conversation/xai_conversation_client.py",
    "orchestrator/conversation/anthropic_conversation_client.py",
    "orchestrator/classification/sensenova_classification_client.py",
    "orchestrator/reaction/sensenova_reaction_client.py",
    "orchestrator/memory/sensenova_memory_client.py",
    "orchestrator/conversation/sensenova_conversation_client.py",
    "orchestrator/classification/minimax_classification_client.py",
    "orchestrator/reaction/minimax_reaction_client.py",
    "orchestrator/memory/minimax_memory_client.py",
    "orchestrator/conversation/minimax_conversation_client.py",
]


def test_migrated_adapters_do_not_import_provider_sdks_directly():
    repo_root = Path(__file__).resolve().parents[2]
    forbidden = ("import openai", "from openai", "import anthropic", "from anthropic")

    offenders = []
    for relative_path in MIGRATED_ADAPTERS:
        text = (repo_root / relative_path).read_text(encoding="utf-8")
        for pattern in forbidden:
            if pattern in text:
                offenders.append(f"{relative_path}: {pattern}")

    assert offenders == []


def test_sensenova_adapters_do_not_use_legacy_jwt_or_http_provider_calls():
    repo_root = Path(__file__).resolve().parents[2]
    sensenova_adapters = [path for path in MIGRATED_ADAPTERS if "sensenova_" in path]
    forbidden = (
        "import jwt",
        "from jwt",
        "import httpx",
        ".post(",
        ".stream(",
        "sensenova_ak",
        "sensenova_sk",
    )

    offenders = []
    for relative_path in sensenova_adapters:
        text = (repo_root / relative_path).read_text(encoding="utf-8")
        for pattern in forbidden:
            if pattern in text:
                offenders.append(f"{relative_path}: {pattern}")

    assert offenders == []


def test_minimax_adapters_do_not_use_direct_http_provider_calls():
    repo_root = Path(__file__).resolve().parents[2]
    minimax_adapters = [path for path in MIGRATED_ADAPTERS if "minimax_" in path]
    forbidden = ("import httpx", ".post(", ".stream(")

    offenders = []
    for relative_path in minimax_adapters:
        text = (repo_root / relative_path).read_text(encoding="utf-8")
        for pattern in forbidden:
            if pattern in text:
                offenders.append(f"{relative_path}: {pattern}")

    assert offenders == []
