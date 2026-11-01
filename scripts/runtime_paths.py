"""Locate a user-provided runtime, a skill-local installation, or legacy cache."""
import os
from pathlib import Path


def default_runtime_root():
    configured = os.environ.get('BGM_ANALYSIS_ROOT')
    if configured:
        return Path(configured).expanduser()
    local = Path(__file__).resolve().parents[1] / '.runtime'
    legacy = Path('D:/SelfMedia/管理/工具/bgm-analysis')
    return local if local.exists() or not legacy.exists() else legacy
