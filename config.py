"""Settings loaded from environment variables."""
import os
from dataclasses import dataclass, field


@dataclass
class Settings:
    # Server
    host: str = field(default_factory=lambda: os.getenv("MCP_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.getenv("MCP_PORT", "8765")))

    # MinerU CLI
    mineru_cmd: str = field(default_factory=lambda: os.getenv("MINERU_CMD", ""))
    mineru_backend: str = field(default_factory=lambda: os.getenv("MINERU_BACKEND", "vlm-http-client"))
    mineru_vlm_http_url: str = field(default_factory=lambda: os.getenv("MINERU_VLM_HTTP_URL", "http://localhost:30000"))
    mineru_method: str = field(default_factory=lambda: os.getenv("MINERU_METHOD", "auto"))
    mineru_lang: str = field(default_factory=lambda: os.getenv("MINERU_LANG", "ch"))
    mineru_timeout_sec: int = field(default_factory=lambda: int(os.getenv("MINERU_TIMEOUT_SEC", "1800")))

    # Work dir for temp files
    work_dir: str = field(default_factory=lambda: os.getenv("MINERU_WORK_DIR", "/tmp/mineru_mcp"))
