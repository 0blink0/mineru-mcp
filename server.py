"""MinerU MCP Service — FastMCP SSE server."""
from __future__ import annotations

import argparse
import base64
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

# Load .env before anything else reads os.getenv()
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

from fastmcp import FastMCP

from config import Settings
from mineru_runner import MineruApiClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

settings = Settings()
mcp = FastMCP("mineru-mcp")
client = MineruApiClient(settings)

_jobs: Dict[str, Dict[str, Any]] = {}


def _job_dir(job_id: str) -> Path:
    return Path(settings.work_dir) / "jobs" / job_id


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@mcp.tool()
async def upload_document(
    job_id: str,
    filename: str,
    data_base64: str,
    chunk_index: int = 0,
    total_chunks: int = 1,
) -> Dict[str, Any]:
    """
    Upload a document for parsing. Supports chunked upload for large files.
    chunk_index starts at 0. Send chunks in order; parse is ready when the last chunk is received.
    """
    jdir = _job_dir(job_id)
    jdir.mkdir(parents=True, exist_ok=True)

    suffix = Path(filename).suffix.lower() or ".pdf"
    input_file = jdir / f"input{suffix}"

    data = base64.b64decode(data_base64)
    mode = "wb" if chunk_index == 0 else "ab"
    with open(input_file, mode) as f:
        f.write(data)

    if chunk_index == total_chunks - 1:
        size = input_file.stat().st_size
        _jobs[job_id] = {"status": "uploaded", "file_path": str(input_file), "filename": filename}
        return {"job_id": job_id, "status": "uploaded", "size": size}

    _jobs.setdefault(job_id, {})["status"] = "uploading"
    return {"job_id": job_id, "status": "uploading", "chunk": chunk_index, "received": len(data)}


@mcp.tool()
async def parse_document(
    job_id: str,
    backend: Optional[str] = None,
    lang: Optional[str] = None,
    start_page: Optional[int] = None,
    end_page: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Parse an uploaded document via mineru-api.
    Call upload_document first to get a job_id with status=uploaded.
    """
    job = _jobs.get(job_id)
    if not job or job.get("status") != "uploaded":
        return {"success": False, "error": f"job '{job_id}' not found or not ready — upload first"}

    _jobs[job_id]["status"] = "parsing"
    result = await client.parse(
        job["file_path"],
        backend=backend,
        lang=lang,
        start_page=start_page,
        end_page=end_page,
    )
    _jobs[job_id]["status"] = "done" if result.get("success") else "error"
    return result


@mcp.tool()
async def parse_file(
    file_path: str,
    backend: Optional[str] = None,
    lang: Optional[str] = None,
    start_page: Optional[int] = None,
    end_page: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Parse a file already on the server by its local path.
    Useful when the server has direct access to a shared storage volume.
    """
    return await client.parse(
        file_path,
        backend=backend,
        lang=lang,
        start_page=start_page,
        end_page=end_page,
    )


@mcp.tool()
async def cleanup_job(job_id: str) -> Dict[str, Any]:
    """Remove uploaded file for a job."""
    jdir = _job_dir(job_id)
    if jdir.exists():
        shutil.rmtree(jdir, ignore_errors=True)
    _jobs.pop(job_id, None)
    return {"job_id": job_id, "status": "cleaned"}


@mcp.tool()
async def server_info() -> Dict[str, Any]:
    """Return server configuration and active job count."""
    return {
        "service": "mineru-mcp",
        "mineru_api_url": settings.mineru_api_url,
        "backend": settings.mineru_backend,
        "vlm_url": settings.mineru_vlm_http_url,
        "active_jobs": len(_jobs),
    }


@mcp.tool()
async def mineru_api_openapi() -> Dict[str, Any]:
    """Fetch the OpenAPI spec from mineru-api to inspect available endpoints and parameters."""
    import aiohttp
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{settings.mineru_api_url}/openapi.json",
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                return await resp.json()
    except Exception as exc:
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MinerU MCP Service")
    parser.add_argument("--transport", choices=["stdio", "sse"], default="sse")
    parser.add_argument("--host", default=settings.host)
    parser.add_argument("--port", type=int, default=settings.port)
    args = parser.parse_args()

    if args.transport == "stdio":
        mcp.run(transport="stdio")
    else:
        # Use SSE transport for compatibility with Claude Code / Cursor .mcp.json
        mcp.run(transport="sse", host=args.host, port=args.port)
