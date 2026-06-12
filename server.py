"""MinerU MCP Service — FastMCP SSE server."""
from __future__ import annotations

import argparse
import base64
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

from fastmcp import FastMCP

from config import Settings
from mineru_runner import MineruRunner

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

settings = Settings()
mcp = FastMCP("mineru-mcp")
runner = MineruRunner(settings)

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

    For a single file: chunk_index=0, total_chunks=1.
    For chunked upload: send chunks in order from 0 to total_chunks-1.
    The file is ready to parse when the last chunk (chunk_index == total_chunks-1) is received.
    """
    jdir = _job_dir(job_id)
    jdir.mkdir(parents=True, exist_ok=True)

    suffix = Path(filename).suffix.lower() or ".pdf"
    input_file = jdir / f"input{suffix}"

    data = base64.b64decode(data_base64)
    mode = "wb" if chunk_index == 0 else "ab"
    with open(input_file, mode) as f:
        f.write(data)

    is_last = chunk_index == total_chunks - 1
    if is_last:
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
    Parse an uploaded document with MinerU.
    Call upload_document first to get a job_id with status=uploaded.
    """
    job = _jobs.get(job_id)
    if not job or job.get("status") != "uploaded":
        return {"success": False, "error": f"job '{job_id}' not found or not ready — upload first"}

    _jobs[job_id]["status"] = "parsing"
    result = await runner.parse(
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
    Parse a file that already exists on the server by its local path.
    Useful when the server has direct access to a shared storage volume.
    """
    return await runner.parse(
        file_path,
        backend=backend,
        lang=lang,
        start_page=start_page,
        end_page=end_page,
    )


@mcp.tool()
async def cleanup_job(job_id: str) -> Dict[str, Any]:
    """Remove uploaded file and MinerU scratch directories for a job."""
    jdir = _job_dir(job_id)
    if jdir.exists():
        shutil.rmtree(jdir, ignore_errors=True)

    job = _jobs.pop(job_id, None)
    if job:
        scratch = job.get("scratch_root")
        if scratch and Path(scratch).name.startswith("mineru_parse_"):
            shutil.rmtree(scratch, ignore_errors=True)

    return {"job_id": job_id, "status": "cleaned"}


@mcp.tool()
async def server_info() -> Dict[str, Any]:
    """Return server configuration and active job count."""
    return {
        "service": "mineru-mcp",
        "backend": settings.mineru_backend,
        "vlm_url": settings.mineru_vlm_http_url,
        "active_jobs": len(_jobs),
    }


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
        mcp.run()
    else:
        import uvicorn
        uvicorn.run(mcp.http_app(), host=args.host, port=args.port)
