"""MinerU CLI runner — direct subprocess, no Docker."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def _normalize_backend(raw: str) -> str:
    s = str(raw or "pipeline").strip().lower() or "pipeline"
    return {"vlm": "vlm-auto-engine"}.get(s, s)


def _effective_api_url(backend: str, vlm_http_url: str) -> Optional[str]:
    if backend in ("vlm-http-client", "hybrid-http-client"):
        v = vlm_http_url.strip()
        return v if v else None
    return None


def _build_argv(
    *,
    cmd: str,
    input_path: str,
    output_path: str,
    backend: str,
    method: str,
    lang: str,
    api_url: Optional[str],
    start_page: Optional[int],
    end_page: Optional[int],
) -> List[str]:
    argv = [cmd, "-p", input_path, "-o", output_path, "-b", backend, "-m", method, "-l", lang]
    if api_url:
        # MinerU 2.x: use --url / -u, NOT --api-url (silently ignored via ignore_unknown_options)
        argv.extend(["--url", api_url])
    if start_page is not None:
        argv.extend(["-s", str(start_page)])
    if end_page is not None:
        argv.extend(["-e", str(end_page)])
    return argv


def _subprocess_sync(argv: List[str], timeout_sec: int) -> tuple[int, bytes, bytes]:
    proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        stdout, stderr = proc.communicate(timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        raise RuntimeError(f"mineru timeout after {timeout_sec}s")
    return proc.returncode, stdout or b"", stderr or b""


async def _run_mineru(argv: List[str], timeout_sec: int) -> tuple[bytes, bytes]:
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except NotImplementedError:
        rc, stdout, stderr = await asyncio.to_thread(_subprocess_sync, argv, timeout_sec)
        if rc != 0:
            err = (stderr or b"").decode("utf-8", errors="ignore").strip()[-1200:]
            raise RuntimeError(f"mineru failed (code={rc}): {err}")
        return stdout, stderr

    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_sec)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.communicate()
        raise RuntimeError(f"mineru timeout after {timeout_sec}s")
    rc = int(proc.returncode) if proc.returncode is not None else -1
    if rc != 0:
        err = (stderr or b"").decode("utf-8", errors="ignore").strip()[-1200:]
        raise RuntimeError(f"mineru failed (code={rc}): {err}")
    return stdout or b"", stderr or b""


# ---------------------------------------------------------------------------
# Output collection
# ---------------------------------------------------------------------------

def _infer_page_size(items: List[Dict]) -> tuple[float, float]:
    max_x, max_y = 1.0, 1.0
    for item in items:
        bbox = item.get("bbox")
        if not (isinstance(bbox, list) and len(bbox) == 4):
            continue
        try:
            _, _, x1, y1 = [float(v) for v in bbox]
            max_x, max_y = max(max_x, x1), max(max_y, y1)
        except (TypeError, ValueError):
            pass
    return max_x, max_y


def _normalize_bbox(bbox: Any, pw: float, ph: float) -> Optional[Dict[str, float]]:
    if not (isinstance(bbox, list) and len(bbox) == 4):
        return None
    try:
        x0, y0, x1, y1 = [float(v) for v in bbox]
    except (TypeError, ValueError):
        return None
    if pw <= 0 or ph <= 0:
        return None
    return {
        "x0": max(0.0, min(1.0, x0 / pw)),
        "y0": max(0.0, min(1.0, y0 / ph)),
        "x1": max(0.0, min(1.0, x1 / pw)),
        "y1": max(0.0, min(1.0, y1 / ph)),
    }


def _extract_text(node: Any) -> str:
    texts: List[str] = []
    stack = [node]
    while stack:
        cur = stack.pop()
        if isinstance(cur, dict):
            for k, v in cur.items():
                if k.lower() == "html":
                    continue
                if k.lower() == "content" and isinstance(v, str) and v.strip():
                    texts.append(v.strip())
                elif isinstance(v, (dict, list)):
                    stack.append(v)
        elif isinstance(cur, list):
            stack.extend(cur)
    return "\n".join(texts).strip()


def _table_cells_from_html(html: str) -> List[List[str]]:
    if not html:
        return []
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", html, flags=re.IGNORECASE | re.DOTALL)
    result = []
    for row_html in rows:
        cols = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row_html, flags=re.IGNORECASE | re.DOTALL)
        parsed = [re.sub(r"<[^>]+>", "", col).strip() for col in cols]
        if parsed:
            result.append(parsed)
    return result


def _table_html_from_cells(cells: List[List[str]]) -> str:
    rows = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in cells)
    return f"<table>{rows}</table>"


def collect_structured(output_dir: Path, file_id: str) -> Dict[str, Any]:
    """Parse MinerU output directory into structured elements."""
    # Try content_list_v2.json first (MinerU 3.0+)
    v2 = next(iter(sorted(output_dir.rglob("*_content_list_v2.json"))), None)
    if v2 is not None:
        return _parse_v2(v2, file_id)

    # Fall back to flat content_list.json (MinerU 2.x)
    legacy = next(iter(sorted(output_dir.rglob("*_content_list.json"))), None)
    if legacy is not None:
        return _parse_legacy(legacy, file_id)

    return {"file_id": file_id, "content": "", "elements": [], "metadata": {"parser_mode": "no_structured"}}


def _parse_v2(content_file: Path, file_id: str) -> Dict[str, Any]:
    try:
        payload = json.loads(content_file.read_text(encoding="utf-8", errors="ignore"))
    except (OSError, json.JSONDecodeError):
        return {"file_id": file_id, "content": "", "elements": [], "metadata": {"parser_mode": "invalid_v2"}}
    if not isinstance(payload, list):
        return {"file_id": file_id, "content": "", "elements": [], "metadata": {"parser_mode": "invalid_v2"}}

    elements: List[Dict] = []
    content_units: List[str] = []
    table_count = image_count = 0

    for page_idx, page_items in enumerate(payload, start=1):
        if not isinstance(page_items, list):
            continue
        pw, ph = _infer_page_size(page_items)
        for item in page_items:
            if not isinstance(item, dict):
                continue
            bbox = _normalize_bbox(item.get("bbox"), pw, ph)
            if bbox is None:
                continue
            src = {"file_id": file_id, "page_no": page_idx, "bbox": bbox}
            itype = str(item.get("type") or "").lower()
            content = item.get("content")
            if itype == "page_number":
                continue
            raw_text = _extract_text(content)

            if itype == "table":
                html = content.get("html", "") if isinstance(content, dict) else ""
                cells = _table_cells_from_html(html) or ([[raw_text]] if raw_text else [[""]])
                elements.append({"element_type": "table", "table_cells": cells, "table_html": html or _table_html_from_cells(cells), "source_ref": src})
                table_count += 1
                if raw_text:
                    content_units.append(raw_text)
            elif itype in {"image", "seal", "figure"}:
                caption = _extract_text(content.get("image_caption")) if isinstance(content, dict) else ""
                ocr = raw_text or caption or "[image]"
                elements.append({"element_type": "image", "image_ocr_text": ocr, "image_caption": caption, "source_ref": src})
                image_count += 1
                content_units.append(ocr)
            elif raw_text:
                elements.append({"element_type": "text", "content": raw_text, "source_ref": src})
                content_units.append(raw_text)

    return {
        "file_id": file_id,
        "content": "\n\n".join(content_units),
        "elements": elements,
        "metadata": {
            "parser_mode": "content_list_v2",
            "page_count": len(payload),
            "table_count": table_count,
            "image_count": image_count,
        },
    }


def _join_textish(v: Any) -> str:
    if isinstance(v, str):
        return v.strip()
    if isinstance(v, list):
        return "\n".join(str(x).strip() for x in v if str(x).strip())
    return ""


def _parse_legacy(content_file: Path, file_id: str) -> Dict[str, Any]:
    try:
        payload = json.loads(content_file.read_text(encoding="utf-8", errors="ignore"))
    except (OSError, json.JSONDecodeError):
        return {"file_id": file_id, "content": "", "elements": [], "metadata": {"parser_mode": "invalid_legacy"}}
    if not isinstance(payload, list):
        return {"file_id": file_id, "content": "", "elements": [], "metadata": {"parser_mode": "invalid_legacy"}}

    elements: List[Dict] = []
    content_units: List[str] = []
    table_count = image_count = 0

    for item in payload:
        if not isinstance(item, dict):
            continue
        itype = str(item.get("type") or "").lower()
        if itype == "page_number":
            continue
        page_no = int(item.get("page_idx", 0)) + 1
        bbox = _normalize_bbox(item.get("bbox"), 1000.0, 1000.0) or {"x0": 0.0, "y0": 0.0, "x1": 1.0, "y1": 1.0}
        src: Dict[str, Any] = {"file_id": file_id, "page_no": page_no, "bbox": bbox}

        if itype == "table":
            tb = item.get("table_body")
            html = str(tb).strip() if isinstance(tb, str) else ""
            cells = _table_cells_from_html(html)
            cap = _join_textish(item.get("table_caption"))
            if not cells:
                cells = [[cap]] if cap else [[""]]
            if isinstance(item.get("img_path"), str) and item["img_path"].strip():
                src["asset_ref"] = {"image_path": item["img_path"].strip()}
            elements.append({"element_type": "table", "table_cells": cells, "table_html": html or _table_html_from_cells(cells), "source_ref": src})
            table_count += 1
            if cap:
                content_units.append(cap)
        elif itype in {"image", "chart", "figure", "seal"}:
            cap = _join_textish(item.get("image_caption"))
            ocr = cap or _join_textish(item.get("text")) or "[image]"
            if isinstance(item.get("img_path"), str) and item["img_path"].strip():
                src["asset_ref"] = {"image_path": item["img_path"].strip()}
            elements.append({"element_type": "image", "image_ocr_text": ocr, "image_caption": cap, "source_ref": src})
            image_count += 1
            content_units.append(ocr)
        elif itype == "equation":
            txt = _join_textish(item.get("text"))
            if txt:
                elements.append({"element_type": "text", "content": txt, "source_ref": src})
                content_units.append(txt)
        else:
            txt = _join_textish(item.get("text")) or _join_textish(item.get("code_body"))
            if txt:
                elements.append({"element_type": "text", "content": txt, "source_ref": src})
                content_units.append(txt)

    return {
        "file_id": file_id,
        "content": "\n\n".join(content_units),
        "elements": elements,
        "metadata": {
            "parser_mode": "content_list_legacy",
            "table_count": table_count,
            "image_count": image_count,
        },
    }


def collect_text(output_dir: Path) -> List[str]:
    """Fallback: extract raw text from .md files or JSON text fields."""
    texts: List[str] = []
    for md in sorted(output_dir.rglob("*.md")):
        try:
            body = md.read_text(encoding="utf-8", errors="ignore").strip()
            if body:
                texts.append(body)
        except OSError:
            pass
    if texts:
        return texts
    for jf in sorted(output_dir.rglob("*.json")):
        try:
            payload = json.loads(jf.read_text(encoding="utf-8", errors="ignore"))
        except (OSError, json.JSONDecodeError):
            continue
        stack = [payload]
        while stack:
            item = stack.pop()
            if isinstance(item, dict):
                for k, v in item.items():
                    if k.lower() in {"text", "content"} and isinstance(v, str) and v.strip():
                        texts.append(v.strip())
                    elif isinstance(v, (dict, list)):
                        stack.append(v)
            elif isinstance(item, list):
                stack.extend(item)
    return texts


# ---------------------------------------------------------------------------
# Runner class
# ---------------------------------------------------------------------------

class MineruRunner:
    def __init__(self, settings):
        self.settings = settings

    def _cmd(self) -> str:
        if self.settings.mineru_cmd:
            return self.settings.mineru_cmd
        exe = "mineru.exe" if os.name == "nt" else "mineru"
        return str(Path(sys.executable).parent / exe)

    async def parse(
        self,
        file_path: str,
        *,
        backend: Optional[str] = None,
        lang: Optional[str] = None,
        start_page: Optional[int] = None,
        end_page: Optional[int] = None,
    ) -> Dict[str, Any]:
        backend = _normalize_backend(backend or self.settings.mineru_backend)
        lang = lang or self.settings.mineru_lang
        api_url = _effective_api_url(backend, self.settings.mineru_vlm_http_url)

        work_dir = Path(self.settings.work_dir) / f"mineru_parse_{uuid4().hex}"
        work_dir.mkdir(parents=True, exist_ok=True)
        out_dir = work_dir / "output"
        out_dir.mkdir()

        try:
            argv = _build_argv(
                cmd=self._cmd(),
                input_path=str(Path(file_path).expanduser().resolve()),
                output_path=str(out_dir),
                backend=backend,
                method=self.settings.mineru_method,
                lang=lang,
                api_url=api_url,
                start_page=start_page,
                end_page=end_page,
            )
            logger.info("mineru argv: %s", argv)

            _, stderr = await _run_mineru(argv, self.settings.mineru_timeout_sec)

            result = collect_structured(out_dir, file_id=Path(file_path).name)
            if not result.get("elements"):
                texts = collect_text(out_dir)
                if not texts:
                    stderr_tail = (stderr or b"").decode("utf-8", errors="ignore").strip()[-2000:]
                    return {
                        "success": False,
                        "error": f"MinerU produced empty output. stderr: {stderr_tail}",
                        "content": "",
                        "elements": [],
                        "metadata": {},
                    }
                result["elements"] = [
                    {"element_type": "text", "content": t, "source_ref": {"file_id": Path(file_path).name, "page_no": i + 1}}
                    for i, t in enumerate(texts)
                ]
                result["content"] = "\n\n".join(texts)
                result.setdefault("metadata", {})["parser_mode"] = "fallback_text"

            result["success"] = True
            result.setdefault("metadata", {}).update(
                {
                    "scratch_root": str(work_dir),
                    "backend": backend,
                    "element_count": len(result["elements"]),
                }
            )
            return result

        except Exception as exc:
            shutil.rmtree(work_dir, ignore_errors=True)
            logger.exception("mineru parse failed: %s", file_path)
            return {"success": False, "error": str(exc), "content": "", "elements": [], "metadata": {}}
