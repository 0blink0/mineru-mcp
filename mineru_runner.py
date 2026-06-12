"""MinerU HTTP API client — calls mineru-api REST service."""
from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp

logger = logging.getLogger(__name__)

_POLL_INTERVAL = 3  # seconds between task status polls


# ---------------------------------------------------------------------------
# Content list parsing (flat format from MinerU 2.x API response)
# ---------------------------------------------------------------------------

def _join_textish(v: Any) -> str:
    if isinstance(v, str):
        return v.strip()
    if isinstance(v, list):
        return "\n".join(str(x).strip() for x in v if str(x).strip())
    return ""


def _normalize_bbox(bbox: Any) -> Dict[str, float]:
    default = {"x0": 0.0, "y0": 0.0, "x1": 1.0, "y1": 1.0}
    if not (isinstance(bbox, list) and len(bbox) == 4):
        return default
    try:
        x0, y0, x1, y1 = [float(v) for v in bbox]
        # MinerU 2.x bbox is in pixel coords; normalize to [0,1] with max 1000
        pw = ph = 1000.0
        return {
            "x0": max(0.0, min(1.0, x0 / pw)),
            "y0": max(0.0, min(1.0, y0 / ph)),
            "x1": max(0.0, min(1.0, x1 / pw)),
            "y1": max(0.0, min(1.0, y1 / ph)),
        }
    except (TypeError, ValueError):
        return default


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


def parse_content_list(items: List[Dict], file_id: str) -> Dict[str, Any]:
    """Convert MinerU 2.x flat content_list into our element format."""
    elements: List[Dict] = []
    content_units: List[str] = []
    table_count = image_count = 0

    for item in items:
        if not isinstance(item, dict):
            continue
        itype = str(item.get("type") or "").lower()
        if itype == "page_number":
            continue

        page_no = int(item.get("page_idx", 0)) + 1
        src: Dict[str, Any] = {
            "file_id": file_id,
            "page_no": page_no,
            "bbox": _normalize_bbox(item.get("bbox")),
        }

        if itype == "table":
            tb = item.get("table_body")
            html = str(tb).strip() if isinstance(tb, str) else ""
            cells = _table_cells_from_html(html)
            cap = _join_textish(item.get("table_caption"))
            if not cells:
                cells = [[cap]] if cap else [[""]]
            if isinstance(item.get("img_path"), str) and item["img_path"].strip():
                src["asset_ref"] = {"image_path": item["img_path"].strip()}
            elements.append({
                "element_type": "table",
                "table_cells": cells,
                "table_html": html or _table_html_from_cells(cells),
                "source_ref": src,
            })
            table_count += 1
            if cap:
                content_units.append(cap)

        elif itype in {"image", "chart", "figure", "seal"}:
            cap = _join_textish(item.get("image_caption"))
            ocr = cap or _join_textish(item.get("text")) or "[image]"
            if isinstance(item.get("img_path"), str) and item["img_path"].strip():
                src["asset_ref"] = {"image_path": item["img_path"].strip()}
            elements.append({
                "element_type": "image",
                "image_ocr_text": ocr,
                "image_caption": cap,
                "source_ref": src,
            })
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
            "parser_mode": "content_list",
            "table_count": table_count,
            "image_count": image_count,
        },
    }


# ---------------------------------------------------------------------------
# HTTP client
# ---------------------------------------------------------------------------

class MineruApiClient:
    def __init__(self, settings):
        self.api_url = settings.mineru_api_url.rstrip("/")
        self.vlm_url = settings.mineru_vlm_http_url
        self.backend = settings.mineru_backend
        self.lang = settings.mineru_lang
        self.timeout = settings.mineru_timeout_sec

    def _form_data(
        self,
        file_path: str,
        *,
        backend: Optional[str],
        lang: Optional[str],
        start_page: Optional[int],
        end_page: Optional[int],
    ) -> aiohttp.FormData:
        backend = backend or self.backend
        lang = lang or self.lang

        data = aiohttp.FormData()
        data.add_field(
            "files",
            open(file_path, "rb"),
            filename=Path(file_path).name,
            content_type="application/octet-stream",
        )
        data.add_field("backend", backend)
        # lang_list is a Form array — send as repeated fields, not JSON string
        for l in (lang.split(",") if "," in lang else [lang]):
            data.add_field("lang_list", l.strip())
        data.add_field("return_content_list", "true")
        data.add_field("return_md", "true")

        if backend in ("vlm-http-client", "hybrid-http-client"):
            data.add_field("server_url", self.vlm_url)

        if start_page is not None:
            data.add_field("start_page_id", str(start_page))
        if end_page is not None:
            data.add_field("end_page_id", str(end_page))

        return data

    async def _call_file_parse(self, file_path: str, **kwargs) -> Dict[str, Any]:
        """POST to /file_parse (synchronous endpoint in MinerU 2.7.x)."""
        data = self._form_data(file_path, **kwargs)
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.api_url}/file_parse",
                data=data,
                timeout=aiohttp.ClientTimeout(total=self.timeout),
            ) as resp:
                raw = await resp.text()
                logger.info("file_parse status=%s body_head=%s", resp.status, raw[:500])
                if resp.status != 200:
                    raise RuntimeError(f"mineru-api /file_parse returned {resp.status}: {raw[:500]}")
                import json as _json
                return _json.loads(raw)

    def _build_result(self, api_response: Dict, file_id: str) -> Dict[str, Any]:
        # MinerU 2.7.x /file_parse returns {filename: {md, content_list, ...}}
        # Try each known wrapper key, then fall back to treating the whole response as the result
        first: Dict = {}
        for key in ("results", "data"):
            val = api_response.get(key)
            if isinstance(val, dict) and val:
                first = next(iter(val.values()), {})
                break
            if isinstance(val, list) and val and isinstance(val[0], dict):
                first = val[0]
                break
        if not first:
            # Flat response: top-level keys are filenames or fields directly
            # Pick first dict value that has 'md' or 'content_list'
            for v in api_response.values():
                if isinstance(v, dict) and ("md" in v or "content_list" in v):
                    first = v
                    break
        if not first:
            first = api_response  # last resort: response IS the result

        md = first.get("md", "")
        content_list = first.get("content_list")

        if content_list and isinstance(content_list, list):
            parsed = parse_content_list(content_list, file_id)
            if md:
                parsed["content"] = md
        else:
            parsed = {
                "file_id": file_id,
                "content": md,
                "elements": [],
                "metadata": {"parser_mode": "md_only"},
            }

        if not parsed.get("content") and not parsed.get("elements"):
            raise RuntimeError("mineru-api returned empty content and no elements")

        parsed["success"] = True
        parsed.setdefault("metadata", {}).update({
            "backend": self.backend,
            "element_count": len(parsed.get("elements", [])),
            "mineru_version": api_response.get("version", ""),
        })
        return parsed

    async def parse(
        self,
        file_path: str,
        *,
        backend: Optional[str] = None,
        lang: Optional[str] = None,
        start_page: Optional[int] = None,
        end_page: Optional[int] = None,
    ) -> Dict[str, Any]:
        file_id = Path(file_path).name
        try:
            api_response = await self._call_file_parse(
                file_path,
                backend=backend,
                lang=lang,
                start_page=start_page,
                end_page=end_page,
            )
            return self._build_result(api_response, file_id)
        except Exception as exc:
            logger.exception("mineru parse failed: %s", file_path)
            return {"success": False, "error": str(exc), "content": "", "elements": [], "metadata": {}}
