# Document ingestion endpoint
from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, Form
from fastapi.responses import JSONResponse

from src.services.ingestion import ingest_file

router = APIRouter(prefix="/ingest", tags=["ingestion"])

SUPPORTED_SUFFIXES = {".md", ".txt", ".html", ".htm", ".pdf"}


@router.post("", summary="Ingest a documentation file")
async def ingest(
    file: UploadFile,
    source_name: str | None = Form(default=None),
) -> JSONResponse:
    """Upload and ingest a Markdown, text, HTML, or PDF file into the vector index."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{suffix}'. Supported: {sorted(SUPPORTED_SUFFIXES)}",
        )

    # Write upload to a temp file so ingestion can memory-map PDFs etc.
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = Path(tmp.name)

    try:
        result = await ingest_file(
            tmp_path,
            source_name=source_name or file.filename,
            source_file=file.filename,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    finally:
        tmp_path.unlink(missing_ok=True)

    return JSONResponse({"status": "success", **result})
