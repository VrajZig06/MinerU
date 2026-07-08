from fastapi import (
    FastAPI,
    Request,
    status as http_status,
    File,
    UploadFile,
    HTTPException,
)
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import tempfile
import shutil

load_dotenv()

app = FastAPI(title="MinerU File Parsing")

MINERU_BACKEND: str = "pipeline"
EXPECTED_SECRET = os.getenv("EXPECTED_SECRET")


@app.middleware("http")
async def verify_header(request: Request, call_next):
    secret = request.headers.get("X-API-KEY")

    if secret != EXPECTED_SECRET:
        return JSONResponse(
            status_code=401,
            content={"success": False, "message": "Unauthorized. Invalid API Key."},
        )

    response = await call_next(request)
    return response


def parse_with_mineru(
    pdf_bytes: bytes,
    pdf_stem: str,
    output_dir: str,
    backend: str,
) -> tuple[str, int, list[str]]:
    """
    Run MinerU on the given PDF bytes and return (markdown_text, page_count, image_urls).

    The raw bytes from the uploaded file are passed directly to MinerU —
    no intermediate file is written to disk for the input.

    MinerU 3.x writes output to:
        <output_dir>/<pdf_stem>/auto/<pdf_stem>.md
        <output_dir>/<pdf_stem>/auto/<pdf_stem>_content_list.json
        <output_dir>/<pdf_stem>/auto/images/   ← extracted images (copied to static/)

    Args:
        pdf_bytes:  Raw bytes of the uploaded PDF.
        pdf_stem:   Filename stem used to name MinerU's output files.
        output_dir: Directory where MinerU will write its output.
        backend:    "pipeline" (CPU) or "vlm-transformers" (GPU).

    Returns:
        A tuple of (markdown_str, page_count, image_urls).
        image_urls is a list of /static/... HTTP paths for each extracted image.
    """
    import json as _json

    from mineru.cli.common import do_parse

    parse_method = "auto"

    do_parse(
        output_dir=output_dir,
        pdf_file_names=[pdf_stem],
        pdf_bytes_list=[pdf_bytes],
        p_lang_list=[""],
        backend=backend,
        parse_method=parse_method,
        f_draw_layout_bbox=False,
        f_draw_span_bbox=False,
        f_dump_orig_pdf=False,
        f_dump_model_output=False,
        f_dump_middle_json=False,
        f_dump_content_list=True,
        f_dump_md=True,
    )

    # MinerU 3.x output layout: <output_dir>/<stem>/auto/<stem.md>
    md_path = os.path.join(output_dir, pdf_stem, parse_method, f"{pdf_stem}.md")
    if not os.path.exists(md_path):
        raise FileNotFoundError(f"MinerU did not produce expected output at: {md_path}")

    with open(md_path, "r", encoding="utf-8") as f:
        markdown_text = f.read()

    # Estimate page count from MinerU's content list JSON
    json_path = os.path.join(
        output_dir, pdf_stem, parse_method, f"{pdf_stem}_content_list.json"
    )
    page_count = 0
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as jf:
            content_list = _json.load(jf)
        pages_seen = {block.get("page_idx", 0) for block in content_list}
        page_count = len(pages_seen)

    # Persist extracted images to static/uploads/<pdf_stem>/images/
    # and rewrite relative image references in the Markdown to HTTP URLs.
    temp_image_dir = os.path.join(output_dir, pdf_stem, parse_method, "images")
    persistent_image_dir = os.path.join("static", "uploads", pdf_stem, "images")
    image_urls: list[str] = []

    if os.path.isdir(temp_image_dir):
        os.makedirs(persistent_image_dir, exist_ok=True)
        for img_file in sorted(os.listdir(temp_image_dir)):
            src = os.path.join(temp_image_dir, img_file)
            dst = os.path.join(persistent_image_dir, img_file)
            shutil.copy2(src, dst)
            url = f"/static/uploads/{pdf_stem}/images/{img_file}"
            image_urls.append(url)

        # Rewrite ![...](images/foo.png) → ![...](/static/uploads/<stem>/images/foo.png)
        markdown_text = markdown_text.replace(
            "](images/",
            f"](/static/uploads/{pdf_stem}/images/",
        )

    return markdown_text, page_count, image_urls


@app.post("/parse-mineru-file")
async def parse_mineru_file(
    file: UploadFile = File(...),
):
    """
    Accepts a PDF document upload, reads the file, uses MinerU to extract
    structured markdown text locally (no API key required), reconstructs
    textbook layout artifacts, and chunks it using RecursiveCharacterTextSplitter.
    Authentication is required.
    """
    try:
        # Validate that file is a PDF
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=http_status.HTTP_400_BAD_REQUEST,
                detail="Only PDF files are allowed",
            )

        # Create tmp directory if it doesn't exist
        os.makedirs("parsed_files", exist_ok=True)

        # Read the uploaded file bytes directly into memory
        file_bytes = await file.read()
        pdf_stem = os.path.splitext(file.filename)[0]

        # Create a dedicated output directory for MinerU results
        mineru_output_dir = tempfile.mkdtemp(dir="parsed_files", prefix="mineru_")

        # Parse the document using MinerU — bytes are passed directly, no temp file written
        raw_markdown, page_count, image_urls = parse_with_mineru(
            pdf_bytes=file_bytes,
            pdf_stem=pdf_stem,
            output_dir=mineru_output_dir,
            backend=MINERU_BACKEND,
        )

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separators=["\n\n", "\n", " ", ""],
        )
        raw_chunks = text_splitter.split_text(raw_markdown)

        chunks = []
        for idx, chunk in enumerate(raw_chunks):
            chunks.append(
                {
                    "id": f"chunk-{idx}",
                    "text": chunk,
                    "metadata": {"chunk_index": idx},
                }
            )

        return JSONResponse(
            status_code=http_status.HTTP_200_OK,
            content={
                "page_count": page_count,
                "image_urls": image_urls,
                "chunks": chunks,
            },
        )
    except HTTPException as e:
        raise HTTPException(status_code=http_status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )
