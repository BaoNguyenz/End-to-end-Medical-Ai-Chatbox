"""
convert_pdf_to_md.py
High-performance, layout-aware PDF to Markdown converter for:
The Gale Encyclopedia of Medicine (3rd Edition).

Features:
- Supports multi-column and mixed 1-col / 2-col / 3-col layouts via pymupdf4llm.
- Mode 'sample': Quickly converts a specific range of pages (e.g. 45-55) for visual inspection.
- Mode 'entries': Automatically detects medical topics (Diseases, Drugs, Procedures)
  and saves individual .md files with clean metadata for RAG & GraphRAG.
- Mode 'chunks': Splits into batch Markdown files (e.g. 100 pages per file).
- Auto-cleans hyphenated line breaks (e.g. 'anti- \n inflammatory' -> 'anti-inflammatory').
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

import pymupdf as fitz  # PyMuPDF
import pymupdf4llm
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent

# Tự động nhận diện thư mục Data dù script đặt ở đâu (trong Data/ hoặc ở thư mục gốc)
if SCRIPT_DIR.name.lower() == "data":
    DATA_DIR = SCRIPT_DIR
else:
    DATA_DIR = SCRIPT_DIR / "Data"

# Tự động tìm file PDF trong thư mục Data
_pdf_candidates = list(DATA_DIR.glob("*.pdf"))
if _pdf_candidates:
    DEFAULT_PDF_PATH = _pdf_candidates[0]
else:
    DEFAULT_PDF_PATH = DATA_DIR / "The-Gale-Encyclopedia-of-Medicine-3rd-Edition-staibabussalamsula.pdf"

DEFAULT_OUTPUT_DIR = DATA_DIR / "markdown_output"


def clean_markdown_text(text: str) -> str:
    """
    Clean up common OCR / PDF extraction artifacts in Markdown:
    - Fix hyphenated word breaks at end of lines (e.g. 'medi-\ncine' -> 'medicine')
    - Remove standalone running headers/footers
    - Normalize multiple blank lines
    """
    if not text:
        return ""

    # Fix hyphenated words across line breaks (e.g. "cholesty-\nramine" -> "cholestyramine")
    text = re.sub(r"(\b\w+)-\n(\w+\b)", r"\1\2", text)

    # Remove repeated header artifacts like "GALE ENCYCLOPEDIA OF MEDICINE 3"
    text = re.sub(
        r"(?i)^.*?GALE ENCYCLOPEDIA OF MEDICINE.*?$\n?",
        "",
        text,
        flags=re.MULTILINE,
    )

    # Normalize 3+ consecutive newlines to 2 newlines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def convert_sample(
    doc: fitz.Document,
    start_page: int,
    end_page: int,
    output_path: Path,
) -> None:
    """
    Convert a specific page range into a single Markdown file for inspection.
    Page numbers are 1-indexed for user friendliness.
    """
    print(f"\n[Sample Mode] Converting pages {start_page} to {end_page}...")
    page_indices = list(range(start_page - 1, min(end_page, len(doc))))

    md_text = pymupdf4llm.to_markdown(
        doc,
        pages=page_indices,
        show_progress=True,
    )

    cleaned_md = clean_markdown_text(md_text)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(cleaned_md, encoding="utf-8")

    print(f" Saved sample Markdown to: {output_path}")
    print(f" Total characters: {len(cleaned_md):,}")


def convert_chunks(
    doc: fitz.Document,
    chunk_size: int,
    output_dir: Path,
    start_page: int = 1,
    end_page: Optional[int] = None,
) -> None:
    """
    Convert PDF in chunks of N pages (e.g. 100 pages per file).
    """
    total_pages = len(doc)
    actual_end = min(end_page or total_pages, total_pages)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[Chunk Mode] Converting pages {start_page} to {actual_end} in batches of {chunk_size} pages...")

    for chunk_start in range(start_page - 1, actual_end, chunk_size):
        chunk_end = min(chunk_start + chunk_size, actual_end)
        pages_to_process = list(range(chunk_start, chunk_end))

        chunk_filename = output_dir / f"pages_{chunk_start + 1:04d}_to_{chunk_end:04d}.md"

        if chunk_filename.exists():
            print(f"⏩ Skipping existing chunk: {chunk_filename.name}")
            continue

        print(f"\nProcessing pages {chunk_start + 1} to {chunk_end} ({len(pages_to_process)} pages)...")
        md_text = pymupdf4llm.to_markdown(
            doc,
            pages=pages_to_process,
            show_progress=False,
        )

        cleaned_md = clean_markdown_text(md_text)
        chunk_filename.write_text(cleaned_md, encoding="utf-8")
        print(f" Saved chunk: {chunk_filename.name}")


def convert_by_entries(
    doc: fitz.Document,
    output_dir: Path,
    start_page: int = 25,  # Skip front-matter by default
    end_page: Optional[int] = None,
) -> None:
    """
    Converts and splits the encyclopedia into individual topic/disease files.
    Ideal for Medical RAG and Knowledge Graph extraction.
    """
    total_pages = len(doc)
    actual_end = min(end_page or total_pages, total_pages)
    entries_dir = output_dir / "entries"
    entries_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[Entries Mode] Converting pages {start_page} to {actual_end} and splitting by medical topic...")

    topic_header_pattern = re.compile(r"^#\s+([A-Z0-9][A-Za-z0-9\s,\-\(\)\/]{2,80})$", re.MULTILINE)

    current_title = "Introduction"
    current_content: list[str] = []
    entry_count = 0

    batch_size = 50
    pbar = tqdm(total=actual_end - start_page + 1, desc="Converting & Parsing")

    for p_start in range(start_page - 1, actual_end, batch_size):
        p_end = min(p_start + batch_size, actual_end)
        pages = list(range(p_start, p_end))

        raw_md = pymupdf4llm.to_markdown(doc, pages=pages, show_progress=False)
        cleaned_md = clean_markdown_text(raw_md)

        lines = cleaned_md.splitlines()
        for line in lines:
            match = topic_header_pattern.match(line)
            if match:
                # Flush previous entry
                if current_content and current_title:
                    safe_title = re.sub(r'[\\/*?:"<>|]', "", current_title).strip()
                    safe_title = safe_title.replace(" ", "_")[:60]
                    if safe_title:
                        entry_file = entries_dir / f"{safe_title}.md"
                        entry_file.write_text("\n".join(current_content), encoding="utf-8")
                        entry_count += 1

                current_title = match.group(1).strip()
                current_content = [f"# {current_title}\n"]
            else:
                current_content.append(line)

        pbar.update(len(pages))

    # Flush final entry
    if current_content and current_title:
        safe_title = re.sub(r'[\\/*?:"<>|]', "", current_title).strip()
        safe_title = safe_title.replace(" ", "_")[:60]
        if safe_title:
            entry_file = entries_dir / f"{safe_title}.md"
            entry_file.write_text("\n".join(current_content), encoding="utf-8")
            entry_count += 1

    pbar.close()
    print(f"\n Finished! Successfully created {entry_count:,} medical topic files in: {entries_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Medical Encyclopedia PDF to Markdown")
    parser.add_argument("--pdf", type=str, default=str(DEFAULT_PDF_PATH), help="Path to input PDF file")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Output directory or file path")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["sample", "chunks", "entries", "all"],
        default="sample",
        help="Conversion mode: 'sample' (test a few pages), 'chunks' (batches of pages), 'entries' (split by disease/topic), 'all' (single md file)",
    )
    parser.add_argument("--start-page", type=int, default=1, help="Start page number (1-indexed)")
    parser.add_argument("--end-page", type=int, default=10, help="End page number (1-indexed)")
    parser.add_argument("--chunk-size", type=int, default=100, help="Number of pages per chunk in 'chunks' mode")

    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"❌ Error: PDF file not found at: {pdf_path}")
        sys.exit(1)

    print(f"📖 Opening PDF: {pdf_path.name}")
    t0 = time.time()
    doc = fitz.open(str(pdf_path))
    print(f"📊 Total Pages: {len(doc):,}")

    output_path = Path(args.output)

    if args.mode == "sample":
        out_file = output_path if output_path.suffix == ".md" else output_path / f"sample_pages_{args.start_page}_to_{args.end_page}.md"
        convert_sample(doc, args.start_page, args.end_page, out_file)
    elif args.mode == "chunks":
        convert_chunks(doc, args.chunk_size, output_path, args.start_page, args.end_page)
    elif args.mode == "entries":
        convert_by_entries(doc, output_path, args.start_page, args.end_page)
    elif args.mode == "all":
        out_file = output_path if output_path.suffix == ".md" else output_path / "full_medical_encyclopedia.md"
        convert_sample(doc, 1, len(doc), out_file)

    doc.close()
    elapsed = time.time() - t0
    print(f"\n Done in {elapsed:.2f}s!")


if __name__ == "__main__":
    main()