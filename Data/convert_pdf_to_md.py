"""
convert_pdf_to_md.py
Bộ chuyển đổi PDF sang Markdown tối ưu cho:
The Gale Encyclopedia of Medicine (3rd Edition).
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

# Cấu hình UTF-8 cho Windows Terminal
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

SCRIPT_DIR = Path(__file__).resolve().parent

# Tự động nhận diện thư mục Data dù script đặt ở đâu
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

# Danh sách các tên đề mục con cần bỏ qua khi nhận diện Tiêu đề bệnh/thuốc chính
SECTION_SUBHEADINGS = {
    "definition", "purpose", "description", "causes and symptoms", "diagnosis",
    "treatment", "prognosis", "prevention", "key terms", "resources", "precautions",
    "preparation", "aftercare", "risks", "normal results", "abnormal results",
    "periodicals", "organizations", "books", "other", "inclusion criteria",
    "about the contributors", "how to use this book", "scope", "introduction",
    "advisors", "contributors", "alternative treatment", "demographics",
    "questions to ask your doctor", "significance", "gale encyclopedia of medicine 3"
}


def clean_markdown_text(text: str) -> str:
    """
    Làm sạch văn bản an toàn không làm mất nội dung:
    - Nối các từ bị ngắt dòng bằng dấu gạch nối (cholesty-\nramine -> cholestyramine)
    - Chuẩn hóa các dòng trống thừa
    """
    if not text:
        return ""

    # Nối từ bị ngắt gạch nối xuống dòng
    text = re.sub(r"(\b\w+)-\n(\w+\b)", r"\1\2", text)

    # Chuẩn hóa khoảng cách dòng
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def is_entry_header(line: str) -> Optional[str]:
    """
    Kiểm tra xem một dòng có phải là Tiêu đề mục từ Y khoa (Tên Bệnh/Thuốc/Thủ thuật) không.
    """
    stripped = line.strip()
    match = re.match(r"^#{1,2}\s+(.+)$", stripped)
    if not match:
        return None

    title = match.group(1).strip()
    # Loại bỏ các chữ cái phân chương đơn lẻ như '# A', '# B' hoặc đề mục quá ngắn
    if len(title) <= 1 or re.match(r"^[A-Z]$", title):
        return None

    # Loại bỏ các đề mục con chuẩn (Definition, Description, Key terms...)
    clean_title = re.sub(r"[^\w\s]", "", title).strip().lower()
    if clean_title in SECTION_SUBHEADINGS or title.lower() in SECTION_SUBHEADINGS:
        return None

    return title


def convert_by_entries(
    doc: fitz.Document,
    output_dir: Path,
    start_page: int = 30,  # Trang 30 bắt đầu mục từ chữ A (Abdominal ultrasound)
    end_page: Optional[int] = None,
    batch_size: int = 50,
) -> None:
    """
    Tự động nhận diện và tách từng mục từ (Bệnh / Thuốc) thành từng file Markdown độc lập.
    Ghi trực tiếp từng file ra đĩa ngay khi hoàn tất mục từ đó.
    """
    total_pages = len(doc)
    actual_end = min(end_page or total_pages, total_pages)
    entries_dir = output_dir / "entries"
    entries_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[Entries Mode] Đang xử lý từ trang {start_page} đến {actual_end} và tách theo từng mục từ...")

    current_title: Optional[str] = None
    current_content: list[str] = []
    entry_count = 0

    pbar = tqdm(total=actual_end - start_page + 1, desc="Đang chuyển đổi & tách bài")

    def save_current_entry() -> None:
        nonlocal entry_count
        if current_title and current_content:
            # Tạo tên file an toàn cho hệ điều hành
            safe_title = re.sub(r'[\\/*?:"<>|]', "", current_title).strip()
            safe_title = safe_title.replace(" ", "_")[:60]
            if safe_title:
                entry_file = entries_dir / f"{safe_title}.md"
                entry_file.write_text("\n".join(current_content), encoding="utf-8")
                entry_count += 1

    for p_start in range(start_page - 1, actual_end, batch_size):
        p_end = min(p_start + batch_size, actual_end)
        pages = list(range(p_start, p_end))

        raw_md = pymupdf4llm.to_markdown(doc, pages=pages, show_progress=False)
        cleaned_md = clean_markdown_text(raw_md)

        lines = cleaned_md.splitlines()
        for line in lines:
            detected_title = is_entry_header(line)
            if detected_title:
                # Ghi mục từ trước đó ra file
                save_current_entry()

                # Bắt đầu mục từ mới
                current_title = detected_title
                current_content = [f"# {current_title}\n"]
            else:
                if current_title:
                    current_content.append(line)

        pbar.update(len(pages))

    # Ghi mục từ cuối cùng
    save_current_entry()

    pbar.close()
    print(f"\n🎉 Hoàn tất! Đã tạo thành công {entry_count:,} file bệnh học tại: {entries_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Medical Encyclopedia PDF to Markdown")
    parser.add_argument("--pdf", type=str, default=str(DEFAULT_PDF_PATH), help="Đường dẫn file PDF")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Thư mục xuất kết quả")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["sample", "entries"],
        default="entries",
        help="Chế độ: 'entries' (tách theo từng bài bệnh), 'sample' (thử nghiệm vài trang)",
    )
    parser.add_argument("--start-page", type=int, default=30, help="Trang bắt đầu (mục từ bắt đầu từ trang 30)")
    parser.add_argument("--end-page", type=int, default=None, help="Trang kết thúc (None = hết sách)")

    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"❌ Lỗi: Không tìm thấy file PDF tại: {pdf_path}")
        sys.exit(1)

    print(f"📖 Đang mở file: {pdf_path.name}")
    t0 = time.time()
    doc = fitz.open(str(pdf_path))
    print(f"📊 Tổng số trang: {len(doc):,}")

    output_path = Path(args.output)
    if args.mode == "entries":
        convert_by_entries(doc, output_path, args.start_page, args.end_page)

    doc.close()
    elapsed = time.time() - t0
    print(f"\n⏱️ Thời gian thực thi: {elapsed:.2f}s!")


if __name__ == "__main__":
    main()
