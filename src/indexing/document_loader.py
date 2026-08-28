"""
document_loader.py
Load medical encyclopedia markdown documents from Data/markdown_output/
into Document objects for indexing into Qdrant.

Medical data layout:
  Data/markdown_output/
    entries/       ← Individual medical topic .md files (one per topic)

Each file follows the Gale Encyclopedia structure:
  # Topic Title
  ### Definition
  ### Description
  ...
"""

from pathlib import Path
from typing import Optional

from src.models import Document, DocumentType


def load_all_documents(data_dir: Optional[Path] = None) -> list[Document]:
    """
    Load all .md medical encyclopedia entries from the data directory.

    The medical data uses a flat structure:
      data_dir/entries/*.md  → one file per medical topic/entry

    Args:
        data_dir: Path to the markdown_output directory.
                  Defaults to settings.data_dir (Data/markdown_output).

    Returns:
        List of Document objects with medical metadata.
    """
    if data_dir is None:
        from src.config import settings
        data_dir = settings.data_dir

    docs: list[Document] = []

    # Medical entries folder — all topics live here
    entries_folder = data_dir / "entries"

    if not entries_folder.exists():
        # Fallback: try loading directly from data_dir (all .md files)
        print(f"  [WARN] entries/ folder not found at {entries_folder}")
        print(f"  [INFO] Falling back to loading .md files directly from {data_dir}")
        md_files = sorted(data_dir.glob("*.md"))
    else:
        md_files = sorted(entries_folder.glob("*.md"))

    if not md_files:
        print(f"  [WARN] No .md files found in {entries_folder}")
        return docs

    for filepath in md_files:
        content = filepath.read_text(encoding="utf-8")

        # Extract the title from the first H1 line of the Markdown
        entry_title = filepath.stem.replace("_", " ").title()
        for line in content.splitlines():
            line = line.strip()
            if line.startswith("# "):
                entry_title = line.lstrip("# ").strip()
                break

        # Determine entry type from content keywords
        entry_type = _classify_entry_type(content)

        docs.append(Document(
            page_content=content,
            metadata={
                "source": str(filepath),
                "filename": filepath.name,
                "document_type": DocumentType.TECHNICAL.value,  # All medical entries are TECHNICAL
                "doc_id": filepath.stem,                         # e.g. "Asthma", "Appendectomy"
                "entry_title": entry_title,
                "entry_type": entry_type,                        # e.g. "Disease", "Drug", "Procedure"
            },
        ))

    # Summary
    print(f"[Loader] Loaded {len(docs)} medical encyclopedia entries from {entries_folder}")
    type_counts: dict[str, int] = {}
    for doc in docs:
        et = doc.metadata.get("entry_type", "Unknown")
        type_counts[et] = type_counts.get(et, 0) + 1
    for et, count in sorted(type_counts.items()):
        print(f"  {et}: {count} entries")

    return docs


def _classify_entry_type(content: str) -> str:
    """
    Heuristic classification of a medical encyclopedia entry type
    based on its section headers.

    Returns one of: "Disease", "Drug", "Procedure", "Test", "General"
    """
    content_lower = content.lower()

    # Disease/condition entries always have "Causes and symptoms"
    if "causes and symptoms" in content_lower or "causes & symptoms" in content_lower:
        return "Disease"

    # Drug entries always have "Recommended dosage"
    if "recommended dosage" in content_lower or "drug interactions" in content_lower:
        return "Drug"

    # Procedure entries have "Aftercare" or "Preparation"
    if "aftercare" in content_lower and "preparation" in content_lower:
        return "Procedure"

    # Diagnostic test entries have "Normal results" or "Abnormal results"
    if "normal results" in content_lower or "abnormal results" in content_lower:
        return "Test"

    return "General"


if __name__ == "__main__":
    documents = load_all_documents()
    for doc in documents[:10]:
        preview = doc.page_content[:100].replace("\n", " ")
        print(f"  [{doc.metadata['entry_type']}] {doc.doc_id}: {preview}...")
