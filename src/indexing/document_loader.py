"""
document_loader.py
Load markdown documents from the data/ directory into Document objects.
"""

from pathlib import Path
from typing import Optional

from src.models import Document, DocumentType


# Mapping folder name -> document type
_FOLDER_TYPE_MAP: dict[str, str] = {
    "technical_docs": DocumentType.TECHNICAL.value,
    "policy_docs": DocumentType.POLICY.value,
    "product_catalog": DocumentType.PRODUCT.value,
}


def load_all_documents(data_dir: Optional[Path] = None) -> list[Document]:
    """
    Load all .md documents from data/ subdirectories.

    Args:
        data_dir: Path to the data directory. Defaults to project data/ dir.

    Returns:
        List of Document objects with metadata.
    """
    if data_dir is None:
        from src.config import settings
        data_dir = settings.data_dir

    docs: list[Document] = []

    for folder_name, doc_type in _FOLDER_TYPE_MAP.items():
        folder_path = data_dir / folder_name

        if not folder_path.exists():
            print(f"  [WARN] Folder not found: {folder_path}")
            continue

        for filepath in sorted(folder_path.glob("*.md")):
            content = filepath.read_text(encoding="utf-8")
            docs.append(Document(
                page_content=content,
                metadata={
                    "source": str(filepath),
                    "filename": filepath.name,
                    "document_type": doc_type,
                    "doc_id": filepath.stem,
                },
            ))

    # Summary
    print(f"[Loader] Loaded {len(docs)} documents:")
    for doc_type in _FOLDER_TYPE_MAP.values():
        count = sum(1 for d in docs if d.document_type == doc_type)
        print(f"  {doc_type}: {count} documents")

    return docs


if __name__ == "__main__":
    documents = load_all_documents()
    for doc in documents:
        preview = doc.page_content[:100].replace("\n", " ")
        print(f"  [{doc.doc_id}] {preview}...")
