"""Extract supplied lecture text locally; original PDFs remain the source of truth."""
from pathlib import Path
from pypdf import PdfReader

root = Path(__file__).resolve().parents[1]
out = root / "tmp" / "notes"
out.mkdir(parents=True, exist_ok=True)
for source in sorted((root / "CUDA_Notes").rglob("*.pdf")):
    reader = PdfReader(source)
    text = "\n\n".join(f"PAGE {i + 1}\n{page.extract_text() or ''}" for i, page in enumerate(reader.pages))
    (out / (source.stem + ".txt")).write_text(text, encoding="utf-8")
    print(f"{source.name}: {len(reader.pages)} pages, {len(text)} characters")
