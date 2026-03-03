"""
Multimodal File Parser

    Supports PDF, DOCX, TXT/MD, CSV/TSV/XLSX, and Image OCR extraction (EasyOCR).
"""
import io
import os
import sys
import logging
import subprocess
from typing import Optional

from PIL import Image

from app.infra.hardware import HardwareProbe

logger = logging.getLogger(__name__)


class FileParser:
    def __init__(self, ocr_engine: Optional[str] = None):
        self.ocr_engine = (ocr_engine or os.getenv("OCR_ENGINE", "easyocr")).lower()
        self._easy = None

    def parse(self, filename: str, file_bytes: bytes) -> str:
        if not filename:
            raise ValueError("Filename is required for file type detection.")
        if not file_bytes:
            raise ValueError("File content is empty.")

        ext = os.path.splitext(filename)[1].lower()
        if ext == ".pdf":
            return self._parse_pdf(file_bytes)
        if ext == ".docx":
            return self._parse_docx(file_bytes)
        if ext in [".csv", ".tsv"]:
            return self._parse_tabular(file_bytes, delimiter="\t" if ext == ".tsv" else ",")
        if ext == ".xlsx":
            return self._parse_xlsx(file_bytes)
        if ext in [".txt", ".md"]:
            return self._parse_text(file_bytes)
        if ext in [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"]:
            return self.parse_image_text(file_bytes)

        raise ValueError(f"Unsupported file type: {ext}")

    def _parse_text(self, file_bytes: bytes) -> str:
        try:
            return file_bytes.decode("utf-8")
        except Exception:
            return file_bytes.decode("latin-1", errors="ignore")

    def _parse_pdf(self, file_bytes: bytes) -> str:
        try:
            import fitz  # PyMuPDF
        except Exception as e:
            raise RuntimeError(f"PyMuPDF (fitz) not installed or failed to import: {e}")

        doc = fitz.open(stream=file_bytes, filetype="pdf")
        text = " ".join([page.get_text() for page in doc]).strip()

        # OCR fallback for scanned PDFs or image-only pages
        min_chars = int(os.getenv("PDF_OCR_MIN_CHARS", "30"))
        ocr_fallback = os.getenv("PDF_OCR_FALLBACK", "true").lower() == "true"
        if text and len(text) >= min_chars:
            return text
        if not ocr_fallback:
            return text

        try:
            ocr_texts = []
            for page in doc:
                pix = page.get_pixmap(dpi=200)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                ocr_text = self.parse_image_text(self._image_to_bytes(img))
                if ocr_text:
                    ocr_texts.append(ocr_text)
            return "\n".join(ocr_texts).strip()
        except Exception as e:
            logger.warning(f"[PDF] OCR fallback failed, returning extracted text only: {e}")
            return text

    def _parse_docx(self, file_bytes: bytes) -> str:
        try:
            from docx import Document
        except Exception as e:
            raise RuntimeError(f"python-docx not installed or failed to import: {e}")

        doc = Document(io.BytesIO(file_bytes))
        return "\n".join([p.text for p in doc.paragraphs])

    def _parse_tabular(self, file_bytes: bytes, delimiter: str = ",") -> str:
        try:
            import pandas as pd
        except Exception as e:
            raise RuntimeError(f"pandas not installed or failed to import: {e}")

        max_rows = int(os.getenv("TABULAR_MAX_ROWS", "200"))
        buf = io.BytesIO(file_bytes)
        try:
            df = pd.read_csv(buf, sep=delimiter)
        except Exception:
            # Fallback to latin-1 if encoding is off
            buf.seek(0)
            df = pd.read_csv(buf, sep=delimiter, encoding="latin-1")
        preview = df.head(max_rows)
        return preview.to_csv(index=False)

    def _parse_xlsx(self, file_bytes: bytes) -> str:
        try:
            import pandas as pd
        except Exception as e:
            raise RuntimeError(f"pandas not installed or failed to import: {e}")

        max_rows = int(os.getenv("TABULAR_MAX_ROWS", "200"))
        buf = io.BytesIO(file_bytes)
        sheets = pd.read_excel(buf, sheet_name=None)
        parts = []
        for sheet_name, df in sheets.items():
            preview = df.head(max_rows)
            parts.append(f"[SHEET: {sheet_name}]\n{preview.to_csv(index=False)}")
        return "\n\n".join(parts).strip()

    def _get_easyocr(self):
        if self.ocr_engine != "easyocr":
            raise RuntimeError(f"OCR engine '{self.ocr_engine}' not supported. Use OCR_ENGINE=easyocr.")
        if self._easy is None:
            try:
                import easyocr
            except Exception as e:
                raise RuntimeError(f"EasyOCR not installed or failed to import: {e}")
            lang = os.getenv("OCR_LANG", "en")
            profile = HardwareProbe.get_profile()
            use_gpu = profile.get("primary_device") == "cuda"
            os.environ.setdefault("EASYOCR_DOWNLOAD_VERBOSE", "0")
            self._easy = easyocr.Reader([lang], gpu=use_gpu, verbose=False)
        return self._easy

    def parse_image_text(self, file_bytes: bytes) -> str:
        if not file_bytes:
            return ""

        try:
            import numpy as np
        except Exception as e:
            raise RuntimeError(f"numpy not installed or failed to import: {e}")

        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        np_img = np.array(image)

        ocr = self._get_easyocr()
        result = ocr.readtext(np_img)

        if not result:
            return ""

        texts = []
        for _, text, _ in result:
            if text:
                texts.append(text)

        return "\n".join(texts).strip()

    @staticmethod
    def _image_to_bytes(image: Image.Image) -> bytes:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue()
