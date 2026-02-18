import fitz  # PyMuPDF
import docx
import os

def load_pdf(file_path: str) -> str:
    """
    Extracts text from a PDF file using PyMuPDF (fitz).
    
    PyMuPDF is chosen for its high speed and accurate text extraction capabilities compared to PyPDF2.
    
    Args:
        file_path (str): Absolute path to the .pdf file.
        
    Returns:
        str: The extracted text content joined from all pages.
    """
    text = ""
    try:
        with fitz.open(file_path) as doc:
            for page in doc:
                text += page.get_text()
    except Exception as e:
        print(f"Error reading PDF {file_path}: {e}")
    return text

def load_docx(file_path: str) -> str:
    """
    Extracts text from a DOCX file using python-docx.
    
    Iterates through all paragraphs in the document.
    
    Args:
        file_path (str): Absolute path to the .docx file.
        
    Returns:
        str: The extracted text content, with paragraphs separated by newlines.
    """
    text = ""
    try:
        doc = docx.Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        print(f"Error reading DOCX {file_path}: {e}")
    return text

def load_text(file_path: str) -> str:
    """
    Extracts text from a plain TXT file.
    
    Args:
        file_path (str): Absolute path to the .txt file.
        
    Returns:
        str: The raw text content.
    """
    text = ""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    except Exception as e:
        print(f"Error reading TXT {file_path}: {e}")
    return text

def load_document(file_path: str) -> str:
    """
    Dispatcher function to load document content based on file extension.
    
    Supports: .pdf, .docx, .txt
    
    Args:
        file_path (str): Absolute path to the file.
        
    Returns:
        str: Extracted text content.
        
    Raises:
        ValueError: If the file extension is not supported.
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return load_pdf(file_path)
    elif ext == ".docx":
        return load_docx(file_path)
    elif ext == ".txt":
        return load_text(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
