import fitz  # pyMuPDF
import os
import re
from typing import Optional, List
import logging


def extract_text_from_folder(folderPath: str, max_files: int =1)-> Optional[List[str]]:

    if not os.path.exists(folderPath):
        return None
    
    if not os.path.isdir(folderPath):
        raise Exception(f"path is not directory: {folderPath}")
    
    try:
        pdf_files = [f for f in os.listdir(folderPath) if f.lower().endswith('pdf')]

        if not pdf_files:
            return []
        
        extracted_text = []

        for pdf_file in pdf_files[:max_files]:
            pdf_path = os.path.join(folderPath, pdf_file)
            try:
                pdf_text = extract_text_from_pdf(pdf_path)
                extracted_text.append(pdf_text)
            except Exception as e:
                logging.warning(f"Could not load extract from {pdf_path}: {e}")
        
        return extracted_text
    except Exception as e:
        logging.error(f"Error processing folder {folderPath}: {e}")

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Open a PDF and extract all text as a single string.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Extracted text from all pages
        
    Raises:
        FileNotFoundError: If PDF file doesn't exist
        Exception: If PDF cannot be opened or processed
    """
    try:
        doc = fitz.open(pdf_path)
        pages = []
        for page in doc:
            page_text = page.get_text()
            page_text = clean_extracted_text(page_text)
            if len(page_text)>10:
                pages.append(page_text)
        doc.close()

        if not pages:
            logging.warning(f"No extractable text found in PDF: {pdf_path}")

        return '\n'.join(pages)
    
    except FileNotFoundError:
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    except Exception as e:
        raise Exception(f"Error processing PDF {pdf_path}: {str(e)}")

def clean_extracted_text(text: str) -> str:
    #remove excessive white space
    text = re.sub(r'\s+', ' ', text)

    #remove weird characters
    text = re.sub(r'[^\w\s\.\,\!\?\-\(\)]', '', text)

    text = text.strip()

    return text


def extract_text_from_folder(folderpath: str, max_files: int=1) -> Optional[List[str]]:
    """
    Extract text from PDF files in a folder.
    
    Args:
        folderpath: Path to the folder containing PDF files
        max_files: Maximum number of PDF files to process (default: 1)
        
    Returns:
        List of extracted text strings from PDF files, or None if folder doesn't exist
        
    Raises:
        Exception: If folder cannot be accessed or PDFs cannot be processed
    """

    if not os.path.exists(folderpath):
        return None
    
    if not os.path.isdir(folderpath):
        raise Exception(f"Path is not directory: {folderpath}")
    
    try:
        pdf_files = [f for f in os.listdir(folderpath) if f.lower().endswith('pdf')]

        if not pdf_files:
            return []
        
        extracted_texts = []

        for pdf_file in pdf_files[:max_files]:
            pdf_path = os.path.join(folderpath, pdf_file)
            try:
                text = extract_text_from_pdf(pdf_path)
                extracted_texts.append(text)
            except Exception as e:
                logging.warning(f"Could not extract from {pdf_path}: {e}")
        return extracted_texts
    
    except Exception as e:
        logging.error(f"Error processing folder {folderpath}: {e}")
        return None
    

