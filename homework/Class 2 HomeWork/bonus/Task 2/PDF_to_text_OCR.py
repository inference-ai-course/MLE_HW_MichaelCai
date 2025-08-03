"""
arxiv Paper Abstract Scraper

Query any subcategory to fetch the latest 200 papers
Scrab the /abs/ page and use Trafilatura to clean the content
"""

import requests
import xml.etree.ElementTree as ET
import time
from typing import List, Dict, Optional
from dataclasses import dataclass
import logging
from time import sleep
import pytesseract
import json
from PIL import Image
from pdf2image import convert_from_bytes
import os


@dataclass
class Paper:
    """Data class to represent arXiv paper"""
    id: str
    title: str
    authors: List[str]
    abstract: str
    categories: List[str]
    published: str
    url: str
    ocr_full_text: Optional[str] = None


    def to_dict(self) -> Dict:
        ###Convert Paper to dictionary for JSON export###
        return {
            'url': self.url,
            'title': self.title,
            'abstract': self.abstract,
            'authors': self.authors,
            'date': self.published,
            'ocr_text_path': self.ocr_full_text
        }
    
class ArxivScraper:
    ###arXiv paper abstract scraper

    def __init__(self):
        self.base_url = "http://export.arxiv.org/api/query"
        self.session = requests.session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Max OS X 10_15_7) AppleWebKit/537.36"
        })

    def query_arxiv(self, category: str, max_results: int = 20) -> List[Dict]:
        """
        Query arXiv API for papers in a specific category

        Args:
            category: arXiv category(e.g., 'cs.CL', 'cs.AI', 'math.CO')
            max_results: Maximum number of papers to fetch (default: 200)
        
        Return:
            List of paper metadata dictionaries
        """

        params = {
            'search_query': f'cat:{category}',
            'start': 0,
            'max_results': max_results,
            'sortBy': 'submittedDate',
            'sortOrder': 'descending'
        }

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.session.get(
                    self.base_url, params = params, timeout=30)
                response.raise_for_status()
                return self._parse_arxiv_response(response.text)

            except requests.RequestException as e:
                if attempt == max_retries -1:
                    logging.error(f"failed after {max_retries}")
                    return []

                logging.warning(f"Attemp {attempt+1} failed: {e}")
                sleep(2 ** attempt)
    
    def _parse_arxiv_response(self, xml_content:str) -> List[Dict]:

        papers=[]

        try:
            root = ET.fromstring(xml_content)  

            #define namespace
            namespaces = {
                'atom': 'http://www.w3.org/2005/Atom',
                'arxiv': 'http://arxiv.org/schemas/atom'
            }   

            for entry in root.findall('atom:entry', namespaces):
                paper_data = {}

                #extract ID
                id_elem = entry.find('atom:id', namespaces)
                if id_elem is not None:
                    paper_data['id'] = id_elem.text.split('/')[-1]  
                
                #extract URL
                for link in entry.findall('atom.url', namespaces):
                    if link.get('title') == 'pdf':
                        paper_data['url'] = link.get('href')
                        break
                    else:
                        #fall back to abstract URL
                        if 'id' in paper_data:
                            paper_data['url'] = f"https://arxiv.org/abs/{paper_data['id']}"
                

                #extract the title
                title_elem = entry.find('atom:title', namespaces)
                if title_elem is not None:
                    paper_data['title'] = title_elem.text.strip()
                
                #extract abstract
                summary_elem = entry.find('atom:summary', namespaces)
                if summary_elem is not None:
                    paper_data['abstract'] = summary_elem.text.strip()

                #Extract authors
                authors=[]
                for author in entry.find('atom:author', namespaces):
                    name_elem = author.find('atom:name', namespaces)
                    if name_elem is not None:
                        authors.append(name_elem.text.strip())
                paper_data['authors'] = authors

                

                #extract categories
                # categories = []
                # for category in entry.find('atom.category', namespaces):
                #     term = categor.get('term')   
                #     if term and term.strip():
                #         categories.append(term)
                
                #extract published date
                published_elem = entry.find('atom:published', namespaces)
                if published_elem is not None:
                    paper_data['published'] = published_elem.text.strip()
                
                papers.append(paper_data)

        except ET.ParseError as e:
            print(f"Error passing XML response: {e}")
        
        return papers
    
    
    def download_pdf(self, arxiv_id: str) -> Optional[bytes]:
        """Download PDF file from arXiv"""
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        
        try:
            response = self.session.get(pdf_url, timeout=60)
            response.raise_for_status()
            return response.content
        except requests.RequestException as e:
            logging.error(f"Error downloading PDF {pdf_url}: {e}")
            return None
    
    def pdf_to_images(self, pdf_bytes: bytes) -> List[Image.Image]:
        """Convert PDF bytes to list of PIL Images"""
        try:
            # Use higher DPI for better OCR quality
            images = convert_from_bytes(pdf_bytes, dpi=600, fmt='png')
            return images
        except Exception as e:
            logging.error(f"Error converting PDF to images: {e}")
            return []
    
    def extract_text_from_pdf_with_layout(self, pdf_bytes: bytes) -> Optional[str]:
        """Extract text from PDF using OCR with layout preservation"""
        images = self.pdf_to_images(pdf_bytes)
        if not images:
            return None
        
        full_text = []
        
        for page_num, image in enumerate(images, 1):
            try:
                # Use PSM 3 for fully automatic page segmentation with better layout preservation
                custom_config = r'--oem 3 --psm 3 -c preserve_interword_spaces=1 -c textord_heavy_nr=1 -c textord_min_linesize=2.5'
                page_text = pytesseract.image_to_string(image, config=custom_config)
                
                if page_text.strip():
                    full_text.append(f"--- Page {page_num} ---\n{page_text.strip()}")
                    
            except Exception as e:
                logging.error(f"Error extracting text from page {page_num}: {e}")
                continue
        
        return "\n\n".join(full_text) if full_text else None
    
    def save_to_json(self, papers: List[Paper], filename:str="arxiv_clean.json", max_size_mb:float=1.0):

        data = [paper.to_dict() for paper in papers]

        #Convert to JSON string to check size
        json_str = json.dumps(data, indent=2, ensure_ascii=False)
        size_mb = len(json_str.encode('utf_8')) / (1024*1024)

        if size_mb > max_size_mb:
            print(f"Warning: JSON file size ({size_mb:.2f}MB) exceeds limit ({max_size_mb}MB)")
            while size_mb > max_size_mb and data:
                data.pop()
                json_str = json.dumps(data, indent=2, ensure_ascii=False)
                size_mb = len(json_str.encode('utf-8')) / (1024 * 1024)
            print(f"Truncated to {len(data)} papers to fit size limit")

        
        #save to file
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(json_str)
        
        print(f"Saved {len(data)} papers to {filename} ({size_mb:.2f}MB)")
            

    def clean_up(self):
        # No cleanup needed for PDF processing
        pass
    
    def scrape_papers(self, category: str, max_results: int = 200) -> List[Paper]:

        logging.info(f"Querying arXiv for category '{category}' (max {max_results} papers)...")

        paper_data = self.query_arxiv(category, max_results)

        if not paper_data:
            logging.warning("No paper found")
            return []
        
        print(f"Found {len(paper_data)} papers. Scraping abstracts...")

        papers = []
        for i, data in enumerate(paper_data, 1):
            print(f"Processing paper {i}/{len(paper_data)}: {data.get('id', 'Unknown ID')}")

            paper = Paper(
                id=data.get('id', ''),
                title=data.get('title', ''),
                authors=data.get('authors', []),
                abstract=data.get('abstract', ''),
                categories=data.get('categories', []),
                published=data.get('published', ''),
                url=data.get('url', '')
            )

            # Download PDF and extract full text using OCR
            if paper.id:
                logging.info(f"  Downloading PDF and running OCR...")
                pdf_bytes = self.download_pdf(paper.id)
                if pdf_bytes:
                    # Create output directory
                    os.makedirs("pdf_ocr", exist_ok=True)
                    
                    # Extract full text with layout preservation and save as text file
                    ocr_full_text = self.extract_text_from_pdf_with_layout(pdf_bytes)
                    if ocr_full_text:
                        # Save OCR text to file
                        text_filename = f"{paper.id}_ocr.txt"
                        text_path = os.path.join("pdf_ocr", text_filename)
                        
                        with open(text_path, 'w', encoding='utf-8') as f:
                            f.write(ocr_full_text)
                        
                        paper.ocr_full_text = text_path  # Store path instead of content
            
            papers.append(paper)

            time.sleep(0.1)
        
        return papers

def main():
    
    scrapper = ArxivScraper()

    try:

        category = "cs.CL"
        max_papers = 1
        papers = scrapper.scrape_papers(category, max_results=max_papers)

        print(f"\nScraping completed! Found {len(papers)} papers.")

         # Save to JSON
        scrapper.save_to_json(papers, "arxiv_clean_tesseract.json")

        #Display a few papers
        for i, paper in enumerate(papers[:3], 1):
            print(f"\n--- Paper {i} ---")
            print(f"ID: {paper.id}")
            print(f"Title: {paper.title}")
            print(f"Authors: {', '.join(paper.authors)}")
            print(f"Categories: {', '.join(paper.categories)}")
            print(f"Published: {paper.published}")
            print(f"Abstract: {paper.abstract[:200]}...")
            if paper.ocr_full_text:
                print(f"OCR Text Path: {paper.ocr_full_text}")
            print(f"URL: {paper.url}")
        
    finally:
        scrapper.clean_up()
       

if __name__=="__main__":
    main()