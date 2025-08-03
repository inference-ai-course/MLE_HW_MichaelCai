"""
arxiv Paper Abstract Scraper

Query any subcategory to fetch the latest 200 papers
Scrab the /abs/ page and use Trafilatura to clean the content
"""

import requests
import xml.etree.ElementTree as ET
from urllib.parse import urlparse, urljoin
import time
from typing import List, Dict, Optional
import trafilatura
from dataclasses import dataclass
import logging
from time import sleep
import pytesseract
import io
import json
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from PIL import Image


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
    ocr_abstract: Optional[str] = None


    def to_dict(self) -> Dict:
        ###Convert Paper to dictionary for JSON export###
        return {
            'url': self.url,
            'title': self.title,
            'abstract': self.abstract,
            'authors': self.authors,
            'date': self.published
        }
    
class ArxivScraper:
    ###arXiv paper abstract scraper

    def __init__(self):
        self.base_url = "http://export.arxiv.org/api/query"
        self.session = requests.session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Max OS X 10_15_7) AppleWebKit/537.36"
        })
        self.driver = None
        self._setup_webdriver()

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
    
    def scrap_abstract_page(self, arxiv_id:str) -> Optional[str]:
        """
        scrap the arXiv abstract page and clean content with Trafilatura
        
        """
        abs_url = f"https://arxiv.org/abs/{arxiv_id}"

        try:
            response = self.session.get(abs_url, timeout=30)
            response.raise_for_status()

            #use Trafilatura to extract and clean content
            if response.text and response.text.strip():
                cleaned_content = trafilatura.extract(response.text)
                if cleaned_content:
                    return cleaned_content
                else:
                    return "Content extraction failed"
            else: 
                return "No response content"
            
        
        except requests.RequestException as e:
            print(f"Error scraping {abs_url}: {e}")
            return None
    
    def _setup_webdriver(self):
        """set up Chrome webdriver for screenshot"""
        try:
            chrome_options = Options()
            chrome_options.add_argument('--headless')
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('disable-dev-shm-usage')
            chrome_options.add_argument('--window-size=1920,1080')
            self.driver = webdriver.Chrome(options= chrome_options)
        except Exception as e:
            logging.warning(f("Warning: could not setup Webdriver: {e}"))
            self.driver = None
    
    def take_screenshot(self, url:str) -> Optional[bytes]:
        if not self.driver:
            return None
        
        try:
            self.driver.get(url)
            #Wait page to load
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )

            #Take screenshot
            screenshot = self.driver.get_screenshot_as_png()

            return screenshot
        
        except Exception as e:
            logging.error(f"Error taking screenshot of {url}: {e}")
            return None
    
    def extract_text_from_screenshot(self, screenshot_bytes:bytes) ->Optional[str]:
        """Use Tesseract OCR to extract text from screenshot"""

        try:
            image = Image.open(io.BytesIO(screenshot_bytes))

            #user Tesseract to extract text
            extracted_text = pytesseract.image_to_string(image)

            return extracted_text.strip()
        
        except Exception as e:
            logging.error(f"Error extracting text with OCR: {e}")
            return None
    
    def scrap_with_ocr(self, arxiv_id:str) -> Optional[str]:
        """scrap abstract page using screenshots and OCR"""
        abs_url = f"https://arxiv.org/abs/{arxiv_id}"

        #take screenshot
        screenshot = self.take_screenshot(abs_url)
        if not screenshot:
            return None
        
        #Extract screenshot to ext
        ocr_text = self.extract_text_from_screenshot(screenshot)

        if not ocr_text:
            return None
        
        lines = ocr_text.split('\n')
        abstract_start = -1

        for i, line in enumerate(lines):
            if 'abstract' in line.lower() and len(line.strip())<20:
                abstract_start = i+1
                break

        if abstract_start>=0:
            abstract_lines=[]
            for line in lines[abstract_start:]:
                line = line.strip()
                if not line:
                    continue
                if any(header in line.lower() for header in ['subjects:', 'cite as:', 'submission']):
                    break
                abstract_lines.append(line)
            
            return ' '.join(abstract_lines) if abstract_lines else ocr_text
        
        return ocr_text
    
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
        if self.driver:
            self.driver.quit()
    
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

            #scrap using OCR for additonal content
            if paper.id:
                logging.info(f"  Taking screenshot and running OCR...")
                ocr_abstract = self.scrap_with_ocr(paper.id)
                if ocr_abstract:
                    paper.abstract = ocr_abstract
            
            papers.append(paper)

            time.sleep(0.1)
        
        return papers

def main():
    
    scrapper = ArxivScraper()

    try:

        category = "cs.CL"
        max_papers = 10
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
            if paper.ocr_abstract:
                print(f"OCR Abstract: {paper.ocr_abstract[:200]}...")
            print(f"URL: {paper.url}")
        
    finally:
        scrapper.clean_up()
       

if __name__=="__main__":
    main()