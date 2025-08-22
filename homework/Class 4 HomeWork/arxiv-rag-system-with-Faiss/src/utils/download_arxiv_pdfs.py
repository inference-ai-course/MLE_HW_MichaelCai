"""
Download 50 PDFs from category cs.CL

"""

import requests
import xml.etree.ElementTree as ET
import time
from typing import List, Dict, Optional
from dataclasses import dataclass
import logging
from time import sleep
import os

    
class ArxivScraper:
    ###arXiv paper abstract scraper

    def __init__(self):
        self.base_url = "http://export.arxiv.org/api/query"
        self.session = requests.Session()
        
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
            print(f"Error parsing XML response: {e}")
        
        return papers
    
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

                logging.warning(f"Attempt {attempt+1} failed: {e}")
                sleep(2 ** attempt)
    
    
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
    
   
            

    def clean_up(self):
        # No cleanup needed for PDF processing
        pass
    
    def scrape_papers(self, category: str, max_results: int = 200):

        logging.info(f"Querying arXiv for category '{category}' (max {max_results} papers)...")

        paper_data = self.query_arxiv(category, max_results)

        for i, data in enumerate(paper_data, 1):
            
            document_id = data.get('id')
         
            if data.get('id'):
                print(f"id is {document_id}")
                pdf_content = self.download_pdf(document_id)
                pdf_filename = f"{document_id}.pdf"
                #save to file

                if pdf_content:
                    pdf_filename = os.path.join("../../data", pdf_filename)
                    with open(pdf_filename, 'wb') as f:
                        f.write(pdf_content)
                else:
                    logging.error(f"Failed to download PDF for {document_id}")



def main():
    
    scrapper = ArxivScraper()

    try:

        category = "cs.CL"
        max_papers = 50
        scrapper.scrape_papers(category, max_results=max_papers)

        
    finally:
        scrapper.clean_up()
       

if __name__=="__main__":
    main()