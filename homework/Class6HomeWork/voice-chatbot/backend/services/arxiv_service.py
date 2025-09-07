import requests
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class ArxivPaper:
    title: str
    authors: List[str]
    abstract: str
    pdf_url: str
    published: str
    arxiv_id: str

class ArxivService:
    """Service for searching and retrieving arXiv papers"""
    
    BASE_URL = "http://export.arxiv.org/api/query"
    
    def __init__(self):
        self.session = requests.Session()
        
    def search_papers(self, query: str, max_results: int = 5) -> List[ArxivPaper]:
        """
        Search arXiv for papers matching the query
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            
        Returns:
            List of ArxivPaper objects
        """
        try:
            params = {
                'search_query': f'all:{query}',
                'start': 0,
                'max_results': max_results,
                'sortBy': 'relevance',
                'sortOrder': 'descending'
            }
            
            response = self.session.get(self.BASE_URL, params=params)
            response.raise_for_status()
            
            papers = self._parse_response(response.text)
            logger.info(f"Found {len(papers)} papers for query: {query}")
            return papers
            
        except requests.RequestException as e:
            logger.error(f"Error searching arXiv: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error in arXiv search: {e}")
            return []
    
    def _parse_response(self, xml_response: str) -> List[ArxivPaper]:
        """Parse arXiv API XML response into ArxivPaper objects"""
        papers = []
        
        try:
            root = ET.fromstring(xml_response)
            
            # Define namespaces
            namespaces = {
                'atom': 'http://www.w3.org/2005/Atom',
                'arxiv': 'http://arxiv.org/schemas/atom'
            }
            
            entries = root.findall('atom:entry', namespaces)
            
            for entry in entries:
                title = entry.find('atom:title', namespaces).text.strip()
                
                # Extract authors
                authors = []
                for author in entry.findall('atom:author', namespaces):
                    name = author.find('atom:name', namespaces).text
                    authors.append(name)
                
                abstract = entry.find('atom:summary', namespaces).text.strip()
                
                # Get PDF URL
                pdf_url = ""
                for link in entry.findall('atom:link', namespaces):
                    if link.get('type') == 'application/pdf':
                        pdf_url = link.get('href')
                        break
                
                published = entry.find('atom:published', namespaces).text
                
                # Extract arXiv ID from the entry ID
                entry_id = entry.find('atom:id', namespaces).text
                arxiv_id = entry_id.split('/')[-1]
                
                paper = ArxivPaper(
                    title=title,
                    authors=authors,
                    abstract=abstract,
                    pdf_url=pdf_url,
                    published=published,
                    arxiv_id=arxiv_id
                )
                
                papers.append(paper)
                
        except ET.ParseError as e:
            logger.error(f"Error parsing XML response: {e}")
        except Exception as e:
            logger.error(f"Error processing arXiv response: {e}")
            
        return papers
    
    def format_search_results(self, papers: List[ArxivPaper], max_papers: int = 3) -> str:
        """Format search results into a readable string for the chatbot"""
        if not papers:
            return "No papers found for your query."
        
        result = f"Found {len(papers)} relevant papers:\n\n"
        
        for i, paper in enumerate(papers[:max_papers], 1):
            authors_str = ", ".join(paper.authors[:3])  # Show first 3 authors
            if len(paper.authors) > 3:
                authors_str += f" et al."
                
            result += f"{i}. **{paper.title}**\n"
            result += f"   Authors: {authors_str}\n"
            result += f"   Abstract: {paper.abstract[:200]}...\n"
            result += f"   arXiv ID: {paper.arxiv_id}\n\n"
        
        if len(papers) > max_papers:
            result += f"... and {len(papers) - max_papers} more papers found."
            
        return result

def search_arxiv(query: str) -> str:
    """
    Convenience function for searching arXiv papers
    
    Args:
        query: Search query string
        
    Returns:
        Formatted string with search results
    """
    service = ArxivService()
    papers = service.search_papers(query)
    return service.format_search_results(papers)