import sqlite3
import os
from typing import List, Dict, Optional, Tuple
import logging

class DatabaseService:
    def __init__(self, db_path: str = "arxiv_database.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the SQLite database with required tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create documents table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    document_id INTEGER PRIMARY KEY,
                    chunk_id INTEGER,
                    source TEXT,
                    chunk_length INTEGER
                )
            """)
            
            # Create FTS5 virtual table for document chunks
            cursor.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS doc_chunks USING fts5(
                    content,
                    content='documents',
                    content_rowid='document_id'
                )
            """)
            
            conn.commit()
            logging.info("Database initialized successfully")
    
    def insert_document(self, doc_id: int, chunk_id: int, source: str, chunk_length: int) -> int:
        """Insert a new document and return its doc_id."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO documents (document_id, chunk_id, source, chunk_length)
                VALUES (?, ?, ?, ?)
            """, (doc_id, chunk_id, source, chunk_length))
            conn.commit()
            return cursor.lastrowid
    
    def insert_document_list(self, chunk_meta_data: List[Dict]):

        if chunk_meta_data is None:
            raise ValueError("No Meta data found")
        """Insert a new document and return its doc_id."""
        
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            for meta in chunk_meta_data:
                cursor.execute("""
                    INSERT INTO documents (document_id, chunk_id, source, chunk_length)
                    VALUES (?, ?, ?, ?)
                """, (meta["document_id"], meta["chunk_id"], meta["source"], meta["chunk_length"]))
            conn.commit()
    
    def insert_chunk(self, row_id: int, content: str):
        """Insert a document chunk into the FTS5 table."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO doc_chunks (content_rowid, content)
                VALUES (?, ?)
            """, (row_id, content))
            conn.commit()
    
    def insert_chunk_list(self, chunks: list[str], chunk_meta_data: list[Dict]):
        """Insert document chunks into the FTS5 table."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            for i, chunk in enumerate(chunks):
                cursor.execute("""
                    INSERT INTO doc_chunks (content_rowid, content)
                    VALUES (?, ?)
                """, (chunk_meta_data[i]["document_id"], chunk))
            conn.commit()
    
    def search_documents(self, query: str, limit: int = 10) -> List[Dict]:
        """Search documents using FTS5 full-text search."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT d.document_id, d.chunk_id, d.source, d.chunk_length, 
                       dc.content, rank
                FROM doc_chunks dc
                JOIN documents d ON d.document_id = dc.rowid
                WHERE doc_chunks MATCH ?
                ORDER BY rank
                LIMIT ?
            """, (query, limit))
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    'document_id': row[0],
                    'chunk_id': row[1],
                    'source': row[2],
                    'chunk_length': row[3],
                    'content': row[4],
                    'rank': row[5]
                })
            return results
    
    def get_document(self, doc_id: int) -> Optional[Dict]:
        """Get a specific document by ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT document_id, chunk_id, source, chunk_length
                FROM documents
                WHERE document_id = ?
            """, (doc_id,))
            
            row = cursor.fetchone()
            if row:
                return {
                    'document_id': row[0],
                    'chunk_id': row[1],
                    'source': row[2],
                    'chunk_length': row[3]
                }
            return None
    
    def get_all_documents(self) -> List[Dict]:
        """Get all documents."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT document_id, chunk_id, source, chunk_length
                FROM documents
            """)
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    'document_id': row[0],
                    'chunk_id': row[1],
                    'source': row[2],
                    'chunk_length': row[3]
                })
            return results
    
    def delete_document(self, doc_id: int):
        """Delete a document and its chunks."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Delete from FTS5 table first
            cursor.execute("DELETE FROM doc_chunks WHERE rowid = ?", (doc_id,))
            
            # Delete from documents table
            cursor.execute("DELETE FROM documents WHERE document_id = ?", (doc_id,))
            
            conn.commit()
    
    def get_stats(self) -> Dict:
        """Get database statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM documents")
            total_docs = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM doc_chunks")
            total_chunks = cursor.fetchone()[0]
            
            return {
                'total_documents': total_docs,
                'total_chunks': total_chunks
            }