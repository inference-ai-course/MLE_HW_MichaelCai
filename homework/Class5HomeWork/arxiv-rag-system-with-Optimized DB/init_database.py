#!/usr/bin/env python3
"""
Database initialization script for arXiv RAG system.
Creates SQLite database with documents and doc_chunks tables.
"""

import sqlite3
import os
import sys

def create_database(db_path: str = "arxiv_database.db"):
    """Create the SQLite database with required tables."""
    
    # Remove existing database if it exists
    if os.path.exists(db_path):
        print(f"Removing existing database: {db_path}")
        os.remove(db_path)
    
    print(f"Creating new database: {db_path}")
    
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        
        # Create documents table
        print("Creating documents table...")
        cursor.execute("""
            CREATE TABLE documents (
                doc_id    INTEGER PRIMARY KEY,
                title     TEXT,
                author    TEXT,
                year      INTEGER,
                keywords  TEXT
            )
        """)
        
        # Create FTS5 virtual table for document chunks
        print("Creating doc_chunks FTS5 virtual table...")
        cursor.execute("""
            CREATE VIRTUAL TABLE doc_chunks USING fts5(
                content,
                content='documents',
                content_rowid='doc_id'
            )
        """)
        
        conn.commit()
        print("Database created successfully!")
        
        # Verify tables were created
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        print(f"Created tables: {[table[0] for table in tables]}")

def main():
    """Main function to run database initialization."""
    db_path = sys.argv[1] if len(sys.argv) > 1 else "arxiv_database.db"
    
    try:
        create_database(db_path)
        print(f"\n✅ Database initialization completed successfully!")
        print(f"Database file: {os.path.abspath(db_path)}")
        
    except Exception as e:
        print(f"❌ Error creating database: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()