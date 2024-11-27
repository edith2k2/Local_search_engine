import os
from pathlib import Path
from typing import List, Dict
import mimetypes
from datetime import datetime
import pandas as pd

class DocumentCrawler:
    def __init__(self):
        # Define supported document types - only common document formats
        self.supported_extensions = {
            # Text and Documents
            '.txt', '.rtf', '.md',
            # PDF files
            '.pdf',
            # Microsoft Office
            '.doc', '.docx',
            '.ppt', '.pptx',
            # OpenOffice/LibreOffice
            '.odt', '.ods', '.odp',
            # Others
            '.epub', '.csv'
        }
        
        # Patterns to exclude (hidden files, temp files, etc.)
        self.exclude_patterns = {
            # Hidden files and directories
            '.*',  # Any hidden file/directory
            '__pycache__',
            'node_modules',
            # Temp files
            '~$*',  # Microsoft Office temp files
            '*.tmp',
            '*.temp',
            # System files
            'Thumbs.db',
            '.DS_Store',
            # Python specific
            '*.pyc',
            'venv',
            '.env',
            # Git related
            '.git',
        }
        
        self.documents = []

    def should_process_file(self, file_path: Path) -> bool:
        """
        Determine if a file should be processed based on its name and path
        """
        # Check if file or any parent directory is hidden or should be excluded
        parts = file_path.parts
        for part in parts:
            if any(part.startswith(exclude.replace('*', '')) 
                  for exclude in self.exclude_patterns):
                return False

        # Check if extension is supported
        return file_path.suffix.lower() in self.supported_extensions

    def get_file_metadata(self, file_path: Path) -> Dict:
        """Extract metadata from file"""
        stats = file_path.stat()
        mime_type = mimetypes.guess_type(str(file_path))[0] or 'application/octet-stream'

        # Get file size in appropriate units
        size_bytes = stats.st_size
        if size_bytes < 1024:
            size_str = f"{size_bytes} bytes"
        elif size_bytes < 1024 * 1024:
            size_str = f"{size_bytes/1024:.1f} KB"
        else:
            size_str = f"{size_bytes/(1024*1024):.1f} MB"

        return {
            'file_path': str(file_path),
            'file_name': file_path.name,
            'extension': file_path.suffix.lower(),
            'parent_folder': str(file_path.parent),
            'size_bytes': size_bytes,
            'size_readable': size_str,
            'created_time': datetime.fromtimestamp(stats.st_ctime).strftime('%Y-%m-%d %H:%M:%S'),
            'modified_time': datetime.fromtimestamp(stats.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
            'mime_type': mime_type
        }

    def crawl_directory(self, root_dir: str) -> List[Dict]:
        """Recursively crawl directory and collect document information"""
        root_path = Path(root_dir)
        
        # Validate directory exists
        if not root_path.exists():
            raise ValueError(f"Directory not found: {root_dir}")
        
        print(f"Starting document crawl from: {root_dir}")
        processed_files = 0
        
        # Walk through directory recursively
        for current_path, dirs, files in os.walk(root_dir):
            # Remove hidden directories from dirs list to prevent walking into them
            dirs[:] = [d for d in dirs if not d.startswith('.') and 
                      not any(pattern.replace('*', '') in d 
                             for pattern in self.exclude_patterns)]
            
            # Process each file in current directory
            for file_name in files:
                file_path = Path(current_path) / file_name
                
                # Check if file should be processed
                if self.should_process_file(file_path):
                    try:
                        # Get file metadata
                        doc_info = self.get_file_metadata(file_path)
                        self.documents.append(doc_info)
                        processed_files += 1
                        
                        # Progress update
                        print(f"Found document ({processed_files}): {file_path}")
                    except Exception as e:
                        print(f"Error processing {file_path}: {str(e)}")
        
        print(f"\nCrawl completed. Found {len(self.documents)} documents.")
        return self.documents

    def get_statistics(self) -> Dict:
        """Generate statistics about crawled documents"""
        if not self.documents:
            return {}
        
        df = pd.DataFrame(self.documents)
        
        total_size_bytes = df['size_bytes'].sum()
        total_size_mb = total_size_bytes / (1024 * 1024)
        
        stats = {
            'total_documents': len(df),
            'total_size': f"{total_size_mb:.2f} MB",
            'extensions_found': df['extension'].value_counts().to_dict(),
            'folders_containing_documents': df['parent_folder'].nunique(),
            'date_range': {
                'oldest_file': df['created_time'].min(),
                'newest_file': df['created_time'].max()
            }
        }
        
        return stats

    def save_to_csv(self, output_file: str = 'document_index.csv'):
        """Save document index to CSV"""
        if self.documents:
            df = pd.DataFrame(self.documents)
            df.to_csv(output_file, index=False)
            print(f"Document index saved to: {output_file}")
            return df
        return None
    
    def get_files(self):
        return self.documents

def main():
    # Initialize crawler
    crawler = DocumentCrawler()
    
    try:
        # Get root directory from user
        root_dir = "/Users/battalavamshi/Desktop"
        
        # Crawl directory
        documents = crawler.crawl_directory(root_dir)
        
        print("\nSample of documents found:")
        print(crawler.get_files()[:5])

        # Print statistics
        stats = crawler.get_statistics()
        print("\nDocument Crawl Statistics:")
        print(f"Total Documents: {stats['total_documents']}")
        print(f"Total Size: {stats['total_size']}")
        print(f"Number of Folders: {stats['folders_containing_documents']}")
        
        print("\nDocument Types Found:")
        for ext, count in stats['extensions_found'].items():
            print(f"{ext}: {count} files")
            
        print(f"\nDate Range: {stats['date_range']['oldest_file']} to {stats['date_range']['newest_file']}")
        
        # Save to CSV
        df = crawler.save_to_csv()
        if df is not None:
            print("\nSample of documents found:")
            pd.set_option('display.max_columns', None)
            print(df[['file_name', 'extension', 'size_readable', 'modified_time']].head())
        

    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()

# # crawler.py
# import os
# from pathlib import Path
# from typing import List, Generator
# import asyncio
# import aiofiles
# from concurrent.futures import ProcessPoolExecutor
# import logging

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# class FileCrawler:
#     def __init__(self, supported_extensions: set = {'.txt', '.md', '.py'}):
#         self.supported_extensions = supported_extensions
        
#     async def scan_directory(self, root_dir: str) -> List[str]:
#         """Asynchronously scan directory for files."""
#         files = []
#         try:
#             for dirpath, _, filenames in os.walk(root_dir):
#                 for filename in filenames:
#                     if Path(filename).suffix in self.supported_extensions:
#                         full_path = os.path.join(dirpath, filename)
#                         files.append(full_path)
#         except Exception as e:
#             logger.error(f"Error scanning directory {root_dir}: {e}")
            
#         return files

#     async def check_file_modified(self, file_path: str, session) -> bool:
#         """Check if file needs processing by comparing modification time."""
#         try:
#             mtime = os.path.getmtime(file_path)
#             existing = session.query(DBDocument).filter_by(file_path=file_path).first()
#             return not existing or existing.created_at.timestamp() < mtime
#         except Exception as e:
#             logger.error(f"Error checking file {file_path}: {e}")
#             return True
