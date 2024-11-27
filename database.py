
# database.py
from sqlalchemy import create_engine, Column, String, JSON, LargeBinary, ForeignKey, DateTime, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from typing import List
from datetime import datetime

Base = declarative_base()


class DBChunk(Base):
    __tablename__ = 'chunks'
    
    id = Column(String, primary_key=True)
    content = Column(String)
    metadata = Column(JSON)
    embedding = Column(LargeBinary)
    document_id = Column(String, ForeignKey('documents.id'))
    document = relationship("DBDocument", back_populates="chunks")

class DBDocument(Base):
    __tablename__ = 'documents'
    
    id = Column(String, primary_key=True)
    content = Column(String)
    file_path = Column(String, unique=True)
    metadata = Column(JSON)
    embedding = Column(LargeBinary)  # Store numpy array as binary
    created_at = Column(DateTime, default=datetime.utcnow)
    chunks: List[DBChunk] = relationship(
        "DBChunk",
        back_populates="document",
        uselist=True,  # This makes it one-to-many
        cascade="all, delete-orphan"  # Automatically handle related chunks
    )
    hash = Column(String)
    status = Column(String)  # 'pending', 'processing', 'completed', 'error'
