from sqlalchemy import create_engine, Column, String, Text, Integer, DateTime, Float, Index
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from pgvector.sqlalchemy import Vector
import uuid
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/therapist_rag")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Conversation(Base):
    __tablename__ = "conversations"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    context = Column(Text, nullable=False)
    response = Column(Text, nullable=False)
    context_length = Column(Integer)
    response_length = Column(Integer)
    category = Column(String(50))
    quality_score = Column(Float)
    embedding = Column(Vector(1536))
    extra_data = Column(JSONB)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('ix_conversations_category', 'category'),
        Index('ix_conversations_quality', 'quality_score'),
        Index('ix_conversations_embedding', 'embedding', postgresql_using='ivfflat', postgresql_ops={'embedding': 'vector_cosine_ops'}),
    )

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def create_tables():
    Base.metadata.create_all(bind=engine)

def init_db():
    from sqlalchemy import text
    
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
    
    create_tables()