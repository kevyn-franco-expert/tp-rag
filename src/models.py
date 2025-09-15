from pydantic import BaseModel, Field
from typing import List, Optional

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=10)
    top_k: int = Field(default=5, ge=1, le=20)
    min_similarity: float = Field(default=0.3, ge=0.0, le=1.0)
    category_filter: Optional[str] = Field(default=None)

class CaseResult(BaseModel):
    similarity: float
    context: str
    response: str
    category: str
    quality_score: float
    context_length: int
    response_length: int

class SearchResponse(BaseModel):
    results: List[CaseResult]
    total_found: int
    query: str

class GuidanceRequest(BaseModel):
    patient_context: str = Field(..., min_length=20)
    therapist_question: str = Field(..., min_length=3)
    top_k: int = Field(default=3, ge=1, le=10)

class GuidanceResponse(BaseModel):
    guidance: str
    confidence_score: float
    similar_cases: List[CaseResult]
    warnings: List[str]
    recommendations: List[str]

class SystemStats(BaseModel):
    status: str
    total_conversations: int
    embedding_dimension: int
    embedding_model: str
    categories: List[str]
    avg_context_length: float
    avg_response_length: float
    avg_quality_score: float

class HealthCheck(BaseModel):
    status: str
    embeddings_loaded: bool
    total_cases: int
    timestamp: str