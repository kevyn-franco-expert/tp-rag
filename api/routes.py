from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import HTMLResponse
from pathlib import Path
from datetime import datetime
import logging

from src.models import (
    SearchRequest, SearchResponse, CaseResult,
    GuidanceRequest, GuidanceResponse,
    SystemStats, HealthCheck
)
from src.rag_engine import TherapistRAGEngine

logger = logging.getLogger(__name__)

router = APIRouter()

rag_engine = None

def get_rag_engine() -> TherapistRAGEngine:
    global rag_engine
    if rag_engine is None:
        try:
            rag_engine = TherapistRAGEngine()
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"Failed to initialize RAG engine: {str(e)}"
            )
    
    return rag_engine

@router.get("/", response_class=HTMLResponse)
async def root():
    frontend_path = Path(__file__).parent.parent / "frontend" / "index.html"
    if frontend_path.exists():
        return HTMLResponse(content=frontend_path.read_text())
    return HTMLResponse(content="<h1>Therapist RAG System</h1><p>Frontend not found</p>")

@router.get("/health", response_model=HealthCheck)
async def health_check(rag: TherapistRAGEngine = Depends(get_rag_engine)):
    health_data = rag.health_check()
    health_data["timestamp"] = datetime.now().isoformat()
    return HealthCheck(**health_data)

@router.get("/stats", response_model=SystemStats)
async def get_system_stats(rag: TherapistRAGEngine = Depends(get_rag_engine)):
    try:
        stats = rag.get_system_stats()
        if stats.get('status') != 'loaded':
            raise HTTPException(status_code=503, detail="System not properly loaded")
        return SystemStats(**stats)
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving system statistics")

@router.post("/search", response_model=SearchResponse)
async def search_similar_cases(
    request: SearchRequest,
    rag: TherapistRAGEngine = Depends(get_rag_engine)
):
    try:
        similar_cases = rag.search_similar_cases(
            query=request.query,
            top_k=request.top_k,
            min_similarity=request.min_similarity,
            category_filter=request.category_filter
        )
        
        results = []
        for case in similar_cases:
            metadata = case['metadata']
            result = CaseResult(
                similarity=case['similarity'],
                context=metadata['Context'],
                response=metadata['Response'],
                category=metadata['category'],
                quality_score=metadata['quality_score'],
                context_length=metadata['context_length'],
                response_length=metadata['response_length']
            )
            results.append(result)
        
        return SearchResponse(
            results=results,
            total_found=len(results),
            query=request.query
        )
    
    except Exception as e:
        logger.error(f"Error in search: {e}")
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

@router.post("/guidance", response_model=GuidanceResponse)
async def generate_guidance(
    request: GuidanceRequest,
    rag: TherapistRAGEngine = Depends(get_rag_engine)
):
    try:
        guidance_data = rag.generate_guidance(
            patient_context=request.patient_context,
            therapist_question=request.therapist_question,
            top_k=request.top_k
        )
        
        similar_cases = []
        for case in guidance_data['similar_cases']:
            metadata = case['metadata']
            result = CaseResult(
                similarity=case['similarity'],
                context=metadata['Context'],
                response=metadata['Response'],
                category=metadata['category'],
                quality_score=metadata['quality_score'],
                context_length=metadata['context_length'],
                response_length=metadata['response_length']
            )
            similar_cases.append(result)
        
        return GuidanceResponse(
            guidance=guidance_data['guidance'],
            confidence_score=guidance_data['confidence_score'],
            similar_cases=similar_cases,
            warnings=guidance_data['warnings'],
            recommendations=guidance_data['recommendations']
        )
    
    except Exception as e:
        logger.error(f"Error generating guidance: {e}")
        raise HTTPException(status_code=500, detail=f"Guidance generation error: {str(e)}")

@router.get("/categories")
async def get_categories(rag: TherapistRAGEngine = Depends(get_rag_engine)):
    try:
        stats = rag.get_system_stats()
        return {"categories": stats.get('categories', [])}
    except Exception as e:
        logger.error(f"Error getting categories: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving categories")

@router.get("/sample-queries")
async def get_sample_queries():
    return {
        "sample_queries": [
            "Patient feeling depressed and having trouble sleeping",
            "Anxiety and panic attacks in social situations",
            "Relationship problems and communication issues",
            "Low self-esteem and feelings of worthlessness",
            "Trauma recovery and PTSD symptoms",
            "Teenage depression and family conflicts"
        ]
    }

@router.get("/api-info")
async def get_api_info():
    return {
        "name": "Therapist RAG System API",
        "version": "1.0.0",
        "description": "RAG-based system for finding similar therapy cases and generating guidance",
        "endpoints": {
            "GET /health": "Health check",
            "GET /stats": "System statistics",
            "POST /search": "Search similar cases",
            "POST /guidance": "Generate therapeutic guidance",
            "GET /categories": "Available categories",
            "GET /sample-queries": "Sample queries for testing"
        },
        "note": "This is a Proof of Concept for demonstration purposes only"
    }