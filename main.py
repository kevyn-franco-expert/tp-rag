from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import logging
import uvicorn
from dotenv import load_dotenv
import os

from api.routes import router

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Therapist RAG System",
    description="RAG-based system for therapists",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1", tags=["rag"])

frontend_dir = Path(__file__).parent / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")

@app.on_event("startup")
async def startup_event():
    logger.info("Starting Therapist RAG System...")
    base_dir = Path(__file__).parent
    embeddings_path = base_dir / "data" / "processed" / "embeddings.pkl"
    
    if not embeddings_path.exists():
        logger.warning("Embeddings not found. Run: python -m src.data_processor && python -m src.embeddings")
    else:
        logger.info("System ready.")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", 8001)),
        reload=os.getenv("DEBUG", "True").lower() == "true"
    )