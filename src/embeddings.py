import pandas as pd
from openai import OpenAI
from typing import List, Dict, Optional
import os
from pathlib import Path
import logging
from dotenv import load_dotenv
from .database import SessionLocal, Conversation, init_db
from sqlalchemy import func, distinct

load_dotenv()
logger = logging.getLogger(__name__)


class OpenAIEmbeddingManager:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        self.dimension = int(os.getenv("EMBEDDING_DIMENSION", "1536"))

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        logger.info(f"Generating embeddings for {len(texts)} texts using {self.model}")

        try:
            response = self.client.embeddings.create(input=texts, model=self.model)

            embeddings = [data.embedding for data in response.data]
            actual_dimension = len(embeddings[0]) if embeddings else 0
            logger.info(
                f"Generated {len(embeddings)} embeddings with dimension {actual_dimension}"
            )
            return embeddings

        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise

    def load_data_and_store_embeddings(self, data_path: str):
        logger.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)

        init_db()

        db = SessionLocal()
        try:
            db.query(Conversation).delete()
            db.commit()

            batch_size = 100
            total_rows = len(df)

            for i in range(0, total_rows, batch_size):
                batch_df = df.iloc[i : i + batch_size]

                combined_texts = []
                for _, row in batch_df.iterrows():
                    combined_text = (
                        f"Context: {row['Context']} Response: {row['Response']}"
                    )
                    combined_texts.append(combined_text)

                embeddings = self.generate_embeddings(combined_texts)

                conversations = []
                for j, (_, row) in enumerate(batch_df.iterrows()):
                    conversation = Conversation(
                        context=row["Context"],
                        response=row["Response"],
                        context_length=row["context_length"],
                        response_length=row["response_length"],
                        category=row["category"],
                        quality_score=row["quality_score"],
                        embedding=embeddings[j],
                        extra_data={
                            "original_id": int(row["id"]),
                            "combined_text": combined_texts[j],
                        },
                    )
                    conversations.append(conversation)

                db.add_all(conversations)
                db.commit()

                logger.info(
                    f"Processed batch {i//batch_size + 1}/{(total_rows-1)//batch_size + 1}"
                )

            total_stored = db.query(Conversation).count()
            logger.info(
                f"Successfully stored {total_stored} conversations with embeddings"
            )

        except Exception as e:
            db.rollback()
            logger.error(f"Error storing embeddings: {e}")
            raise
        finally:
            db.close()

    def search_similar(
        self,
        query: str,
        top_k: int = 5,
        min_similarity: float = 0.7,
        category_filter: Optional[str] = None,
    ) -> List[Dict]:
        try:
            query_embedding = self.generate_embeddings([query])[0]

            db = SessionLocal()
            try:
                query_obj = db.query(
                    Conversation.id,
                    Conversation.context,
                    Conversation.response,
                    Conversation.category,
                    Conversation.quality_score,
                    Conversation.context_length,
                    Conversation.response_length,
                    Conversation.extra_data,
                    (1 - Conversation.embedding.cosine_distance(query_embedding)).label(
                        "similarity"
                    ),
                ).filter(
                    (1 - Conversation.embedding.cosine_distance(query_embedding))
                    > min_similarity
                )

                if category_filter:
                    query_obj = query_obj.filter(
                        Conversation.category == category_filter
                    )

                query_obj = query_obj.order_by(
                    Conversation.embedding.cosine_distance(query_embedding)
                ).limit(top_k)

                rows = query_obj.all()

                results = []
                for row in rows:
                    results.append(
                        {
                            "similarity": float(row.similarity),
                            "metadata": {
                                "id": str(row.id),
                                "Context": row.context,
                                "Response": row.response,
                                "category": row.category,
                                "quality_score": float(row.quality_score),
                                "context_length": row.context_length,
                                "response_length": row.response_length,
                                "extra_data": row.extra_data,
                            },
                        }
                    )

                return results

            finally:
                db.close()

        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            raise

    def get_stats(self) -> Dict:
        db = SessionLocal()
        try:

            total_conversations = db.query(func.count(Conversation.id)).scalar()
            categories = db.query(distinct(Conversation.category)).all()
            avg_context_length = db.query(
                func.avg(Conversation.context_length)
            ).scalar()
            avg_response_length = db.query(
                func.avg(Conversation.response_length)
            ).scalar()
            avg_quality_score = db.query(func.avg(Conversation.quality_score)).scalar()

            return {
                "status": "loaded",
                "total_conversations": total_conversations or 0,
                "embedding_dimension": 1536,
                "embedding_model": self.model,
                "categories": [cat[0] for cat in categories if cat[0]],
                "avg_context_length": float(avg_context_length or 0),
                "avg_response_length": float(avg_response_length or 0),
                "avg_quality_score": float(avg_quality_score or 0),
            }

        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {"status": "error", "error": str(e)}
        finally:
            db.close()


def main():
    base_dir = Path(__file__).parent.parent
    data_path = base_dir / "data" / "processed" / "cleaned_conversations.csv"

    if not data_path.exists():
        logger.error(f"Processed data not found at {data_path}")
        logger.error("Please run data_processor.py first")
        return

    if (
        not os.getenv("OPENAI_API_KEY")
        or os.getenv("OPENAI_API_KEY") == "your_openai_api_key_here"
    ):
        logger.error("Please set your OPENAI_API_KEY in the .env file")
        return

    embedding_manager = OpenAIEmbeddingManager()
    embedding_manager.load_data_and_store_embeddings(str(data_path))

    stats = embedding_manager.get_stats()
    logger.info("EMBEDDINGS GENERATION COMPLETE")
    logger.info(f"Total conversations: {stats['total_conversations']}")
    logger.info(f"Embedding dimension: {stats['embedding_dimension']}")
    logger.info(f"Model used: {stats['embedding_model']}")
    logger.info(f"Categories: {', '.join(stats['categories'])}")


if __name__ == "__main__":
    main()
