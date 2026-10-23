from sqlalchemy import text
from sentence_transformers import SentenceTransformer
from pgvector.sqlalchemy import Vector

from dao.database import SessionLocal


class VectorSearchService:

    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def buscar(self, pergunta: str, limite: int = 5):

        embedding = self.model.encode(pergunta).tolist()

        with SessionLocal() as db:
            resultados = db.execute(
                text("""
                    SELECT
                        document_id,
                        content,
                        embedding <=> CAST(:embedding AS vector) AS distancia
                    FROM public.document_chunk
                    WHERE embedding IS NOT NULL
                    ORDER BY embedding <=> CAST(:embedding AS vector)
                    LIMIT :limite
                """),
                {
                    "embedding": str(embedding),
                    "limite": limite
                }
            ).mappings().all()

        return [dict(row) for row in resultados]