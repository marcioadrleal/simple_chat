from sqlalchemy import text
from dao.database import SessionLocal

def testar_conexao():
    with SessionLocal() as db:
        resultado = db.execute(
            text("SELECT COUNT(*) FROM public.document_chunk")
        ).scalar_one()

        print(f"Conexão OK! Total de chunks: {resultado}")


if __name__ == "__main__":
    testar_conexao()