
import os

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

# Carrega as variáveis do arquivo .env
load_dotenv()

DATABASE_URL = os.environ["DATABASE_URL"]


# Classe base das entidades
class Base(DeclarativeBase):
    pass


# Engine: gerencia as conexões com PostgreSQL
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    echo=False
)


# Fábrica de sessões
SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    expire_on_commit=False
)


# Dependência utilizada pelos Controllers FastAPI
def get_db():
    db = SessionLocal()

    try:
        yield db
    finally:
        db.close()


