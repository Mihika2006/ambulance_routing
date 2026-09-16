import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# SQLite fallback if PostgreSQL is not running locally
DB_URL = os.getenv("DATABASE_URL", "sqlite:///./ambulance_system.db")

engine = create_engine(
    DB_URL, 
    connect_args={"check_same_thread": False} if "sqlite" in DB_URL else {}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()