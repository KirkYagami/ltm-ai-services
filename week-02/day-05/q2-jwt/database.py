from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# SQLite database
DATABASE_URL = "sqlite:///./myapp.db"

# SQLite normally restricts connections to the thread that created them.
# FastAPI may handle requests in different threads, so disable that check.
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}
)

# Session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

# Base class for ORM models
Base = declarative_base()