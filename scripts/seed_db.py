import sys
import os

# Add root folder to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.database import engine, Base
from models.database_models import Hospital

def seed_database():
    print("[INFO] Initializing database tables...")
    Base.metadata.create_all(bind=engine)
    print("[INFO] Database schema successfully created.")

if __name__ == "__main__":
    seed_database()