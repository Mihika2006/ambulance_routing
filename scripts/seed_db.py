import sys
import os

# Add root folder (ambulance_routing) to Python search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.database import engine, Base
from models.database_models import Hospital

def seed_database():
    print("[INFO] Creating database tables...")
    Base.metadata.create_all(bind=engine)
    print("[INFO] Database schema initialized.")

if __name__ == "__main__":
    seed_database()