from sqlalchemy import Column, Integer, String, Float, DateTime
from config.database import Base
import datetime

class Hospital(Base):
    __tablename__ = "hospitals"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    emergency_type = Column(String, nullable=False)
    icu_beds_available = Column(Integer, default=5)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)

class TripLog(Base):
    __tablename__ = "trip_logs"

    id = Column(Integer, primary_key=True, index=True)
    ambulance_id = Column(String, index=True)
    news2_score = Column(Integer)
    spo2 = Column(Float)
    heart_rate = Column(Float)
    travel_time_min = Column(Float)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)