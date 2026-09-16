from pydantic import BaseModel

class TelemetryInput(BaseModel):
    ambulance_id: str
    heart_rate: float
    spo2: float
    systolic_bp: float
    respiratory_rate: float
    current_lat: float
    current_lon: float