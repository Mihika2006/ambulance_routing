from pydantic import BaseModel

class TelemetryInput(BaseModel):
    ambulance_id: str
    ambulance_type: str  # Added: ALS (Advanced Life Support) / BLS
    heart_rate: float
    spo2: float
    systolic_bp: float
    respiratory_rate: float
    current_lat: float
    current_lon: float

class LoginRequest(BaseModel):
    username: str
    password: str
    role: str  # "driver" or "hospital"