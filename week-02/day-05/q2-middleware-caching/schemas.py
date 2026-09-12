from pydantic import BaseModel


class WeatherForecast(BaseModel):
    city: str
    temperature_c: float
    condition: str
    source: str