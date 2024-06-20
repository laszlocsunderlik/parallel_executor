from pydantic import BaseModel
from datetime import date

class SnippedGeomMeta(BaseModel):
    iso_a2: str
    name: str
    type: str
    region: str
    sub_region: str
    postal: str


class Coordinate(BaseModel):
    lat: float
    lon: float


class ImageCoordinates(BaseModel):
    top_left: Coordinate
    top_right: Coordinate
    bottom_right: Coordinate
    bottom_left: Coordinate


class RasterMeta(BaseModel):
    soy: int
    corn: int
    spring_wheat: int
    winter_wheat: int
    flight_date: date
    img_coords: ImageCoordinates
