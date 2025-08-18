#Pydantic models for the API
from pydantic import BaseModel
from typing import List

class Topic(BaseModel):
    topic: str
    source_layer: str