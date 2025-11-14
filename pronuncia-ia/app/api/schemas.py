from pydantic import BaseModel
from typing import List, Optional


class Highlights(BaseModel):
    correct: List[str] = []
    incorrect: List[str] = []


class EvaluateResponse(BaseModel):
    score: float
    similarity: Optional[float] = None
    match: bool
    predicted: str
    expected: Optional[str] = None
    feedback: str
    errors: List[str] = []
    suggestions: List[str] = []
    highlights: Highlights = Highlights()
    method: str
    language: Optional[str] = None
    user_id: Optional[str] = None
    transcription_provider: Optional[str] = None
    audio_name: Optional[str] = None


class TranscribeResponse(BaseModel):
    transcript: str


class TaskCategoryModel(BaseModel):
    key: str
    title: str
    description: str
    sample_count: int


class GeneratedItem(BaseModel):
    text: str
    target_words: List[str] = []
    instructions: Optional[str] = None
    estimated_duration_s: Optional[int] = None
