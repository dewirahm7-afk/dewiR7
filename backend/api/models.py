# backend/api/models.py
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from pathlib import Path

class SessionCreate(BaseModel):
    video_name: str
    srt_name: str

class SessionUpdate(BaseModel):
    status: Optional[str] = None
    progress: Optional[int] = None
    current_step: Optional[str] = None
    error: Optional[str] = None

class DiarizationConfig(BaseModel):
    male_ref: str
    female_ref: str
    hf_token: str
    use_gpu: bool = True
    top_n: int = 5

class TranslationConfig(BaseModel):
    api_key: str
    batch_size: int = 20
    workers: int = 1
    timeout: int = 90

class TTSConfig(BaseModel):
    tts_engine: str = "edge"
    voice_male: str = "id-ID-ArdiNeural"
    voice_female: str = "id-ID-GadisNeural"
    voice_unknown: str = "id-ID-ArdiNeural"
    replace_audio: bool = False
    bg_mode: str = "center_cut"
    max_atempo: float = 1.8
    fade_sec: float = 0.02
    mix_chunk: int = 400
    tts_timeout: int = 25
    
    # ElevenLabs specific
    el_api_key: Optional[str] = None
    el_model: str = "eleven_multilingual_v2"
    el_voice_male: Optional[str] = None
    el_voice_female: Optional[str] = None
    el_voice_unknown: Optional[str] = None
    el_stability: float = 0.3
    el_similarity: float = 0.8
    el_style: float = 0.0
    el_boost: bool = True

class EditEntryRequest(BaseModel):
    index: int
    text: str
    gender: str

class WebSocketMessage(BaseModel):
    type: str
    data: Dict[str, Any]