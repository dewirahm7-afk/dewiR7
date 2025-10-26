# backend/utils/video_utils.py
import subprocess
import json
from pathlib import Path
from typing import Tuple, Optional
import tempfile

class VideoUtils:
    @staticmethod
    def get_video_info(video_path: Path) -> dict:
        """Get video information using ffprobe"""
        try:
            result = subprocess.run([
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_format', '-show_streams', str(video_path)
            ], capture_output=True, text=True, check=True)
            
            info = json.loads(result.stdout)
            return info
        except Exception as e:
            print(f"FFprobe error: {e}")
            return {}
    
    @staticmethod
    def get_video_duration(video_path: Path) -> float:
        """Get video duration in seconds"""
        try:
            info = VideoUtils.get_video_info(video_path)
            if info and 'format' in info:
                return float(info['format'].get('duration', 0))
            return 0
        except Exception:
            return 0
    
    @staticmethod
    def extract_audio_segment(video_path: Path, start_time: float, 
                            duration: float, output_path: Path) -> bool:
        """Extract audio segment from video"""
        try:
            subprocess.run([
                'ffmpeg', '-y', '-ss', str(start_time), '-i', str(video_path),
                '-t', str(duration), '-ac', '1', '-ar', '16000',
                '-c', 'pcm_s16le', str(output_path)
            ], check=True, capture_output=True)
            return True
        except Exception as e:
            print(f"Audio extraction error: {e}")
            return False
    
    @staticmethod
    def create_video_preview(video_path: Path, output_path: Path, 
                           duration: int = 30) -> bool:
        """Create short preview of video"""
        try:
            subprocess.run([
                'ffmpeg', '-y', '-i', str(video_path), '-t', str(duration),
                '-c', 'copy', str(output_path)
            ], check=True, capture_output=True)
            return True
        except Exception as e:
            print(f"Preview creation error: {e}")
            return False
    
    @staticmethod
    def generate_waveform(video_path: Path, output_path: Path, 
                         width: int = 800, height: int = 200) -> bool:
        """Generate waveform image from audio"""
        try:
            subprocess.run([
                'ffmpeg', '-y', '-i', str(video_path),
                '-filter_complex', f'[0:a]showwavespic=colors=#007bff:scale=sqrt:s={width}x{height}[wave]',
                '-map', '[wave]', '-frames:v', '1', str(output_path)
            ], check=True, capture_output=True)
            return True
        except Exception as e:
            print(f"Waveform generation error: {e}")
            return False
    
    @staticmethod
    def convert_to_mp4(input_path: Path, output_path: Path) -> bool:
        """Convert video to MP4 format"""
        try:
            subprocess.run([
                'ffmpeg', '-y', '-i', str(input_path),
                '-c:v', 'libx264', '-c:a', 'aac', '-movflags', '+faststart',
                str(output_path)
            ], check=True, capture_output=True)
            return True
        except Exception as e:
            print(f"MP4 conversion error: {e}")
            return False
    
    @staticmethod
    def get_video_thumbnail(video_path: Path, output_path: Path, 
                          time_sec: float = 10) -> bool:
        """Extract thumbnail from video"""
        try:
            subprocess.run([
                'ffmpeg', '-y', '-ss', str(time_sec), '-i', str(video_path),
                '-vframes', '1', '-q:v', '2', str(output_path)
            ], check=True, capture_output=True)
            return True
        except Exception as e:
            print(f"Thumbnail extraction error: {e}")
            return False