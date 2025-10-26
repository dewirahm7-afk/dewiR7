# backend/utils/file_utils.py
import shutil
import json
from pathlib import Path
from typing import List, Dict, Any
import tempfile

class FileUtils:
    @staticmethod
    def safe_copy(src: Path, dst: Path):
        """Safely copy file with error handling"""
        try:
            shutil.copy2(src, dst)
            return True
        except Exception as e:
            print(f"Copy error {src} -> {dst}: {e}")
            return False
    
    @staticmethod
    def safe_move(src: Path, dst: Path):
        """Safely move file with error handling"""
        try:
            shutil.move(str(src), str(dst))
            return True
        except Exception as e:
            print(f"Move error {src} -> {dst}: {e}")
            return False
    
    @staticmethod
    def read_json_safe(path: Path, default=None):
        """Safely read JSON file"""
        try:
            if path.exists():
                return json.loads(path.read_text(encoding='utf-8'))
            return default
        except Exception as e:
            print(f"JSON read error {path}: {e}")
            return default
    
    @staticmethod
    def write_json_safe(path: Path, data: Any):
        """Safely write JSON file"""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
            return True
        except Exception as e:
            print(f"JSON write error {path}: {e}")
            return False
    
    @staticmethod
    def find_latest_file(pattern: str, directory: Path) -> Path:
        """Find the latest file matching pattern"""
        files = list(directory.glob(pattern))
        if not files:
            return None
        return max(files, key=lambda p: p.stat().st_mtime)
    
    @staticmethod
    def cleanup_old_files(directory: Path, pattern: str, keep_count: int = 5):
        """Clean up old files, keeping only the most recent ones"""
        files = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        for old_file in files[keep_count:]:
            try:
                old_file.unlink()
            except Exception as e:
                print(f"Cleanup error {old_file}: {e}")
    
    @staticmethod
    def get_file_size_mb(path: Path) -> float:
        """Get file size in MB"""
        if path.exists():
            return path.stat().st_size / (1024 * 1024)
        return 0
    
    @staticmethod
    def create_temp_dir(prefix: str = "dracin_") -> Path:
        """Create temporary directory"""
        temp_dir = Path(tempfile.mkdtemp(prefix=prefix))
        return temp_dir
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename for safe filesystem use"""
        import re
        # Remove invalid characters
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
        # Limit length
        if len(sanitized) > 255:
            name, ext = sanitized.rsplit('.', 1)
            sanitized = name[:255-len(ext)-1] + '.' + ext
        return sanitized