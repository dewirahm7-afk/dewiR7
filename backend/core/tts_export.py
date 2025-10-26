# backend/core/tts_export.py
import asyncio
import json
from pathlib import Path
from typing import Dict, Any, Callable
import time

class TTSExportEngine:
    def __init__(self):
        self.active_exports = {}
        
    async def process(self, session, config: Dict[str, Any], progress_callback: Callable):
        """Run TTS generation and video export"""
        
        def _run_tts_export():
            try:
                # Import existing dracindub logic
                from dracindub import (
                    build_entries_with_speakers,
                    sanitize_text_for_tts,
                    adjust_to_duration,
                    prevent_overlap,
                    chunked_mix,
                    mux_video,
                    ffprobe_duration
                )
                
                # Prepare paths
                workdir = session.workdir
                video_path = session.video_path
                srt_id = Path(session.srt_id)
                segjson = Path(session.segjson)
                spkjson = Path(session.spkjson)
                
                # Build entries with speaker info
                entries = build_entries_with_speakers(srt_id, segjson, spkjson)
                total_entries = len(entries)
                
                # Progress tracking
                progress_steps = {
                    'tts_generation': 70,  # 70% of progress for TTS
                    'mixing': 20,          # 20% for mixing
                    'export': 10           # 10% for final export
                }
                
                # Step 1: TTS Generation
                segfiles = []
                current_progress = 0
                
                for i, (idx, t0, t1, text, spk, gender) in enumerate(entries):
                    dur = max(0.05, t1 - t0)
                    base = workdir / f"seg_{idx:05d}"
                    adj_wav = workdir / f"seg_{idx:05d}.wav"
                    
                    # Skip if already exists (resume capability)
                    if adj_wav.exists() and adj_wav.stat().st_size > 0:
                        seg_dur = ffprobe_duration(adj_wav)
                        segfiles.append((t0, adj_wav, seg_dur))
                        continue
                    
                    clean_text = sanitize_text_for_tts(text)
                    
                    if not clean_text:
                        # Create silent segment
                        self._create_silent_audio(adj_wav, dur)
                    else:
                        # Generate TTS based on engine
                        raw_wav = self._generate_tts(
                            clean_text, gender, base, config
                        )
                        
                        if raw_wav and raw_wav.exists():
                            # Adjust to target duration
                            tmp_out = workdir / f"{adj_wav.stem}.tmp.wav"
                            adjust_to_duration(
                                raw_wav, dur, tmp_out,
                                min_atempo=1.2,
                                max_atempo=config.get('max_atempo', 1.8),
                                fade=config.get('fade_sec', 0.02)
                            )
                            if adj_wav.exists():
                                adj_wav.unlink()
                            tmp_out.rename(adj_wav)
                    
                    segfiles.append((t0, adj_wav, dur))
                    
                    # Update progress
                    tts_progress = (i + 1) / total_entries * progress_steps['tts_generation']
                    overall_progress = current_progress + tts_progress
                    asyncio.create_task(
                        progress_callback(
                            session.id, 
                            int(overall_progress),
                            f"TTS Generation: {i+1}/{total_entries}"
                        )
                    )
                
                current_progress += progress_steps['tts_generation']
                
                # Step 2: Prevent overlap and prepare mixing
                asyncio.create_task(
                    progress_callback(session.id, current_progress, "Preparing audio segments")
                )
                segfiles = prevent_overlap(segfiles, workdir)
                
                # Step 3: Mix audio segments
                asyncio.create_task(
                    progress_callback(session.id, current_progress + 5, "Mixing audio segments")
                )
                video_dur = ffprobe_duration(video_path)
                final_wav = workdir / f"{video_path.stem}_dubtrack.wav"
                
                chunked_mix(
                    workdir, video_dur, segfiles, final_wav,
                    chunk_size=config.get('mix_chunk', 400)
                )
                
                current_progress += progress_steps['mixing']
                
                # Step 4: Mux final video
                asyncio.create_task(
                    progress_callback(session.id, current_progress + 5, "Creating final video")
                )
                out_video = video_path.with_name(f"{video_path.stem}_dubbed.mp4")
                
                # Prepare args for muxing
                class Args:
                    replace_audio = config.get('replace_audio', False)
                    bg_mode = config.get('bg_mode', 'center_cut')
                    mix_original_db = -18
                
                mux_video(Args(), video_path, final_wav, out_video)
                
                asyncio.create_task(
                    progress_callback(session.id, 100, "Export completed")
                )
                
                return {
                    'success': True,
                    'data': {
                        'output_video': str(out_video),
                        'dub_audio': str(final_wav),
                        'total_segments': len(segfiles)
                    }
                }
                
            except Exception as e:
                return {'success': False, 'error': str(e)}
        
        # Run in executor
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _run_tts_export)
        return result
    
    def _generate_tts(self, text: str, gender: str, base_path: Path, config: Dict[str, Any]) -> Path:
        """Generate TTS audio using configured engine"""
        try:
            if config['tts_engine'] == 'elevenlabs':
                return self._generate_elevenlabs_tts(text, gender, base_path, config)
            else:
                return self._generate_edge_tts(text, gender, base_path, config)
        except Exception as e:
            print(f"TTS generation error: {e}")
            return None
    
    def _generate_edge_tts(self, text: str, gender: str, base_path: Path, config: Dict[str, Any]) -> Path:
        """Generate TTS using Edge TTS"""
        try:
            from dracindub import tts_line_edge
            
            # Determine voice based on gender
            if gender == "Male":
                voice = config.get('voice_male', 'id-ID-ArdiNeural')
            elif gender == "Female":
                voice = config.get('voice_female', 'id-ID-GadisNeural')
            else:
                voice = config.get('voice_unknown', 'id-ID-ArdiNeural')
            
            # Generate TTS
            result = tts_line_edge(
                text, gender, base_path,
                config.get('voice_male', 'id-ID-ArdiNeural'),
                config.get('voice_female', 'id-ID-GadisNeural'),
                config.get('voice_unknown', 'id-ID-ArdiNeural'),
                config.get('rate', '+0%'),
                config.get('volume', '+0%'),
                config.get('pitch', '+0Hz')
            )
            
            return Path(result) if result else None
            
        except Exception as e:
            print(f"Edge TTS error: {e}")
            return None
    
    def _generate_elevenlabs_tts(self, text: str, gender: str, base_path: Path, config: Dict[str, Any]) -> Path:
        """Generate TTS using ElevenLabs"""
        try:
            from dracindub import tts_line_elevenlabs
            
            # Determine voice ID based on gender
            if gender == "Male":
                voice_id = config.get('el_voice_male', '')
            elif gender == "Female":
                voice_id = config.get('el_voice_female', '')
            else:
                voice_id = config.get('el_voice_unknown', '')
            
            if not voice_id:
                raise ValueError("ElevenLabs voice ID not configured")
            
            # Load API keys
            api_keys = [config.get('el_api_key', '')]
            
            # Generate TTS
            result = tts_line_elevenlabs(
                text, gender, base_path,
                config.get('el_voice_male', ''),
                config.get('el_voice_female', ''),
                config.get('el_voice_unknown', ''),
                api_keys
            )
            
            return Path(result) if result else None
            
        except Exception as e:
            print(f"ElevenLabs TTS error: {e}")
            return None
    
    def _create_silent_audio(self, output_path: Path, duration: float):
        """Create silent audio segment"""
        from dracindub import run
        
        run([
            "ffmpeg", "-y", "-f", "lavfi", 
            "-i", f"anullsrc=channel_layout=stereo:sample_rate=48000:duration={duration:.3f}",
            str(output_path)
        ])