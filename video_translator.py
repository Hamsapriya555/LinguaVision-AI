"""
video_translator.py — LinguaVision AI
Full video dubbing + SRT + AI summary + analytics pipeline.
"""

import os, re, shutil, subprocess, logging, warnings, time, asyncio, datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import yt_dlp, whisper, edge_tts
from pydub import AudioSegment
from deep_translator import GoogleTranslator

logging.basicConfig(level=logging.INFO,
    format="[%(asctime)s] %(levelname)s — %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU*")

# ── Generate unique timestamp for this run ────────────────────────────────────
_RUN_TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]  # e.g., 20260624_182300

def _get_run_timestamp():
    """Return timestamp shared across all files in this run."""
    return _RUN_TIMESTAMP

# ── Startup ───────────────────────────────────────────────────────────────────
try:
    log.info("yt-dlp version: %s", yt_dlp.version.__version__)
except Exception:
    pass
if not shutil.which("ffmpeg"):
    log.warning("ffmpeg NOT found in PATH")

# ── Whisper model cache (load once at startup) ────────────────────────────────
_WHISPER_MODEL = None

def _get_whisper_model():
    global _WHISPER_MODEL
    if _WHISPER_MODEL is None:
        log.info("Loading Whisper 'base' model (first time)...")
        _WHISPER_MODEL = whisper.load_model("base")
        log.info("Whisper model loaded and cached.")
    return _WHISPER_MODEL

AVAILABLE_LANGUAGES = {"kn":"Kannada","hi":"Hindi","ta":"Tamil","te":"Telugu","ml":"Malayalam"}
DOWNLOADS_DIR  = "downloads"
FINAL_OUT_DIR  = "final_outputs"
EXTRACTED_WAV  = "extracted_audio.wav"
_AUDIO_ONLY_EXTS = {".wav",".mp3",".aac",".ogg",".flac",".m4a",".opus",".weba",".mka"}
_VIDEO_EXTS      = {".mp4",".mkv",".avi",".mov",".flv",".wmv",".m4v",".ts",".3gp"}

_YT_PATTERN = re.compile(
    r"(https?://)?(www\.)?"
    r"(youtube\.com/(watch\?v=|shorts/|embed/|live/)|youtu\.be/)[\w\-]{11}")

def validate_youtube_url(url):
    return bool(_YT_PATTERN.search(url.strip()))

# ── Stream Probe ──────────────────────────────────────────────────────────────
def _probe_streams(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    if ext in _AUDIO_ONLY_EXTS:
        log.info("  Probe '%s': audio-only (ext shortcut)", os.path.basename(filepath))
        return {"has_video": False, "has_audio": True, "duration": _get_duration(filepath)}
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        try:
            def _probe(sel):
                r = subprocess.run([ffprobe,"-v","quiet","-show_streams",
                    "-select_streams",sel,"-of","csv=p=0",filepath],
                    stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
                return bool(r.stdout.strip())
            hv, ha = _probe("v"), _probe("a")
            log.info("  Probe '%s': has_video=%s has_audio=%s", os.path.basename(filepath), hv, ha)
            return {"has_video": hv, "has_audio": ha, "duration": _get_duration(filepath)}
        except Exception as e:
            log.warning("  ffprobe error: %s", e)
    is_video = ext in _VIDEO_EXTS
    return {"has_video": is_video, "has_audio": True, "duration": _get_duration(filepath)}

def _get_duration(filepath):
    try:
        ffprobe = shutil.which("ffprobe")
        if not ffprobe:
            return 0.0
        r = subprocess.run([ffprobe,"-v","quiet","-show_entries","format=duration",
            "-of","default=noprint_wrappers=1:nokey=1",filepath],
            stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        return float(r.stdout.strip() or 0)
    except Exception:
        return 0.0

# ── YouTube Download — AUDIO ONLY (for Whisper) ───────────────────────────────
def download_audio_from_youtube(url, out_dir=DOWNLOADS_DIR):
    os.makedirs(out_dir, exist_ok=True)
    ffmpeg_bin = shutil.which("ffmpeg")
    outtmpl = os.path.join(out_dir, "yt_audio.%(ext)s")
    base = {"ffmpeg_location":ffmpeg_bin,"noplaylist":True,"socket_timeout":30,
            "quiet":False,"no_warnings":False,"concurrent_fragment_downloads":1,
            "extractor_retries":3,"retries":3}
    wav_pp = [{"key":"FFmpegExtractAudio","preferredcodec":"wav","preferredquality":"0"}]
    attempts = [
        {**base,"format":"bestaudio/best","outtmpl":outtmpl,
         "postprocessors":wav_pp,"postprocessor_args":["-ar","16000","-ac","1"]},
        {**base,"format":"bestaudio[ext=webm]/bestaudio[ext=m4a]/bestaudio","outtmpl":outtmpl,
         "postprocessors":wav_pp,"postprocessor_args":["-ar","16000","-ac","1"]},
        {**base,"format":"best[height<=480]/best",
         "outtmpl":os.path.join(out_dir,"yt_fallback.%(ext)s"),"merge_output_format":"mp4"},
    ]
    last_error = None
    for idx, opts in enumerate(attempts, 1):
        log.info("yt-dlp audio attempt %d: %s", idx, opts.get("format"))
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=False)
                log.info("  Title: %s | Formats: %d", info.get("title"), len(info.get("formats",[])))
                ydl.download([url])
            wav = os.path.join(out_dir, "yt_audio.wav")
            if os.path.exists(wav):
                return wav
            for f in sorted(os.listdir(out_dir)):
                c = os.path.join(out_dir, f)
                if os.path.isfile(c):
                    return c
        except yt_dlp.utils.DownloadError as e:
            last_error = str(e)
            log.error("  Attempt %d failed: %s", idx, last_error)
            if any(k in last_error.lower() for k in ("private","age-restrict","unavailable")):
                break
        except Exception as e:
            last_error = str(e)
            log.error("  Attempt %d error: %s", idx, last_error)
    _raise_friendly_error(last_error)

# ── YouTube Download — FULL VIDEO (for dubbing) ───────────────────────────────
def _extract_youtube_id(url):
    """Extract YouTube video ID from common URL forms."""
    match = re.search(r"(?:v=|youtu\.be/|embed/|shorts/)([A-Za-z0-9_-]{11})", url)
    return match.group(1) if match else None


def download_video_from_youtube(url, out_dir=DOWNLOADS_DIR):
    """Download best video+audio merged MP4 for dubbing."""
    os.makedirs(out_dir, exist_ok=True)
    ffmpeg_bin = shutil.which("ffmpeg")
    video_id = _extract_youtube_id(url) or "video"
    timestamp = _get_run_timestamp()
    basename_prefix = f"video_{video_id}_{timestamp}"
    outtmpl = os.path.join(out_dir, f"{basename_prefix}.%(ext)s")
    log.info("YouTube download request: url=%s video_id=%s output_template=%s", url, video_id, outtmpl)

    attempts = [
        {"format":"bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height<=720]+bestaudio/best[height<=720]/best",
         "outtmpl":outtmpl,"merge_output_format":"mp4",
         "ffmpeg_location":ffmpeg_bin,"noplaylist":True,"socket_timeout":60,
         "quiet":False,"concurrent_fragment_downloads":1,"retries":3},
        {"format":"best[height<=480]/best","outtmpl":outtmpl,"merge_output_format":"mp4",
         "ffmpeg_location":ffmpeg_bin,"noplaylist":True,"socket_timeout":60,
         "quiet":False,"retries":3},
    ]
    last_error = None
    for idx, opts in enumerate(attempts, 1):
        log.info("yt-dlp VIDEO attempt %d: %s", idx, opts.get("format"))
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([url])
            downloaded_file = None
            for f in sorted(os.listdir(out_dir)):
                if f.startswith(basename_prefix) and os.path.isfile(os.path.join(out_dir, f)):
                    downloaded_file = os.path.join(out_dir, f)
                    break
            if downloaded_file:
                size = os.path.getsize(downloaded_file)
                log.info("  Video downloaded: %s | Size: %d bytes", downloaded_file, size)
                return downloaded_file
            log.error("  No downloaded video file found for prefix: %s", basename_prefix)
        except Exception as e:
            last_error = str(e)
            log.error("  Video download attempt %d failed: %s", idx, last_error)
    log.warning("Video download failed, will use audio-only pipeline")
    return None

# ── Audio Extraction ──────────────────────────────────────────────────────────
def extract_audio_from_video(video_path, audio_output_path=EXTRACTED_WAV):
    log.info("Extracting audio from: %s", video_path)
    probe = _probe_streams(video_path)
    if not probe.get("has_audio", False):
        raise RuntimeError(
            f"Failed to extract audio: input file has no audio stream. "
            f"Source file: {video_path}")

    ffmpeg_cmd = [
        "ffmpeg","-y","-i",video_path,
        "-vn","-acodec","pcm_s16le",
        "-ar","16000","-ac","1",audio_output_path
    ]
    log.info("  [FORENSIC] FFmpeg command: %s", " ".join(ffmpeg_cmd))
    try:
        result = subprocess.run(ffmpeg_cmd,
            check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        log.info("  [FORENSIC] FFmpeg stdout: %s", result.stdout.strip())
        log.info("  [FORENSIC] FFmpeg stderr: %s", result.stderr.strip())
        return audio_output_path
    except subprocess.CalledProcessError as e:
        log.error("  [FORENSIC] FFmpeg stderr: %s", e.stderr.strip())
        log.error("  [FORENSIC] FFmpeg stdout: %s", e.stdout.strip())
        raise RuntimeError(
            f"Failed to extract audio:\nCommand: {' '.join(ffmpeg_cmd)}\n"
            f"Return code: {e.returncode}\n"
            f"stderr: {e.stderr.strip()}\n"
            f"stdout: {e.stdout.strip()}") from e

# ── Whisper ───────────────────────────────────────────────────────────────────
def transcribe_audio(audio_file):
    log.info("Transcribing with Whisper (cached model)...")
    try:
        model = _get_whisper_model()
        result = model.transcribe(audio_file)
        text = result.get("text", "").strip()
        segments = result.get("segments", [])
        detected_lang = result.get("language", "en")
        if not text:
            raise RuntimeError(
                "Whisper detected no speech. Ensure the video has clear audio.")
        log.info("Transcription done: %d chars, lang=%s, segments=%d",
                 len(text), detected_lang, len(segments))
        return text, segments, detected_lang
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"Transcription failed: {e}") from e

# ── SRT Generation ────────────────────────────────────────────────────────────
def _fmt_srt_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def generate_srt(segments, output_path, translated_text=None):
    """Generate SRT subtitle file from Whisper segments."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    lines = []
    for i, seg in enumerate(segments, 1):
        start = _fmt_srt_time(seg.get("start", 0))
        end   = _fmt_srt_time(seg.get("end", 0))
        text  = seg.get("text","").strip()
        if text:
            lines.append(f"{i}\n{start} --> {end}\n{text}\n")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log.info("SRT written: %s (%d subtitles)", output_path, len(lines))
    return output_path

def generate_translated_srt(segments, translated_text, output_path):
    """Best-effort: distribute translated text across original time segments."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    words = translated_text.split()
    n_segs = len(segments)
    if n_segs == 0:
        return None
    chunk_size = max(1, len(words) // n_segs)
    lines = []
    for i, seg in enumerate(segments, 1):
        start_w = (i-1) * chunk_size
        end_w   = start_w + chunk_size if i < n_segs else len(words)
        chunk   = " ".join(words[start_w:end_w]).strip()
        if not chunk:
            continue
        start = _fmt_srt_time(seg.get("start",0))
        end   = _fmt_srt_time(seg.get("end",0))
        lines.append(f"{i}\n{start} --> {end}\n{chunk}\n")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log.info("Translated SRT: %s", output_path)
    return output_path

# ── AI Summary ────────────────────────────────────────────────────────────────
def generate_ai_summary(text):
    """Extractive AI summary — no external API needed."""
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) > 20]
    summary = " ".join(sentences[:3]) if sentences else text[:300]

    # Key insights: medium-length sentences
    insights = [s for s in sentences if 30 < len(s) < 150][:5]
    if not insights:
        insights = sentences[:3]

    # Main topics: most frequent content words
    stopwords = {"the","a","an","is","it","in","on","at","to","of","and","or","but",
                 "this","that","was","are","be","for","with","as","by","from","have","has"}
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    freq = {}
    for w in words:
        if w not in stopwords:
            freq[w] = freq.get(w, 0) + 1
    topics = [w.capitalize() for w, _ in sorted(freq.items(), key=lambda x: -x[1])[:6]]

    return {
        "summary": summary,
        "key_insights": insights,
        "main_topics": topics,
    }

# ── Translation ───────────────────────────────────────────────────────────────
def translate_text(text, target_language="kn"):
    log.info("Translating to '%s'...", target_language)
    try:
        chunks = [text[i:i+500] for i in range(0, len(text), 500)]
        result = " ".join(
            t for t in (GoogleTranslator(source="auto",target=target_language).translate(c)
                        for c in chunks) if t)
        log.info("Translation done (%d chars).", len(result))
        return result
    except Exception as e:
        raise RuntimeError(f"Translation to '{target_language}' failed: {e}") from e

# ── edge-tts Neural Voice Map ─────────────────────────────────────────────────
# Structure: {lang_code: {voice_style: edge_tts_voice_name}}
_EDGE_VOICE_MAP = {
    "kn": {
        "male":     "kn-IN-GaganNeural",
        "female":   "kn-IN-SapnaNeural",
        "narrator": "kn-IN-GaganNeural",
        "friendly": "kn-IN-SapnaNeural",
    },
    "hi": {
        "male":     "hi-IN-MadhurNeural",
        "female":   "hi-IN-SwaraNeural",
        "narrator": "hi-IN-MadhurNeural",
        "friendly": "hi-IN-SwaraNeural",
    },
    "ta": {
        "male":     "ta-IN-ValluvarNeural",
        "female":   "ta-IN-PallaviNeural",
        "narrator": "ta-IN-ValluvarNeural",
        "friendly": "ta-IN-PallaviNeural",
    },
    "te": {
        "male":     "te-IN-MohanNeural",
        "female":   "te-IN-ShrutiNeural",
        "narrator": "te-IN-MohanNeural",
        "friendly": "te-IN-ShrutiNeural",
    },
    "ml": {
        "male":     "ml-IN-MidhunNeural",
        "female":   "ml-IN-SobhanaNeural",
        "narrator": "ml-IN-MidhunNeural",
        "friendly": "ml-IN-SobhanaNeural",
    },
    "en": {
        "male":     "en-US-GuyNeural",
        "female":   "en-US-JennyNeural",
        "narrator": "en-US-AriaNeural",
        "friendly": "en-US-AnaNeural",
    },
}
# Rate/pitch modifiers per style (edge-tts SSML parameters)
_EDGE_RATE = {"male": "+0%", "female": "+0%", "narrator": "-10%", "friendly": "+5%"}
_EDGE_PITCH = {"male": "-5Hz", "female": "+5Hz", "narrator": "-10Hz", "friendly": "+0Hz"}


def _resolve_voice(lang, voice):
    """Return the exact edge-tts voice name for (lang, voice_style)."""
    style = voice.lower() if voice else "female"
    lang_map = _EDGE_VOICE_MAP.get(lang, _EDGE_VOICE_MAP["en"])
    resolved = lang_map.get(style, lang_map.get("female", "en-US-JennyNeural"))
    log.info("  Voice resolve: lang='%s' style='%s' → '%s'", lang, style, resolved)
    return resolved


def text_to_speech(text, output_path, lang="kn", voice="male"):
    """Generate TTS using Microsoft edge-tts neural voices."""
    if not text.strip():
        raise RuntimeError("No text for speech synthesis.")

    voice_name = _resolve_voice(lang, voice)
    rate  = _EDGE_RATE.get(voice, "+0%")
    pitch = _EDGE_PITCH.get(voice, "+0Hz")
    log.info("TTS — lang='%s' style='%s' voice='%s' rate=%s pitch=%s",
             lang, voice, voice_name, rate, pitch)

    async def _run():
        communicate = edge_tts.Communicate(text, voice_name, rate=rate, pitch=pitch)
        await communicate.save(output_path)

    try:
        # Run the async edge-tts call in a new event loop (Flask is sync)
        loop = asyncio.new_event_loop()
        loop.run_until_complete(_run())
        loop.close()
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            raise RuntimeError("edge-tts produced an empty file.")
        log.info("  ✅ TTS saved: %s (%.1f KB)",
                 output_path, os.path.getsize(output_path) / 1024)
    except Exception as e:
        raise RuntimeError(f"edge-tts failed for lang='{lang}' voice='{voice_name}': {e}") from e

# ── MP3 → AAC ────────────────────────────────────────────────────────────────
def convert_mp3_to_aac(mp3, aac):
    try:
        subprocess.run(["ffmpeg","-y","-i",mp3,"-c:a","aac","-b:a","192k",aac],
            check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        raise RuntimeError("MP3→AAC failed.") from e

# ── Audio Duration Sync ───────────────────────────────────────────────────────
def _sync_audio_to_video(audio_path, target_duration, output_path):
    """Stretch/compress audio to exactly match video duration using atempo."""
    audio_duration = _get_duration(audio_path)
    if audio_duration <= 0 or target_duration <= 0:
        shutil.copy2(audio_path, output_path)
        return output_path
    ratio = audio_duration / target_duration
    log.info("  Audio sync: audio=%.1fs video=%.1fs ratio=%.3f", audio_duration, target_duration, ratio)
    # atempo supports 0.5–2.0, chain filters for larger ratios
    if 0.5 <= ratio <= 2.0:
        atempo = f"atempo={ratio:.4f}"
    elif ratio > 2.0:
        atempo = f"atempo=2.0,atempo={ratio/2.0:.4f}"
    else:
        atempo = f"atempo=0.5,atempo={ratio*2.0:.4f}"
    try:
        subprocess.run(["ffmpeg","-y","-i",audio_path,"-filter:a",atempo,output_path],
            check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        return output_path
    except Exception:
        shutil.copy2(audio_path, output_path)
        return output_path

# ── Smart Output Production ───────────────────────────────────────────────────
def produce_output_file(source_file, translated_aac, output_path, sync_duration=None):
    probe = _probe_streams(source_file)
    has_video = probe["has_video"]
    log.info("  produce_output_file: has_video=%s source='%s'",
             has_video, os.path.basename(source_file))
    
    # FORENSIC LOGGING
    log.info("  [FORENSIC] Source file: %s", source_file)
    log.info("  [FORENSIC] Source file exists: %s | Size: %d bytes", 
             os.path.exists(source_file), os.path.getsize(source_file) if os.path.exists(source_file) else 0)
    log.info("  [FORENSIC] Output path: %s", output_path)
    log.info("  [FORENSIC] Translated AAC: %s | Size: %d bytes", 
             translated_aac, os.path.getsize(translated_aac) if os.path.exists(translated_aac) else 0)

    if has_video:
        # PATH A — Video dubbing: replace audio in video
        log.info("  → PATH A: VIDEO DUBBING → %s", output_path)
        aac_to_use = translated_aac
        if sync_duration and sync_duration > 0:
            synced = translated_aac.replace(".aac","_synced.aac")
            aac_to_use = _sync_audio_to_video(translated_aac, sync_duration, synced)
        try:
            ffmpeg_cmd = [
                "ffmpeg","-y",
                "-i",source_file,
                "-i",aac_to_use,
                "-c:v","copy","-c:a","aac",
                "-map","0:v:0","-map","1:a:0","-shortest",
                output_path]
            log.info("  [FORENSIC] FFmpeg command: %s", " ".join(ffmpeg_cmd))
            r = subprocess.run(ffmpeg_cmd,
                stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
            if r.returncode != 0:
                log.error("  ffmpeg merge failed:\n%s", r.stderr[-600:])
                raise RuntimeError(f"ffmpeg video merge failed (rc={r.returncode})")
            
            # FORENSIC: Verify output file exists and check size
            if os.path.exists(output_path):
                fsize = os.path.getsize(output_path)
                abs_path = os.path.abspath(output_path)
                log.info("  [FORENSIC] ✅ FFmpeg succeeded - Output file created")
                log.info("  [FORENSIC] Output file: %s | Size: %d bytes", output_path, fsize)
                log.info("  [FORENSIC] Absolute path: %s", abs_path)
            else:
                log.error("  [FORENSIC] ❌ CRITICAL: FFmpeg reported success but output file NOT FOUND")
                log.error("  [FORENSIC] Expected: %s", output_path)
                log.error("  [FORENSIC] Absolute: %s", os.path.abspath(output_path))
                raise RuntimeError(f"FFmpeg created no file at {output_path}")
            
            log.info("  ✅ MP4 dubbed: %s", output_path)
            return output_path, True
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"Video merge failed: {e}") from e
    else:
        # PATH B — Audio only: export MP3
        mp3_out = output_path.replace(".mp4",".mp3")
        log.info("  → PATH B: AUDIO-ONLY → %s", mp3_out)
        try:
            r = subprocess.run(["ffmpeg","-y","-i",translated_aac,
                "-c:a","libmp3lame","-q:a","2",mp3_out],
                stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
            if r.returncode != 0:
                shutil.copy2(translated_aac, mp3_out)
            log.info("  ✅ MP3 audio: %s", mp3_out)
            return mp3_out, False
        except Exception as e:
            raise RuntimeError(f"Audio export failed: {e}") from e

# ── Core Pipeline ─────────────────────────────────────────────────────────────
def _run_pipeline(audio_wav_path, video_source_path, selected_langs, voice="male"):
    os.makedirs(FINAL_OUT_DIR, exist_ok=True)
    t_start = time.time()

    # FORENSIC: Log working environment
    log.info("[FORENSIC] CWD: %s", os.getcwd())
    log.info("[FORENSIC] FINAL_OUT_DIR absolute: %s", os.path.abspath(FINAL_OUT_DIR))
    log.info("[FORENSIC] video_source_path: %s", video_source_path)
    log.info("[FORENSIC] audio_wav_path: %s", audio_wav_path)

    source_probe = _probe_streams(video_source_path)
    has_video    = source_probe["has_video"]
    vid_duration = source_probe.get("duration", 0)
    log.info("Pipeline mode: %s | duration=%.1fs", "VIDEO" if has_video else "AUDIO-ONLY", vid_duration)

    # Save original video/audio to final_outputs for comparison
    timestamp = _get_run_timestamp()
    original_video_path = None
    original_audio_path = None
    
    if has_video:
        original_video_path = os.path.join(FINAL_OUT_DIR, f"original_video_{timestamp}.mp4")
        log.info("Saving original video: %s", original_video_path)
        shutil.copy2(video_source_path, original_video_path)
        log.info("  ✅ Original video saved")
    
    # Save original audio for side-by-side comparison
    original_audio_path = os.path.join(FINAL_OUT_DIR, f"original_audio_{timestamp}.wav")
    log.info("Saving original audio: %s", original_audio_path)
    shutil.copy2(audio_wav_path, original_audio_path)
    log.info("  ✅ Original audio saved")

    # Transcribe (uses cached Whisper model)
    t_transcribe = time.time()
    transcript, segments, detected_lang = transcribe_audio(audio_wav_path)
    transcribe_time = round(time.time() - t_transcribe, 1)
    word_count = len(transcript.split())

    # AI Summary
    ai_data = generate_ai_summary(transcript)

    # Original SRT
    timestamp = _get_run_timestamp()
    orig_srt_path = os.path.join(FINAL_OUT_DIR, f"subtitle_original_{timestamp}.srt")
    generate_srt(segments, orig_srt_path)

    output_files = []
    translations = {}
    srt_files    = {"original": orig_srt_path}
    total_size   = 0

    # ── Per-language work: translate → TTS → merge (parallel across langs) ──
    def _process_one_lang(lang_code):
        """Process a single language — safe to run in a thread."""
        lang_name = AVAILABLE_LANGUAGES.get(lang_code, lang_code)
        log.info("=== START %s (%s) ===", lang_name, lang_code)
        t_lang = time.time()

        tts_mp3   = f"_tts_{lang_code}.mp3"
        aac_path  = f"_translated_{lang_code}.aac"
        # Use timestamp to ensure unique filenames
        final_mp4 = os.path.join(FINAL_OUT_DIR, f"video_{lang_code}_{timestamp}.mp4")

        # Translate
        t0 = time.time()
        translated = translate_text(transcript, target_language=lang_code)
        log.info("  [%s] translate=%.1fs", lang_code, time.time()-t0)

        # TTS (edge-tts neural voice)
        t0 = time.time()
        text_to_speech(translated, tts_mp3, lang=lang_code, voice=voice)
        log.info("  [%s] tts=%.1fs", lang_code, time.time()-t0)

        # MP3 → AAC (needed for FFmpeg muxing)
        convert_mp3_to_aac(tts_mp3, aac_path)
        if os.path.exists(tts_mp3):
            os.remove(tts_mp3)

        # Merge or export
        t0 = time.time()
        actual_path, is_video_out = produce_output_file(
            video_source_path, aac_path, final_mp4,
            sync_duration=vid_duration if has_video else None)
        log.info("  [%s] merge=%.1fs", lang_code, time.time()-t0)
        
        # FORENSIC: Log what was returned
        log.info("  [FORENSIC] Requested output: %s", final_mp4)
        log.info("  [FORENSIC] Actual path returned: %s", actual_path)
        log.info("  [FORENSIC] Match: %s", final_mp4 == actual_path)

        if os.path.exists(aac_path):
            os.remove(aac_path)

        # SRT
        srt_path = os.path.join(FINAL_OUT_DIR, f"subtitle_{lang_code}_{timestamp}.srt")
        generate_translated_srt(segments, translated, srt_path)
        srt_files[lang_code] = srt_path

        fsize = os.path.getsize(actual_path) if os.path.exists(actual_path) else 0
        log.info("=== DONE %s in %.1fs (%.1f KB) ===",
                 lang_name, time.time()-t_lang, fsize/1024)
        
        # FORENSIC: Log the return value
        log.info("  [FORENSIC] Returning for %s:", lang_code)
        log.info("  [FORENSIC]   actual_path: %s", actual_path)
        log.info("  [FORENSIC]   file_size: %d bytes", fsize)
        log.info("  [FORENSIC]   file exists: %s", os.path.exists(actual_path))
        
        return lang_code, translated, actual_path, is_video_out, fsize, srt_path

    # Run languages in parallel (max 3 workers to avoid overwhelming the machine)
    workers = min(len(selected_langs), 3)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_process_one_lang, lc): lc for lc in selected_langs}
        for fut in as_completed(futures):
            try:
                lang_code, translated, actual_path, is_video_out, fsize, srt_path = fut.result()
            except Exception as exc:
                lang_code = futures[fut]
                log.error("Language '%s' failed: %s", lang_code, exc)
                raise RuntimeError(f"Processing failed for language '{lang_code}': {exc}") from exc

            lang_name = AVAILABLE_LANGUAGES.get(lang_code, lang_code)
            translations[lang_code] = translated
            total_size += fsize
            output_files.append({
                "lang_code": lang_code,
                "lang_name": lang_name,
                "file_path": actual_path,
                "file_name": os.path.basename(actual_path),
                "is_video":  is_video_out,
                "file_size": fsize,
                "srt_name":  os.path.basename(srt_path),
            })

    # Restore original language order
    order = {lc: i for i, lc in enumerate(selected_langs)}
    output_files.sort(key=lambda x: order.get(x["lang_code"], 99))

    proc_time = round(time.time() - t_start, 1)
    log.info("✅ Pipeline complete in %.1fs", proc_time)

    return {
        "output_files":     output_files,
        "transcript":       transcript,
        "translations":     translations,
        "detected_language":detected_lang,
        "summary":          ai_data["summary"],
        "key_insights":     ai_data["key_insights"],
        "main_topics":      ai_data["main_topics"],
        "srt_files":        srt_files,
        "source_has_video": has_video,
        "original_video_path": os.path.basename(original_video_path) if original_video_path else None,
        "original_audio_path": os.path.basename(original_audio_path) if original_audio_path else None,
        "run_timestamp":    timestamp,
        "analytics": {
            "duration":        round(vid_duration, 1),
            "words":           word_count,
            "languages":       len(selected_langs),
            "transcribe_time": transcribe_time,
            "processing_time": proc_time,
            "output_size_kb":  round(total_size / 1024, 1),
        },
        "voice": voice,
    }

# ── Error Handling ────────────────────────────────────────────────────────────
def _raise_friendly_error(raw):
    err = (raw or "").lower()
    if "nsig" in err or "sabotage" in err:
        raise RuntimeError("⚠️ YouTube stream extraction blocked. Run: pip install -U yt-dlp")
    if any(k in err for k in ("private","sign in","login","who has the link")):
        raise RuntimeError("⛔ This video is private or requires login.")
    if any(k in err for k in ("age-restrict","confirm your age")):
        raise RuntimeError("🔞 Age-restricted video. Try a different video.")
    if any(k in err for k in ("not available in your country","geo-restrict")):
        raise RuntimeError("🌍 Geo-restricted video.")
    if any(k in err for k in ("video unavailable","has been removed","no longer available")):
        raise RuntimeError("❌ Video unavailable or removed.")
    if any(k in err for k in ("copyright","content id","blocked by")):
        raise RuntimeError("©️ Video blocked due to copyright.")
    if any(k in err for k in ("network","connection","timed out","timeout","ssl")):
        raise RuntimeError("🌐 Network error. Check your connection and retry.")
    log.error("Unclassified yt-dlp error: %s", raw)
    raise RuntimeError(f"Could not download video. (Detail: {str(raw)[:160]})")

# ── Public API ────────────────────────────────────────────────────────────────
def process_youtube_video(youtube_url, selected_langs, voice="male", download_video=True):
    if not youtube_url or not youtube_url.strip():
        raise RuntimeError("Please enter a YouTube URL.")
    if not validate_youtube_url(youtube_url):
        raise RuntimeError("Invalid YouTube URL. Paste a full link (youtube.com/watch?v=...)")
    if not selected_langs:
        raise RuntimeError("Select at least one target language.")

    log.info("=== YouTube pipeline: %s ===", youtube_url)

    # Always download audio for Whisper
    audio_wav = download_audio_from_youtube(youtube_url)

    # Optionally download video for dubbing
    video_file = None
    if download_video:
        log.info("Downloading video stream for dubbing...")
        video_file = download_video_from_youtube(youtube_url)

    source = video_file if video_file else audio_wav
    return _run_pipeline(audio_wav, source, selected_langs, voice=voice)


def process_local_video(local_video_path, selected_langs, voice="male"):
    if not local_video_path or not os.path.isfile(local_video_path):
        raise RuntimeError("Uploaded file not found. Please try again.")
    if not selected_langs:
        raise RuntimeError("Select at least one target language.")

    log.info("=== Local video pipeline: %s ===", local_video_path)
    log.info("  [FORENSIC] Input video file: %s", local_video_path)
    log.info("  [FORENSIC] Input file exists: %s | Size: %d bytes", 
             os.path.exists(local_video_path), os.path.getsize(local_video_path) if os.path.exists(local_video_path) else 0)
    
    audio_wav = extract_audio_from_video(local_video_path, EXTRACTED_WAV)
    return _run_pipeline(audio_wav, local_video_path, selected_langs, voice=voice)