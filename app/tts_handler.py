# tts_handler.py

import edge_tts
import asyncio
import tempfile
import subprocess
import os
import math
import aiohttp
from pathlib import Path

from utils import DETAILED_ERROR_LOGGING
from config import DEFAULT_CONFIGS

# Language default (environment variable)
DEFAULT_LANGUAGE = os.getenv('DEFAULT_LANGUAGE', DEFAULT_CONFIGS["DEFAULT_LANGUAGE"])
TTS_REQUEST_TIMEOUT = int(os.getenv('TTS_REQUEST_TIMEOUT', DEFAULT_CONFIGS['TTS_REQUEST_TIMEOUT']))
TTS_MAX_RETRIES = int(os.getenv('TTS_MAX_RETRIES', DEFAULT_CONFIGS['TTS_MAX_RETRIES']))
if TTS_REQUEST_TIMEOUT <= 0 or not 0 <= TTS_MAX_RETRIES <= 5:
    raise ValueError('TTS_REQUEST_TIMEOUT must be positive; TTS_MAX_RETRIES must be between 0 and 5.')


class SpeechServiceError(RuntimeError):
    """The upstream speech service could not complete the request."""


class SpeechTimeoutError(SpeechServiceError):
    """The upstream speech request exceeded its deadline."""

# OpenAI voice names mapped to edge-tts equivalents
voice_mapping = {
    'alloy': 'en-US-JennyNeural',
    'ash': 'en-US-AndrewNeural',
    'ballad': 'en-GB-ThomasNeural',
    'coral': 'en-AU-NatashaNeural',
    'echo': 'en-US-GuyNeural',
    'fable': 'en-GB-SoniaNeural',
    'nova': 'en-US-AriaNeural',
    'onyx': 'en-US-EricNeural',
    'sage': 'en-US-JennyNeural',
    'shimmer': 'en-US-EmmaNeural',
    'verse': 'en-US-BrianNeural',
}

model_data = [
        {"id": "tts-1", "name": "Text-to-speech v1"},
        {"id": "tts-1-hd", "name": "Text-to-speech v1 HD"},
        {"id": "gpt-4o-mini-tts", "name": "GPT-4o mini TTS"}
    ]

def is_ffmpeg_installed():
    """Check if FFmpeg is installed and accessible."""
    try:
        subprocess.run(['ffmpeg', '-version'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5)
        return True
    except (subprocess.CalledProcessError, OSError, subprocess.TimeoutExpired):
        return False

async def _generate_audio_stream(text, voice, speed):
    """Generate streaming TTS audio using edge-tts."""
    # Determine if the voice is an OpenAI-compatible voice or a direct edge-tts voice
    edge_tts_voice = voice_mapping.get(voice, voice)  # Use mapping if in OpenAI names, otherwise use as-is
    
    # Convert speed to SSML rate format
    speed_rate = speed_to_rate(speed)
    
    sent_audio = False
    try:
        async with asyncio.timeout(TTS_REQUEST_TIMEOUT):
            for attempt in range(TTS_MAX_RETRIES + 1):
                try:
                    communicator = edge_tts.Communicate(
                        text=text, voice=edge_tts_voice, rate=speed_rate,
                        connect_timeout=10, receive_timeout=30,
                    )
                    async for chunk in communicator.stream():
                        if chunk['type'] == 'audio':
                            sent_audio = True
                            yield chunk['data']
                    if not sent_audio:
                        raise edge_tts.exceptions.NoAudioReceived('No audio was received.')
                    return
                except (aiohttp.ClientError, edge_tts.exceptions.EdgeTTSException, TimeoutError) as exc:
                    # Retrying after yielding audio would duplicate speech for SSE clients.
                    if sent_audio or attempt == TTS_MAX_RETRIES:
                        if isinstance(exc, TimeoutError):
                            raise SpeechTimeoutError('Speech service timed out.') from exc
                        raise SpeechServiceError('Speech service is temporarily unavailable.') from exc
                    await asyncio.sleep(0.5 * (2 ** attempt))
    except TimeoutError as exc:
        raise SpeechTimeoutError('Speech service timed out.') from exc

def generate_speech_stream(text, voice, speed=1.0):
    """Generate streaming speech audio (synchronous wrapper)."""
    # Keep one event loop alive while Flask consumes the async audio iterator.
    with asyncio.Runner() as runner:
        queue = asyncio.Queue(maxsize=1)

        async def produce():
            try:
                async for chunk in _generate_audio_stream(text, voice, speed):
                    await queue.put((chunk, None))
            except Exception as exc:
                await queue.put((None, exc))
            else:
                await queue.put((None, None))

        # One producer task keeps aiohttp and timeout contexts on the same task.
        producer = runner.get_loop().create_task(produce())
        try:
            while True:
                chunk, error = runner.run(queue.get())
                if error is not None:
                    raise error
                if chunk is None:
                    break
                yield chunk
        finally:
            producer.cancel()

async def _generate_audio(text, voice, response_format, speed):
    """Generate TTS audio and optionally convert to a different format."""
    speed_to_rate(speed)
    if response_format != 'mp3' and not is_ffmpeg_installed():
        raise RuntimeError('FFmpeg is required for non-MP3 audio formats.')

    # Generate the TTS output in mp3 format first
    temp_mp3_file_obj = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    temp_mp3_path = temp_mp3_file_obj.name
    # Windows requires closing the handle before edge-tts opens this path.
    temp_mp3_file_obj.close()

    # Generate the MP3 file
    try:
        with open(temp_mp3_path, 'wb') as audio_file:
            async for chunk in _generate_audio_stream(text, voice, speed):
                audio_file.write(chunk)
    except BaseException:
        Path(temp_mp3_path).unlink(missing_ok=True)
        raise

    # If the requested format is mp3, return the generated file directly
    if response_format == "mp3":
        return temp_mp3_path

    # Create a new temporary file for the converted output
    converted_file_obj = tempfile.NamedTemporaryFile(delete=False, suffix=f".{response_format}")
    converted_path = converted_file_obj.name
    converted_file_obj.close() # Close file object, ffmpeg will write to the path

    # Build the FFmpeg command
    ffmpeg_command = [
        "ffmpeg",
        "-i", temp_mp3_path,  # Input file path
        "-c:a", {
            "aac": "aac",
            "mp3": "libmp3lame",
            "wav": "pcm_s16le",
            "pcm": "pcm_s16le",
            "opus": "libopus",
            "flac": "flac"
        }.get(response_format, "aac"),  # Default to AAC if unknown
    ]

    if response_format == "pcm":
        ffmpeg_command.extend(["-ar", "24000", "-ac", "1"])
    elif response_format != "wav":
        ffmpeg_command.extend(["-b:a", "192k"])

    ffmpeg_command.extend([
        "-f", {
            "aac": "mp4",  # Preserve the existing AAC-in-MP4 response container.
            "mp3": "mp3",
            "wav": "wav",
            "pcm": "s16le",
            "opus": "ogg",
            "flac": "flac"
        }.get(response_format, response_format),  # Default to matching format
        "-y",  # Overwrite without prompt
        converted_path  # Output file path
    ])

    try:
        # Run FFmpeg command and ensure no errors occur
        subprocess.run(ffmpeg_command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as e:
        # Clean up potentially created (but incomplete) converted file
        Path(converted_path).unlink(missing_ok=True)
        # Clean up the original mp3 file as well, since conversion failed
        Path(temp_mp3_path).unlink(missing_ok=True)
        
        if DETAILED_ERROR_LOGGING and isinstance(e, subprocess.CalledProcessError):
            error_message = f"FFmpeg error during audio conversion. Command: '{' '.join(e.cmd)}'. Stderr: {e.stderr.decode('utf-8', 'ignore')}"
            print(error_message) # Log for server-side diagnosis
        else:
            error_message = f"FFmpeg error during audio conversion: {e}"
            print(error_message) # Log a simpler message
        raise RuntimeError(f"FFmpeg error during audio conversion: {e}") # The raised error will still have details via e

    # Clean up the original temporary file (original mp3) as it's now converted
    Path(temp_mp3_path).unlink(missing_ok=True)

    return converted_path

def generate_speech(text, voice, response_format, speed=1.0):
    return asyncio.run(_generate_audio(text, voice, response_format, speed))

def get_models():
    return model_data

def get_models_formatted():
    return [{ "id": x["id"] } for x in model_data]

def get_voices_formatted():
    return [{ "id": k, "name": v } for k, v in voice_mapping.items()]

async def _get_voices(language=None):
    # List all voices, filter by language if specified
    try:
        async with asyncio.timeout(TTS_REQUEST_TIMEOUT):
            all_voices = await edge_tts.list_voices()
    except TimeoutError as exc:
        raise SpeechTimeoutError('Voice service timed out.') from exc
    except (aiohttp.ClientError, edge_tts.exceptions.EdgeTTSException) as exc:
        raise SpeechServiceError('Voice service is temporarily unavailable.') from exc
    language = language or DEFAULT_LANGUAGE  # Use default if no language specified
    filtered_voices = [
        {"name": v['ShortName'], "gender": v['Gender'], "language": v['Locale']}
        for v in all_voices if language == 'all' or language is None or v['Locale'] == language
    ]
    return filtered_voices

def get_voices(language=None):
    return asyncio.run(_get_voices(language))

def speed_to_rate(speed: float) -> str:
    """
    Converts a multiplicative speed value to the edge-tts "rate" format.
    
    Args:
        speed (float): The multiplicative speed value (e.g., 1.5 for +50%, 0.5 for -50%).
    
    Returns:
        str: The formatted "rate" string (e.g., "+50%" or "-50%").
    """
    if isinstance(speed, bool) or not math.isfinite(speed) or not 0.25 <= speed <= 4.0:
        raise ValueError("Speed must be between 0.25 and 4.0 (inclusive).")

    # Convert speed to percentage change
    percentage_change = (speed - 1) * 100

    # Format with a leading "+" or "-" as required
    return f"{percentage_change:+.0f}%"
