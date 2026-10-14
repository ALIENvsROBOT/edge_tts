# server.py

from flask import Flask, request, send_file, jsonify, Response
from waitress import serve
from werkzeug.exceptions import HTTPException
from dotenv import load_dotenv
import os
import traceback
import json
import base64
import math
from io import BytesIO
from pathlib import Path

from config import DEFAULT_CONFIGS
from handle_text import prepare_tts_input_with_context
from tts_handler import generate_speech, generate_speech_stream, get_models_formatted, get_voices, get_voices_formatted, SpeechServiceError, SpeechTimeoutError
from utils import getenv_bool, require_api_key, AUDIO_FORMAT_MIME_TYPES, DETAILED_ERROR_LOGGING

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024
load_dotenv()

API_KEY = os.getenv('API_KEY', DEFAULT_CONFIGS["API_KEY"])
PORT = int(os.getenv('PORT', str(DEFAULT_CONFIGS["PORT"])))
TTS_WORKERS = int(os.getenv('TTS_WORKERS', DEFAULT_CONFIGS['TTS_WORKERS']))
if not 1 <= TTS_WORKERS <= 64:
    raise ValueError('TTS_WORKERS must be between 1 and 64.')

DEFAULT_VOICE = os.getenv('DEFAULT_VOICE', DEFAULT_CONFIGS["DEFAULT_VOICE"])
DEFAULT_RESPONSE_FORMAT = os.getenv('DEFAULT_RESPONSE_FORMAT', DEFAULT_CONFIGS["DEFAULT_RESPONSE_FORMAT"])
DEFAULT_SPEED = float(os.getenv('DEFAULT_SPEED', str(DEFAULT_CONFIGS["DEFAULT_SPEED"])))

REMOVE_FILTER = getenv_bool('REMOVE_FILTER', DEFAULT_CONFIGS["REMOVE_FILTER"])
EXPAND_API = getenv_bool('EXPAND_API', DEFAULT_CONFIGS["EXPAND_API"])


@app.errorhandler(SpeechServiceError)
def speech_service_error(error):
    status = 504 if isinstance(error, SpeechTimeoutError) else 502
    return jsonify({'error': str(error)}), status


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})


def read_audio_file(path):
    try:
        return Path(path).read_bytes()
    finally:
        Path(path).unlink(missing_ok=True)

# DEFAULT_MODEL = os.getenv('DEFAULT_MODEL', 'tts-1')

# Currently in "beta" — needs more extensive testing where drop-in replacement warranted
def generate_sse_audio_stream(text, voice, speed):
    """Generator function for SSE streaming with JSON events."""
    audio_stream = generate_speech_stream(text, voice, speed)
    try:
        # Generate streaming audio chunks and convert to SSE format
        for chunk in audio_stream:
            # Base64 encode the audio chunk
            encoded_audio = base64.b64encode(chunk).decode('utf-8')
            
            # Create SSE event for audio delta
            event_data = {
                "type": "speech.audio.delta",
                "audio": encoded_audio
            }
            
            # Format as SSE event
            yield f"data: {json.dumps(event_data)}\n\n"
        
        # Send completion event
        completion_event = {
            "type": "speech.audio.done",
            "usage": {
                "input_tokens": len(text.split()),  # Rough estimate
                "output_tokens": 0,  # Edge TTS doesn't provide this
                "total_tokens": len(text.split())
            }
        }
        yield f"data: {json.dumps(completion_event)}\n\n"
        
    except Exception as e:
        app.logger.warning('SSE streaming failed: %s', type(e).__name__)
        # Send error event
        error_event = {
            "type": "error",
            "error": str(e) if isinstance(e, SpeechServiceError) else 'Speech streaming failed.'
        }
        yield f"data: {json.dumps(error_event)}\n\n"
    finally:
        audio_stream.close()

# OpenAI endpoint format
@app.route('/v1/audio/speech', methods=['POST'])
@app.route('/audio/speech', methods=['POST'])  # Add this line for the alias
@require_api_key
def text_to_speech():
    try:
        data = request.get_json(silent=True)
        if not isinstance(data, dict) or 'input' not in data:
            return jsonify({"error": "Missing 'input' in request body"}), 400

        text = data.get('input')
        if not isinstance(text, str) or not text.strip():
            return jsonify({"error": "'input' must be a non-empty string"}), 400
        if len(text) > 4096:
            return jsonify({"error": "'input' must not exceed 4096 characters"}), 400

        if not REMOVE_FILTER:
            text = prepare_tts_input_with_context(text)
        if not text.strip():
            return jsonify({"error": "'input' contains no speakable text"}), 400

        # model = data.get('model', DEFAULT_MODEL)
        voice = data.get('voice', DEFAULT_VOICE)
        response_format = data.get('response_format', DEFAULT_RESPONSE_FORMAT)
        if not isinstance(voice, str) or not voice.strip():
            return jsonify({"error": "'voice' must be a non-empty string"}), 400
        if not isinstance(response_format, str) or response_format not in AUDIO_FORMAT_MIME_TYPES:
            return jsonify({"error": "Unsupported 'response_format'"}), 400
        raw_speed = data.get('speed', DEFAULT_SPEED)
        try:
            speed = float(raw_speed)
        except (TypeError, ValueError, OverflowError):
            return jsonify({"error": "'speed' must be between 0.25 and 4.0"}), 400
        if isinstance(raw_speed, bool) or not math.isfinite(speed) or not 0.25 <= speed <= 4.0:
            return jsonify({"error": "'speed' must be between 0.25 and 4.0"}), 400
        
        # Check stream format - only "sse" triggers streaming
        stream_format = data.get('stream_format', 'audio')  # 'audio' (default) or 'sse'
        if stream_format not in ('audio', 'sse'):
            return jsonify({"error": "'stream_format' must be 'audio' or 'sse'"}), 400
        
        mime_type = AUDIO_FORMAT_MIME_TYPES.get(response_format, "audio/mpeg")
        
        if stream_format == 'sse':
            # Return SSE streaming response with JSON events
            def generate_sse():
                yield from generate_sse_audio_stream(text, voice, speed)
            
            return Response(
                generate_sse(),
                mimetype='text/event-stream',
                headers={
                    'Content-Type': 'text/event-stream',
                    'Cache-Control': 'no-cache',
                    'X-Accel-Buffering': 'no'  # Disable nginx buffering
                }
            )
        else:
            # Return raw audio data (like OpenAI) - can be piped to ffplay
            output_file_path = generate_speech(text, voice, response_format, speed)
            
            # Read the file and return raw audio data
            audio_data = read_audio_file(output_file_path)
            
            return Response(
                audio_data,
                mimetype=mime_type,
                headers={
                    'Content-Type': mime_type,
                    'Content-Length': str(len(audio_data))
                }
            )
            
    except SpeechServiceError as e:
        return speech_service_error(e)
    except HTTPException as e:
        return jsonify({'error': e.description}), e.code
    except Exception as e:
        if DETAILED_ERROR_LOGGING:
            app.logger.error(f"Error in text_to_speech: {str(e)}\n{traceback.format_exc()}")
        else:
            app.logger.error(f"Error in text_to_speech: {str(e)}")
        # Return a 500 error for unhandled exceptions, which is more standard than 400
        return jsonify({"error": "An internal server error occurred"}), 500

# OpenAI endpoint format
@app.route('/v1/models', methods=['GET', 'POST'])
@app.route('/models', methods=['GET', 'POST'])
@app.route('/v1/audio/models', methods=['GET', 'POST'])
@app.route('/audio/models', methods=['GET', 'POST'])
def list_models():
    return jsonify({"models": get_models_formatted()})

# OpenAI endpoint format
@app.route('/v1/audio/voices', methods=['GET', 'POST'])
@app.route('/audio/voices', methods=['GET', 'POST'])
def list_voices_formatted():
    return jsonify({"voices": get_voices_formatted()})

@app.route('/v1/voices', methods=['GET', 'POST'])
@app.route('/voices', methods=['GET', 'POST'])
@require_api_key
def list_voices():
    specific_language = None

    data = request.args if request.method == 'GET' else request.get_json(silent=True)
    if request.method == 'POST' and data is not None and not isinstance(data, dict):
        return jsonify({'error': 'Request body must be a JSON object'}), 400
    if data and ('language' in data or 'locale' in data):
        specific_language = data.get('language') if 'language' in data else data.get('locale')

    return jsonify({"voices": get_voices(specific_language)})

@app.route('/v1/voices/all', methods=['GET', 'POST'])
@app.route('/voices/all', methods=['GET', 'POST'])
@require_api_key
def list_all_voices():
    return jsonify({"voices": get_voices('all')})

"""
Support for ElevenLabs and Azure AI Speech
    (currently in beta)
"""

# http://localhost:5050/elevenlabs/v1/text-to-speech
# http://localhost:5050/elevenlabs/v1/text-to-speech/en-US-AndrewNeural
@app.route('/elevenlabs/v1/text-to-speech/<voice_id>', methods=['POST'])
@require_api_key
def elevenlabs_tts(voice_id):
    if not EXPAND_API:
        return jsonify({"error": "Endpoint not allowed"}), 404
    
    # Parse the incoming JSON payload
    try:
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict) or 'text' not in payload:
            return jsonify({"error": "Missing 'text' in request body"}), 400
        if not isinstance(payload['text'], str) or not payload['text'].strip():
            return jsonify({'error': "'text' must be a non-empty string"}), 400
    except Exception as e:
        return jsonify({"error": f"Invalid JSON payload: {str(e)}"}), 400

    text = payload['text']

    if not REMOVE_FILTER:
        text = prepare_tts_input_with_context(text)
    if not text.strip():
        return jsonify({'error': 'Input contains no speakable text'}), 400

    voice = voice_id  # ElevenLabs uses the voice_id in the URL

    # Use default settings for edge-tts
    response_format = 'mp3'
    speed = DEFAULT_SPEED  # Optional customization via payload.get('speed', DEFAULT_SPEED)

    # Generate speech using edge-tts
    try:
        output_file_path = generate_speech(text, voice, response_format, speed)
    except SpeechServiceError as e:
        return speech_service_error(e)
    except Exception as e:
        return jsonify({"error": f"TTS generation failed: {str(e)}"}), 500

    # Return the generated audio file
    return send_file(BytesIO(read_audio_file(output_file_path)), mimetype="audio/mpeg", as_attachment=True, download_name="speech.mp3")

# tts.speech.microsoft.com/cognitiveservices/v1
# https://{region}.tts.speech.microsoft.com/cognitiveservices/v1
# http://localhost:5050/azure/cognitiveservices/v1
@app.route('/azure/cognitiveservices/v1', methods=['POST'])
@require_api_key
def azure_tts():
    if not EXPAND_API:
        return jsonify({"error": "Endpoint not allowed"}), 404
    
    # Parse the SSML payload
    try:
        ssml_data = request.data.decode('utf-8')
        if not ssml_data:
            return jsonify({"error": "Missing SSML payload"}), 400

        # Extract the text and voice from SSML
        from xml.etree import ElementTree as ET
        root = ET.fromstring(ssml_data)
        text = root.find('.//{http://www.w3.org/2001/10/synthesis}voice').text
        voice = root.find('.//{http://www.w3.org/2001/10/synthesis}voice').get('name')
        if not isinstance(text, str) or not text.strip() or not voice:
            return jsonify({'error': 'SSML must contain a voice name and non-empty text'}), 400
    except Exception as e:
        return jsonify({"error": f"Invalid SSML payload: {str(e)}"}), 400

    # Use default settings for edge-tts
    response_format = 'mp3'
    speed = DEFAULT_SPEED

    if not REMOVE_FILTER:
        text = prepare_tts_input_with_context(text)
    if not text.strip():
        return jsonify({'error': 'Input contains no speakable text'}), 400

    # Generate speech using edge-tts
    try:
        output_file_path = generate_speech(text, voice, response_format, speed)
    except SpeechServiceError as e:
        return speech_service_error(e)
    except Exception as e:
        return jsonify({"error": f"TTS generation failed: {str(e)}"}), 500

    # Return the generated audio file
    return send_file(BytesIO(read_audio_file(output_file_path)), mimetype="audio/mpeg", as_attachment=True, download_name="speech.mp3")

print(f" Edge TTS (Free Azure TTS) Replacement for OpenAI's TTS API")
print(f" ")
print(f" * Serving OpenAI Edge TTS")
print(f" * Server running on http://localhost:{PORT}")
print(f" * TTS Endpoint: http://localhost:{PORT}/v1/audio/speech")
print(f" ")

if __name__ == '__main__':
    # Each worker owns its asyncio loop, so parallel synthesis cannot block all requests.
    serve(app, host='0.0.0.0', port=PORT, threads=TTS_WORKERS,
          connection_limit=100, max_request_body_size=64 * 1024)
