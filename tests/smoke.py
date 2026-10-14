"""Live Docker checks using a test-only key; never loads .env or real credentials."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from http.client import HTTPException
import json
import subprocess
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

TEST_KEY = 'release-smoke-test'
MIME_TYPES = {'mp3': 'audio/mpeg', 'opus': 'audio/ogg', 'aac': 'audio/aac',
              'flac': 'audio/flac', 'wav': 'audio/wav', 'pcm': 'audio/L16'}


def request(base_url, path, payload=None, key=TEST_KEY, content_type='application/json'):
    body = json.dumps(payload).encode() if payload is not None and isinstance(payload, dict) else payload
    headers = {'Authorization': f'Bearer {key}', 'Content-Type': content_type}
    req = Request(base_url + path, data=body, headers=headers)
    try:
        with urlopen(req, timeout=150) as response:
            return response.status, response.headers.get_content_type(), response.read()
    except HTTPError as error:
        return error.code, error.headers.get_content_type(), error.read()


def decode(audio, format_name='mp3'):
    assert len(audio) > 1000, f'Too little {format_name} audio: {len(audio)} bytes'
    command = ['ffmpeg', '-v', 'error']
    if format_name == 'pcm':
        command += ['-f', 's16le', '-ar', '24000', '-ac', '1']
    command += ['-i', 'pipe:0', '-f', 'null', '-']
    subprocess.run(command, input=audio, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, timeout=30)


def speech(base_url, format_name='mp3', voice='alloy', speed=1, streaming=False):
    start = time.monotonic()
    status, mime, body = request(base_url, '/v1/audio/speech', {
        'input': 'Hello. This is the version one point two release verification.',
        'voice': voice, 'response_format': format_name, 'speed': speed,
        'stream_format': 'sse' if streaming else 'audio',
    })
    assert status == 200, f'Synthesis failed: HTTP {status}: {body[:200]!r}'
    if streaming:
        assert mime == 'text/event-stream', mime
        events = [json.loads(line[6:]) for line in body.decode().splitlines() if line.startswith('data: ')]
        assert events[-1]['type'] == 'speech.audio.done', events[-1]
        assert all(event['type'] == 'speech.audio.delta' for event in events[:-1]), events
        body = b''.join(base64.b64decode(event['audio']) for event in events[:-1])
    else:
        assert mime.lower() == MIME_TYPES[format_name].lower(), mime
    decode(body, format_name)
    elapsed = time.monotonic() - start
    print(f'PASS {voice} {format_name} {"SSE" if streaming else "audio"}: {len(body)} bytes, {elapsed:.2f}s', flush=True)
    return elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-url', default='http://127.0.0.1:5050')
    parser.add_argument('--all-formats', action='store_true')
    parser.add_argument('--all-voices', action='store_true')
    args = parser.parse_args()
    base_url = args.base_url.rstrip('/')
    for attempt in range(30):
        try:
            assert request(base_url, '/health')[0] == 200
            break
        except (URLError, OSError, HTTPException, AssertionError):
            if attempt == 29:
                raise
            time.sleep(1)
    assert request(base_url, '/v1/audio/speech', {'input': 'Hi'}, key='wrong')[0] == 401
    assert request(base_url, '/v1/audio/speech', {'input': None})[0] == 400
    status, _, body = request(base_url, '/v1/voices/all')
    assert status == 200 and len(json.loads(body)['voices']) > 100
    print(f'PASS live voices: {len(json.loads(body)["voices"])}', flush=True)
    status, _, body = request(base_url, '/v1/audio/voices')
    aliases = [voice['id'] for voice in json.loads(body)['voices']]
    assert status == 200 and len(aliases) == 11

    # Real network requests overlap on the deployed WSGI server.
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(speech, base_url, 'mp3', 'alloy'),
                   pool.submit(speech, base_url, 'wav', 'echo'),
                   pool.submit(speech, base_url, 'mp3', 'nova', 1, True),
                   pool.submit(speech, base_url, 'mp3', 'en-US-AvaNeural')]
        durations = [future.result() for future in futures]
    wall_time = time.monotonic() - started
    print(f'PASS 4 concurrent clients: {wall_time:.2f}s wall time, {sum(durations):.2f}s combined client time', flush=True)
    if args.all_formats:
        for format_name in MIME_TYPES:
            speech(base_url, format_name)
        for speed in (0.25, 4):
            speech(base_url, speed=speed)
    if args.all_voices:
        for voice in aliases:
            speech(base_url, voice=voice)
    # Expanded provider routes must still return decodable MP3 files.
    for path, payload, content_type in (
        ('/audio/speech', {'input': 'Alias endpoint verification.'}, 'application/json'),
        ('/elevenlabs/v1/text-to-speech/en-US-AndrewNeural', {'text': 'Eleven Labs route verification.'}, 'application/json'),
        ('/azure/cognitiveservices/v1', b'<speak xmlns="http://www.w3.org/2001/10/synthesis"><voice name="en-US-AvaNeural">Azure route verification.</voice></speak>', 'application/ssml+xml'),
    ):
        status, mime, body = request(base_url, path, payload, content_type=content_type)
        assert status == 200 and mime == 'audio/mpeg', (path, status, mime)
        decode(body)
        print(f'PASS {path}', flush=True)
    print('All live Docker checks passed.', flush=True)


if __name__ == '__main__':
    main()
