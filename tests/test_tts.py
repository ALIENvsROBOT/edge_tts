import asyncio
import base64
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import threading
from concurrent.futures import ThreadPoolExecutor
from urllib.request import Request, urlopen
from waitress import create_server
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
# Tests use isolated settings and must never load a developer's .env file.
with patch('dotenv.load_dotenv', return_value=False):
    import server
    import tts_handler as tts
    import utils


class Communicator:
    async def stream(self):
        yield {'type': 'SentenceBoundary', 'text': 'ignored'}
        yield {'type': 'audio', 'data': b'first'}
        yield {'type': 'audio', 'data': b'second'}


class SpeechTests(unittest.TestCase):
    def setUp(self):
        self.communicator = patch.object(tts.edge_tts, 'Communicate', return_value=Communicator())
        self.mock_communicate = self.communicator.start()
        self.addCleanup(self.communicator.stop)

    def test_stream_yields_audio_in_order(self):
        self.assertEqual(list(tts.generate_speech_stream('Hello', 'alloy')), [b'first', b'second'])
        self.assertEqual(self.mock_communicate.call_args.kwargs['voice'], 'en-US-JennyNeural')

    def test_mp3_file_is_closed_and_contains_audio(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(tts.tempfile, 'tempdir', directory):
            path = Path(tts.generate_speech('Hello', 'alloy', 'mp3'))
            self.assertEqual(path.read_bytes(), b'firstsecond')

    def test_generation_failure_cleans_partial_file(self):
        class BrokenCommunicator:
            async def stream(self):
                yield {'type': 'audio', 'data': b'partial'}
                raise tts.edge_tts.exceptions.WebSocketError('upstream failure')
        self.mock_communicate.return_value = BrokenCommunicator()
        with tempfile.TemporaryDirectory() as directory, patch.object(tts.tempfile, 'tempdir', directory):
            with self.assertRaises(tts.SpeechServiceError):
                tts.generate_speech('Hello', 'alloy', 'mp3')
            self.assertEqual(list(Path(directory).iterdir()), [])
        self.assertEqual(self.mock_communicate.call_count, 1)

    def test_transient_failure_retries_before_first_audio(self):
        self.mock_communicate.side_effect = [tts.aiohttp.ClientConnectionError(), Communicator()]
        with patch.object(tts.asyncio, 'sleep', new_callable=AsyncMock):
            self.assertEqual(list(tts.generate_speech_stream('Hello', 'alloy')), [b'first', b'second'])
        self.assertEqual(self.mock_communicate.call_count, 2)

    def test_empty_audio_has_bounded_retries(self):
        class EmptyCommunicator:
            async def stream(self):
                if False:
                    yield
        self.mock_communicate.side_effect = lambda **kwargs: EmptyCommunicator()
        with patch.object(tts.asyncio, 'sleep', new_callable=AsyncMock):
            with self.assertRaises(tts.SpeechServiceError):
                list(tts.generate_speech_stream('Hello', 'alloy'))
        self.assertEqual(self.mock_communicate.call_count, tts.TTS_MAX_RETRIES + 1)

    def test_stream_timeout_uses_one_producer_task(self):
        class SlowCommunicator:
            async def stream(self):
                yield {'type': 'audio', 'data': b'first'}
                await asyncio.sleep(1)
        self.mock_communicate.return_value = SlowCommunicator()
        with patch.object(tts, 'TTS_REQUEST_TIMEOUT', 0.02):
            stream = tts.generate_speech_stream('Hello', 'alloy')
            self.assertEqual(next(stream), b'first')
            with self.assertRaises(tts.SpeechTimeoutError):
                next(stream)

    def test_disconnect_closes_upstream_iterator(self):
        closed = []
        class EndlessCommunicator:
            async def stream(self):
                try:
                    while True:
                        yield {'type': 'audio', 'data': b'audio'}
                finally:
                    closed.append(True)
        self.mock_communicate.return_value = EndlessCommunicator()
        stream = tts.generate_speech_stream('Hello', 'alloy')
        next(stream)
        stream.close()
        self.assertEqual(closed, [True])

    def test_missing_ffmpeg_does_not_mislabel_mp3(self):
        with patch.object(tts, 'is_ffmpeg_installed', return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'FFmpeg is required'):
                tts.generate_speech('Hello', 'alloy', 'wav')
        self.mock_communicate.assert_not_called()

    def test_conversion_failure_cleans_both_files(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(tts.tempfile, 'tempdir', directory):
            with patch.object(tts, 'is_ffmpeg_installed', return_value=True):
                with patch.object(tts.subprocess, 'run', side_effect=subprocess.TimeoutExpired('ffmpeg', 30)):
                    with self.assertRaises(RuntimeError):
                        tts.generate_speech('Hello', 'alloy', 'wav')
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_speed_range_matches_documented_contract(self):
        for speed, rate in ((0.25, '-75%'), (1, '+0%'), (2, '+100%'), (4, '+300%')):
            with self.subTest(speed=speed):
                self.assertEqual(tts.speed_to_rate(speed), rate)
        for speed in (0, -1, 4.1, True, float('nan'), float('inf')):
            with self.subTest(speed=speed), self.assertRaises(ValueError):
                tts.speed_to_rate(speed)


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.client = server.app.test_client()
        auth = patch.object(utils, 'API_KEY', 'test-only-key')
        auth.start()
        self.addCleanup(auth.stop)
        enabled = patch.object(utils, 'REQUIRE_API_KEY', True)
        enabled.start()
        self.addCleanup(enabled.stop)
        self.headers = {'Authorization': 'Bearer test-only-key'}

    def post(self, payload, path='/v1/audio/speech'):
        return self.client.post(path, json=payload, headers=self.headers)

    def test_authentication_is_preserved(self):
        for headers in ({}, {'Authorization': 'Bearer wrong'}):
            self.assertEqual(self.client.post('/v1/audio/speech', json={'input': 'Hi'}, headers=headers).status_code, 401)

    def test_invalid_payloads_return_400_without_synthesis(self):
        payloads = [None, [], ['input'], {}, {'input': None}, {'input': 5},
                    {'input': ''}, {'input': '   '}, {'input': '😀'}, {'input': 'a' * 4097}]
        for key, values in {
            'speed': [None, True, 'invalid', 'nan', 'inf', 0, 4.1, {}],
            'voice': [None, '', []], 'response_format': [None, 'unknown', []],
            'stream_format': [None, 'unknown', []],
        }.items():
            payloads.extend({'input': 'Hello', key: value} for value in values)
        with patch.object(server, 'generate_speech') as generate:
            for payload in payloads:
                with self.subTest(payload=payload):
                    self.assertEqual(self.post(payload).status_code, 400)
            generate.assert_not_called()

    def test_malformed_json_is_400(self):
        response = self.client.post('/v1/audio/speech', data='{', content_type='application/json', headers=self.headers)
        self.assertEqual(response.status_code, 400)

    def test_oversized_request_is_413(self):
        self.assertEqual(self.post({'input': 'a' * 70000}).status_code, 413)

    def test_audio_routes_preserve_bytes_mime_and_cleanup(self):
        for route in ('/v1/audio/speech', '/audio/speech'):
            for format_name, mime in utils.AUDIO_FORMAT_MIME_TYPES.items():
                with self.subTest(route=route, format=format_name), tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / 'audio'
                    path.write_bytes(b'unchanged-audio')
                    with patch.object(server, 'generate_speech', return_value=str(path)):
                        response = self.post({'input': 'Hello', 'response_format': format_name}, route)
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(response.mimetype, mime)
                    self.assertEqual(response.data, b'unchanged-audio')
                    self.assertFalse(path.exists())

    def test_sse_event_contract_and_base64(self):
        with patch.object(tts.edge_tts, 'Communicate', return_value=Communicator()):
            response = self.post({'input': 'Hello', 'stream_format': 'sse'})
            events = [json.loads(line[6:]) for line in response.get_data(as_text=True).splitlines() if line.startswith('data: ')]
        self.assertEqual(response.mimetype, 'text/event-stream')
        self.assertNotIn('Connection', response.headers)
        self.assertEqual([event['type'] for event in events], ['speech.audio.delta', 'speech.audio.delta', 'speech.audio.done'])
        self.assertEqual(b''.join(base64.b64decode(event['audio']) for event in events[:-1]), b'firstsecond')
        self.assertIn('usage', events[-1])

    def test_sse_failure_does_not_emit_success(self):
        def broken(*args):
            raise tts.SpeechServiceError('Speech service is temporarily unavailable.')
            yield
        with patch.object(server, 'generate_speech_stream', side_effect=broken):
            response = self.post({'input': 'Hello', 'stream_format': 'sse'})
            events = response.get_data(as_text=True)
        self.assertIn('"type": "error"', events)
        self.assertNotIn('speech.audio.done', events)

    def test_sse_preserves_legacy_format_parameter_behavior(self):
        with patch.object(tts.edge_tts, 'Communicate', return_value=Communicator()):
            response = self.post({'input': 'Hello', 'stream_format': 'sse', 'response_format': 'wav'})
            self.assertEqual(response.status_code, 200)
            self.assertIn('speech.audio.done', response.get_data(as_text=True))

    def test_expanded_routes_reject_empty_input(self):
        for payload in ([], {'text': None}, {'text': ''}, {'text': '😀'}):
            with self.subTest(payload=payload):
                self.assertEqual(self.post(payload, '/elevenlabs/v1/text-to-speech/en-US-AndrewNeural').status_code, 400)
        ssml = '<speak xmlns="http://www.w3.org/2001/10/synthesis"><voice name="en-US-AvaNeural"></voice></speak>'
        response = self.client.post('/azure/cognitiveservices/v1', data=ssml, content_type='application/ssml+xml', headers=self.headers)
        self.assertEqual(response.status_code, 400)

    def test_upstream_errors_are_502_and_504(self):
        for error, status in ((tts.SpeechServiceError('unavailable'), 502), (tts.SpeechTimeoutError('timeout'), 504)):
            with self.subTest(status=status), patch.object(server, 'generate_speech', side_effect=error):
                self.assertEqual(self.post({'input': 'Hello'}).status_code, status)

    def test_models_and_voice_alias_contracts(self):
        expected = {'models': [{'id': item['id']} for item in tts.model_data]}
        for route in ('/v1/models', '/models', '/v1/audio/models', '/audio/models'):
            self.assertEqual(self.client.get(route).json, expected)
        for route in ('/v1/audio/voices', '/audio/voices'):
            self.assertEqual(self.client.get(route).json, {'voices': tts.get_voices_formatted()})

    def test_voice_service_failure_is_502(self):
        with patch.object(server, 'get_voices', side_effect=tts.SpeechServiceError('unavailable')):
            self.assertEqual(self.client.get('/v1/voices', headers=self.headers).status_code, 502)

    def test_health_is_lightweight(self):
        self.assertEqual(self.client.get('/health').json, {'status': 'ok'})

    def test_wsgi_workers_synthesize_simultaneously(self):
        # A barrier only opens when all four requests are inside synthesis together.
        barrier = threading.Barrier(4, timeout=5)
        def generate(*args):
            barrier.wait()
            with tempfile.NamedTemporaryFile(delete=False) as audio:
                audio.write(b'parallel-audio')
                return audio.name
        http_server = create_server(server.app, host='127.0.0.1', port=0, threads=4,
                                    asyncore_loop_timeout=0.1)
        worker = threading.Thread(target=http_server.run, daemon=True)
        worker.start()
        url = f'http://127.0.0.1:{http_server.effective_port}/v1/audio/speech'
        def call():
            req = Request(url, data=b'{"input":"Hello"}', headers={**self.headers, 'Content-Type': 'application/json'})
            with urlopen(req, timeout=10) as response:
                return response.status, response.read()
        try:
            with patch.object(server, 'generate_speech', side_effect=generate):
                with ThreadPoolExecutor(max_workers=4) as pool:
                    results = list(pool.map(lambda _: call(), range(4)))
            self.assertEqual(results, [(200, b'parallel-audio')] * 4)
        finally:
            http_server.task_dispatcher.shutdown()
            http_server.close()
            worker.join(timeout=2)


if __name__ == '__main__':
    unittest.main()
