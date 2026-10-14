## v1.2

Fix speech generation with Edge TTS 7.3.1, including Microsoft's updated service authentication. Update and pin the direct Python runtime dependencies.

- Preserve speech routes, OpenAI voice aliases, audio MIME types, model/voice list shapes, and SSE audio/completion events.
- Fix SSE iteration and Windows temporary-file handling. Decode-check all six audio formats; preserve the existing AAC-in-MP4 container and support 24 kHz mono signed 16-bit little-endian PCM correctly.
- Serve requests with eight independent Waitress worker threads by default. Add bounded upstream retries before the first audio chunk, a total request deadline, and cleanup on failures or disconnected streams.
- Return 400 for invalid speech input, 413 for oversized requests, 502 for upstream failures, and 504 for upstream timeouts. SSE failures retain the error-event response type.
- Add Windows/Linux regression tests and live Docker speech, format, alias, and concurrent-client checks before image publication. Check the live service daily; open dependency update PRs automatically.

Images: `ghcr.io/alienvsrobot/edge_tts:v1.2` and `ghcr.io/alienvsrobot/edge_tts:1.2`. Main also publishes `ghcr.io/alienvsrobot/edge_tts:latest` after successful checks.

Optional settings: `TTS_WORKERS=8` (1–64), `TTS_REQUEST_TIMEOUT=120` seconds, `TTS_MAX_RETRIES=2` (0–5). Existing API key and voice settings continue to work.

Microsoft's online service remains an external dependency. Failed checks block publication; updates require review and passing checks before release.
