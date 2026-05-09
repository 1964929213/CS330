from __future__ import annotations

import argparse
import base64
import json
import ssl
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

sys.dont_write_bytecode = True

SERVICE_DIR = Path(__file__).resolve().parent
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))

try:  # Support both `python -m service.server` and direct `python service/server.py`.
    from .config import load_config, required_paths, resolve_floorplan_path
    from .localization_engine import Localizer
except ImportError:  # pragma: no cover - exercised by direct script launch.
    from config import load_config, required_paths, resolve_floorplan_path
    from localization_engine import Localizer

CONFIG = load_config()

LOCK = threading.Lock()
LAST_RESULT: dict | None = None
LAST_ERROR: str | None = None
LAST_DURATION_MS: float | None = None
BUSY_DROPPED_COUNT = 0
IS_LOCALIZING = False
LOCALIZER = None


def build_status() -> dict:
    paths = required_paths(CONFIG)
    state = {k: {'path': str(v), 'exists': v.exists()} for k, v in paths.items()}
    floorplan_path = resolve_floorplan_path(CONFIG)
    if floorplan_path is not None:
        state['floorplan'] = {'path': str(floorplan_path), 'exists': floorplan_path.exists()}
    ready = all(item['exists'] for item in state.values())
    return {
        'ready': ready,
        'package': {
            'name': CONFIG.manifest.get('package', {}).get('name'),
            'version': CONFIG.manifest.get('package', {}).get('version'),
            'root': str(CONFIG.paths.package_root),
        },
        'paths': state,
        'floorplan': str(floorplan_path) if floorplan_path is not None else None,
        'last_result': LAST_RESULT,
        'last_error': LAST_ERROR,
        'last_duration_ms': LAST_DURATION_MS,
        'busy_dropped_count': BUSY_DROPPED_COUNT,
        'is_localizing': IS_LOCALIZING,
        'localizer_loaded': LOCALIZER is not None,
    }


def save_query_image_from_data_url(data_url: str) -> Path:
    if not isinstance(data_url, str) or ',' not in data_url:
        raise ValueError('invalid data URL')
    meta, b64 = data_url.split(',', 1)
    if not meta.startswith('data:image/') or ';base64' not in meta:
        raise ValueError('only base64 image data URLs are supported')
    raw = base64.b64decode(b64, validate=True)
    if not raw:
        raise ValueError('image payload is empty')
    CONFIG.paths.tmp_dir.mkdir(parents=True, exist_ok=True)
    out = CONFIG.paths.tmp_dir / 'latest_query.jpg'
    out.write_bytes(raw)
    return out


def ensure_localizer_loaded():
    global LOCALIZER
    if LOCALIZER is None:
        LOCALIZER = Localizer(CONFIG)
    return LOCALIZER


def run_localization(query_image: Path) -> dict:
    global LAST_RESULT, LAST_ERROR, LAST_DURATION_MS, IS_LOCALIZING
    t0 = time.perf_counter()
    IS_LOCALIZING = True
    try:
        loc = ensure_localizer_loaded()
        result = loc.localize(query_image)
        LAST_RESULT = result
        LAST_ERROR = None
        LAST_DURATION_MS = (time.perf_counter() - t0) * 1000.0
        return result
    except Exception as e:
        LAST_ERROR = str(e)
        LAST_DURATION_MS = (time.perf_counter() - t0) * 1000.0
        raise
    finally:
        IS_LOCALIZING = False


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, obj: dict, status: int = 200):
        raw = json.dumps(obj, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(raw)))
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        self.end_headers()
        self.wfile.write(raw)

    def _send_file(self, path: Path, content_type: str):
        raw = path.read_bytes()
        self.send_response(200)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(raw)))
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == '/api/status':
            return self._send_json(build_status())
        if parsed.path == '/api/floorplan':
            p = resolve_floorplan_path(CONFIG)
            if p is None:
                return self._send_json({'error': 'floorplan is not configured'}, status=404)
            p = p.resolve()
            if not p.exists():
                return self._send_json({'error': f'floorplan does not exist: {p}'}, status=404)
            ctype = {
                '.png': 'image/png',
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.webp': 'image/webp',
                '.bmp': 'image/bmp',
            }.get(p.suffix.lower(), 'application/octet-stream')
            return self._send_file(p, ctype)

        rel = 'index.html' if parsed.path in ('', '/') else parsed.path.lstrip('/')
        target = CONFIG.paths.static_dir / rel
        if target.exists() and target.is_file():
            ctype = 'text/plain; charset=utf-8'
            if target.suffix == '.html':
                ctype = 'text/html; charset=utf-8'
            elif target.suffix == '.js':
                ctype = 'application/javascript; charset=utf-8'
            elif target.suffix == '.css':
                ctype = 'text/css; charset=utf-8'
            return self._send_file(target, ctype)
        self.send_error(404)

    def do_POST(self):
        global BUSY_DROPPED_COUNT, LAST_ERROR
        parsed = urlparse(self.path)
        if parsed.path != '/api/localize':
            return self._send_json({'error': 'not found'}, status=404)
        acquired = LOCK.acquire(blocking=False)
        if not acquired:
            BUSY_DROPPED_COUNT += 1
            return self._send_json({'ok': False, 'error': 'busy', 'status': build_status()}, status=429)
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if length <= 0:
                raise ValueError('request body is empty')
            if length > CONFIG.runtime.max_post_bytes:
                raise ValueError(f'request body too large: {length} bytes')
            body = self.rfile.read(length)
            data = json.loads(body.decode('utf-8'))
            image_data_url = data.get('image')
            if not image_data_url:
                raise ValueError('missing image')
            query_image = save_query_image_from_data_url(image_data_url)
            result = run_localization(query_image)
            return self._send_json({'ok': True, 'result': result})
        except Exception as e:
            LAST_ERROR = str(e)
            return self._send_json({'ok': False, 'error': str(e), 'status': build_status()}, status=500)
        finally:
            LOCK.release()


def parse_args():
    parser = argparse.ArgumentParser(description='ALIKED indoor localization service')
    parser.add_argument('--host', default=CONFIG.runtime.host)
    parser.add_argument('--port', type=int, default=CONFIG.runtime.port)
    parser.add_argument('--certfile', type=str, default='')
    parser.add_argument('--keyfile', type=str, default='')
    return parser.parse_args()


def main():
    args = parse_args()
    host = args.host
    port = args.port
    print(f'[INFO] package root = {CONFIG.paths.package_root}')
    print(f'[INFO] reference set = {CONFIG.paths.reference_set_dir}')
    print(f'[INFO] triangulated model = {CONFIG.paths.sfm_model_dir}')
    print(f'[INFO] floorplan = {resolve_floorplan_path(CONFIG)}')
    print('[INFO] localizer = ALIKED + LightGlue, H5-wide retrieval, weighted/topometric XY, PnP heading')
    server = ThreadingHTTPServer((host, port), Handler)

    if args.certfile and args.keyfile:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(certfile=args.certfile, keyfile=args.keyfile)
        server.socket = context.wrap_socket(server.socket, server_side=True)
        print(f'[INFO] serving on https://127.0.0.1:{port}')
    else:
        print(f'[INFO] serving on http://127.0.0.1:{port}')

    server.serve_forever()


if __name__ == '__main__':
    main()
