"""Manage one local dot2dot web process; standard library only."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / '.run'
PID_FILE = STATE / 'web.json'
LOG_FILE = STATE / 'web.log'


def running_process():
    try:
        saved = json.loads(PID_FILE.read_text())
        pid = int(saved['pid'])
        if pid <= 1:
            return None
        inspection = subprocess.run(['ps', '-p', str(pid), '-o', 'stat=', '-o', 'command='],
                                    capture_output=True, text=True)
        if inspection.returncode and inspection.stderr.strip():
            raise RuntimeError('Cannot inspect the server process: ' + inspection.stderr.strip())
        process = inspection.stdout.strip()
        # Never signal an unrelated process after a stale PID has been reused.
        if not process or process.startswith('Z') or not all(
                token in process for token in ('uvicorn', 'dot2dot.web.app:app', '--app-dir', str(ROOT))):
            return None
        return saved
    except (OSError, ValueError, KeyError, TypeError):
        return None


def stop():
    saved = running_process()
    if not saved:
        PID_FILE.unlink(missing_ok=True)
        print('dot2dot is already stopped.')
        return
    os.kill(saved['pid'], signal.SIGTERM)
    deadline = time.monotonic() + 10
    while running_process() and time.monotonic() < deadline:
        time.sleep(0.1)
    if running_process():
        raise RuntimeError('Server has not stopped yet; run make down again shortly.')
    PID_FILE.unlink(missing_ok=True)
    print('Stopped dot2dot.')


def start(port):
    saved = running_process()
    if saved:
        print(f"dot2dot is already running at http://127.0.0.1:{saved['port']} (PID {saved['pid']}).")
        return
    with socket.socket() as probe:
        if probe.connect_ex(('127.0.0.1', port)) == 0:
            raise RuntimeError(f'Port {port} is in use by an unmanaged process. Stop it first or use PORT=8001.')
    command = [sys.executable, '-m', 'uvicorn', 'dot2dot.web.app:app', '--app-dir', str(ROOT),
               '--host', '127.0.0.1', '--port', str(port)]
    with LOG_FILE.open('ab') as log:
        log.write(b'\n--- Starting dot2dot ---\n')
        log.flush()
        process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    PID_FILE.write_text(json.dumps({'pid': process.pid, 'port': port}))
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        if process.poll() is not None:
            PID_FILE.unlink(missing_ok=True)
            raise RuntimeError(f'Server exited during startup. Run make logs ({LOG_FILE}).')
        try:
            with urlopen(f'http://127.0.0.1:{port}/api/config', timeout=0.5) as response:
                if response.status == 200:
                    print(f'Started dot2dot at http://127.0.0.1:{port} (PID {process.pid}).')
                    print('Use make down to stop it; make logs to view output.')
                    return
        except (URLError, TimeoutError, OSError):
            time.sleep(0.2)
    stop()
    raise RuntimeError('Server did not become ready within 20 seconds. Run make logs.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['up', 'down', 'restart', 'status', 'logs'])
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error('port must be between 1 and 65535')
    STATE.mkdir(exist_ok=True)
    # Serializes concurrent make commands so two starts cannot race.
    with (STATE / 'web.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if args.action == 'logs':
            if LOG_FILE.exists():
                print('\n'.join(LOG_FILE.read_text(errors='replace').splitlines()[-80:]))
            else:
                print('No logs yet. Start the app with make up.')
        elif args.action == 'status':
            saved = running_process()
            print(f"Running at http://127.0.0.1:{saved['port']} (PID {saved['pid']})." if saved else 'dot2dot is stopped.')
        else:
            if args.action in ('down', 'restart'):
                stop()
            if args.action in ('up', 'restart'):
                start(args.port)


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, OSError) as error:
        print(f'error: {error}', file=sys.stderr)
        sys.exit(1)
