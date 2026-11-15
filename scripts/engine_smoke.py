"""Manual real-model smoke run. Use a disposable chapter copy.

    python scripts/engine_smoke.py /absolute/chapter /absolute/export

The selected configuration controls inference/network use and may prepare model
files or dependencies. This script does not change settings or supply credentials.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('project', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, QT_QPA_PLATFORM='offscreen', BALLOONTRANS_APP_ROOT=str(root))
    child = subprocess.Popen([sys.executable, '-m', 'ballontranslator.engine'],
                             cwd=root, env=env, stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, text=True)
    request_id = 0
    finished = False

    def receive() -> dict:
        nonlocal finished
        line = child.stdout.readline()
        if not line:
            raise RuntimeError('Engine exited before responding.')
        payload = json.loads(line)
        if 'event' in payload:
            print(json.dumps(payload))
            if payload['event'] == 'run_finished':
                finished = True
        return payload

    def call(method: str, params: dict | None = None) -> dict:
        nonlocal request_id
        request_id += 1
        child.stdin.write(json.dumps(dict(id=request_id, method=method, params=params or {})) + '\n')
        child.stdin.flush()
        while True:
            response = receive()
            if response.get('id') == request_id:
                if 'error' in response:
                    raise RuntimeError(response['error']['message'])
                return response['result']

    try:
        print(json.dumps(call('hello')))
        project = call('open_project', dict(path=str(args.project.resolve())))
        call('run', dict(mode='all'))
        while not finished:
            receive()
        for page in project['pages']:
            page = call('get_page', dict(name=page['name']))
            if page['blocks']:
                block = page['blocks'][0]
                call('update_blocks', dict(page=page['name'], edits=[dict(
                    index=0, translation=block['translation'] + ' ', font_size=block['font_size'])]))
                break
        print(json.dumps(call('export', dict(destination=str(args.destination.resolve()), what='images'))))
        call('shutdown')
    finally:
        child.stdin.close()
        child.wait(timeout=60)
        child.stdout.close()


if __name__ == '__main__':
    main()
