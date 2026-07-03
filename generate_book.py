"""Phase 2 (basic): stream a book draft from a local LM Studio model.

Takes a processed Takeout session (outputs/<session_id>.json) built by
app.py, turns it into a plain-text user description, and asks the local
LM Studio server to start writing "Harry Potter and <name>". The response
is streamed token-by-token to the terminal.

This is intentionally basic — real summarization ("ficha de usuario") and
web integration are future phases.
"""
import argparse
import json
import sys
from pathlib import Path

import requests

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = Path(__file__).parent
OUTPUT_FOLDER = BASE_DIR / 'outputs'

LM_STUDIO_BASE_URL = "http://127.0.0.1:1234"
MODEL_NAME = "mistralai/ministral-3-3b"  # change to switch models


def build_user_description(data: dict) -> str:
    """Serialize profile + activity items into plain text for the prompt.

    Stand-in for the future LLM summarization step ("ficha de usuario").
    """
    lines = []

    profile = data.get('profile', {})
    for key, value in profile.items():
        lines.append(f"{key}: {value}")

    services = data.get('services', {})
    for service_name, info in services.items():
        items = info.get('items', [])
        if not items:
            continue
        lines.append(f"\n{service_name}:")
        for item in items:
            lines.append(f"- {json.dumps(item, ensure_ascii=False)}")

    return '\n'.join(lines)


def stream_book(user_description: str, user_name: str) -> None:
    prompt = (
        f"With this user information:\n\n{user_description}\n\n"
        f"start writing a book called \"Harry Potter and {user_name}\"."
    )

    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
    }

    with requests.post(
        f"{LM_STUDIO_BASE_URL}/v1/chat/completions",
        json=payload,
        stream=True,
    ) as resp:
        resp.raise_for_status()
        for raw_line in resp.iter_lines():
            if not raw_line:
                continue
            line = raw_line.decode('utf-8')
            if not line.startswith("data: "):
                continue
            data_str = line[len("data: "):]
            if data_str.strip() == "[DONE]":
                break
            chunk = json.loads(data_str)
            content = chunk["choices"][0]["delta"].get("content")
            if content:
                print(content, end='', flush=True)
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Stream a book draft from a local LM Studio model using a processed Takeout session."
    )
    parser.add_argument(
        "session_id",
        help="Session ID of the processed Takeout output (outputs/<session_id>.json)",
    )
    args = parser.parse_args()

    output_path = OUTPUT_FOLDER / f"{args.session_id}.json"
    if not output_path.exists():
        print(f"No output found for session {args.session_id} at {output_path}", file=sys.stderr)
        sys.exit(1)

    data = json.loads(output_path.read_text(encoding='utf-8'))
    user_name = data.get('profile', {}).get('name') or 'You'
    user_description = build_user_description(data)

    print(f"Generating \"Harry Potter and {user_name}\" with model {MODEL_NAME}...\n")
    stream_book(user_description, user_name)


if __name__ == '__main__':
    main()
