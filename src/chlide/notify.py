"""Minimal Telegram Bot API client for pipeline notifications.

Vendored verbatim from main_pipeline/scripts/telegram.py (body unchanged below this
docstring), so this repo carries no path dependency on that tree. The import it
replaces was `sys.path.insert(0, '/home/data/yuvaln/main_pipeline/scripts')` -- one of
the hardcoded cross-tree references this migration exists to remove.

STANDARD LIBRARY ONLY, on purpose. The pipeline spans several conda envs (snakemake_env,
af3, lasermpnn, ligandmpnn_env, openmm); making a notification tool depend on `requests`
would mean installing it into whichever env happens to call it, which is a needless way
for a *notifier* to break a run.

Every public function returns bool and never raises. A notifier that can take down its
caller is worse than no notifier -- and the caller here is a watcher sitting next to a
14-hour GPU campaign.

Credentials come from ~/.config/pipeline_telegram.env (mode 0600), falling back to the
environment. The token is deliberately never accepted as a CLI argument: that would put a
live credential into `ps` output and shell history.
"""
from __future__ import annotations

import json
import mimetypes
import os
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Optional, Sequence, Tuple

API_ROOT = "https://api.telegram.org"
ENV_FILE = Path.home() / ".config" / "pipeline_telegram.env"

# Telegram caps an album at 10 items; more than that is rejected outright.
MEDIA_GROUP_MAX = 10
# Bot API upload ceiling is 50 MB per file.
MAX_UPLOAD_BYTES = 50 * 1024 * 1024


class TelegramError(Exception):
    """Internal only -- never escapes a public function."""


def load_credentials(env_file: Path = ENV_FILE) -> Tuple[Optional[str], Optional[str]]:
    """(token, chat_id) from the env file, with real environment taking priority."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    try:
        if env_file.is_file():
            for line in env_file.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key, value = key.strip(), value.strip().strip('"').strip("'")
                if not value:
                    continue
                if key == "TELEGRAM_BOT_TOKEN" and not token:
                    token = value
                elif key == "TELEGRAM_CHAT_ID" and not chat_id:
                    chat_id = value
    except OSError:
        pass
    return token, chat_id


def _encode_multipart(fields: dict, files: Sequence[Tuple[str, Path]]) -> Tuple[bytes, str]:
    """Build a multipart/form-data body. `files` is [(field_name, path), ...]."""
    boundary = f"----pipeline{uuid.uuid4().hex}"
    out = bytearray()
    for name, value in fields.items():
        out += f"--{boundary}\r\n".encode()
        out += f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode()
        out += f"{value}\r\n".encode()
    for name, path in files:
        ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        out += f"--{boundary}\r\n".encode()
        out += (f'Content-Disposition: form-data; name="{name}"; '
                f'filename="{path.name}"\r\n').encode()
        out += f"Content-Type: {ctype}\r\n\r\n".encode()
        out += path.read_bytes()
        out += b"\r\n"
    out += f"--{boundary}--\r\n".encode()
    return bytes(out), f"multipart/form-data; boundary={boundary}"


def _call(method: str, fields: dict, files: Sequence[Tuple[str, Path]] = (),
          *, token: Optional[str] = None, timeout: int = 120,
          retries: int = 3) -> Optional[dict]:
    """POST to the Bot API. Returns the `result` payload, or None on failure.

    Retries on transport errors and on 429/5xx, honouring Telegram's
    `retry_after` when it sends one. A 4xx other than 429 is a permanent
    problem (bad token, bad chat id) and is not retried -- retrying it would
    just delay a message that is never going to be delivered.
    """
    if token is None:
        token, _ = load_credentials()
    if not token:
        return None

    url = f"{API_ROOT}/bot{token}/{method}"
    delay = 2.0
    for attempt in range(1, retries + 1):
        try:
            if files:
                body, ctype = _encode_multipart(fields, files)
            else:
                body = urllib.parse.urlencode(fields).encode()
                ctype = "application/x-www-form-urlencoded"
            req = urllib.request.Request(url, data=body,
                                         headers={"Content-Type": ctype})
            with urllib.request.urlopen(req, timeout=timeout,
                                        context=ssl.create_default_context()) as resp:
                payload = json.loads(resp.read().decode())
            if payload.get("ok"):
                return payload.get("result")
            raise TelegramError(payload.get("description", "unknown error"))

        except urllib.error.HTTPError as exc:
            try:
                detail = json.loads(exc.read().decode())
            except Exception:
                detail = {}
            desc = detail.get("description", str(exc))
            if exc.code == 429:
                wait = float(detail.get("parameters", {}).get("retry_after", delay))
                _log(f"{method}: rate limited, waiting {wait:.0f}s")
                time.sleep(wait)
                continue
            if 400 <= exc.code < 500:
                _log(f"{method}: permanent error {exc.code}: {desc}")
                return None
            _log(f"{method}: HTTP {exc.code} ({desc}), attempt {attempt}/{retries}")
        except Exception as exc:  # noqa: BLE001 - a notifier never raises
            _log(f"{method}: {type(exc).__name__}: {exc}, attempt {attempt}/{retries}")

        if attempt < retries:
            time.sleep(delay)
            delay *= 2
    return None


def _log(message: str) -> None:
    print(f"[telegram] {message}", flush=True)


def _resolve_chat(chat_id: Optional[str]) -> Optional[str]:
    if chat_id:
        return str(chat_id)
    _, from_env = load_credentials()
    return str(from_env) if from_env else None


def send_message(text: str, *, parse_mode: str = "HTML",
                 chat_id: Optional[str] = None) -> bool:
    chat = _resolve_chat(chat_id)
    if not chat:
        _log("no chat id configured; skipping message")
        return False
    fields = {"chat_id": chat, "text": text,
              "disable_web_page_preview": "true"}
    if parse_mode:
        fields["parse_mode"] = parse_mode
    return _call("sendMessage", fields) is not None


def send_photo_album(paths: Sequence[Path], caption: str = "",
                     *, chat_id: Optional[str] = None) -> bool:
    """Send images as one album. Splits automatically past Telegram's limit of 10.

    The caption rides on the first item of the first album, which is how
    Telegram renders a titled group.
    """
    chat = _resolve_chat(chat_id)
    if not chat:
        _log("no chat id configured; skipping album")
        return False
    usable = [p for p in map(Path, paths)
              if p.is_file() and 0 < p.stat().st_size <= MAX_UPLOAD_BYTES]
    if not usable:
        _log("no usable images for album")
        return False

    ok = True
    for offset in range(0, len(usable), MEDIA_GROUP_MAX):
        chunk = usable[offset:offset + MEDIA_GROUP_MAX]
        media, files = [], []
        for idx, path in enumerate(chunk):
            key = f"photo{idx}"
            item = {"type": "photo", "media": f"attach://{key}"}
            if caption and offset == 0 and idx == 0:
                item["caption"] = caption
                item["parse_mode"] = "HTML"
            media.append(item)
            files.append((key, path))
        result = _call("sendMediaGroup",
                       {"chat_id": chat, "media": json.dumps(media)}, files)
        ok = ok and result is not None
    return ok


def send_document(path: Path, caption: str = "",
                  *, chat_id: Optional[str] = None) -> bool:
    chat = _resolve_chat(chat_id)
    if not chat:
        _log("no chat id configured; skipping document")
        return False
    path = Path(path)
    if not path.is_file():
        _log(f"missing document {path}")
        return False
    size = path.stat().st_size
    if size == 0 or size > MAX_UPLOAD_BYTES:
        _log(f"document {path.name} is {size} bytes; outside 1..{MAX_UPLOAD_BYTES}")
        return False
    fields = {"chat_id": chat}
    if caption:
        fields["caption"] = caption
        fields["parse_mode"] = "HTML"
    return _call("sendDocument", fields, [("document", path)]) is not None


def get_me() -> Optional[dict]:
    return _call("getMe", {}, retries=1)


def discover_chat_ids() -> list:
    """Chat ids visible via getUpdates, for first-time setup."""
    updates = _call("getUpdates", {"timeout": "0"}, retries=1) or []
    seen, out = set(), []
    for update in updates:
        for key in ("message", "edited_message", "channel_post", "my_chat_member"):
            chat = (update.get(key) or {}).get("chat")
            if chat and chat.get("id") not in seen:
                seen.add(chat["id"])
                out.append({"id": chat["id"], "type": chat.get("type"),
                            "title": chat.get("title"),
                            "username": chat.get("username"),
                            "first_name": chat.get("first_name")})
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Send a message from the shell, so the harvest can report its own progress.

    `finish_at_iter.sh` runs after the driver and the per-iteration reporter are both dead, so
    without this it has no voice: a harvest could fail at 03:00 and the first anyone knew was
    an absent album. Credentials still come only from ~/.config/pipeline_telegram.env.

        python -m chlide.notify "text"            send
        python -m chlide.notify --photo a.png b.png --caption "..."
    """
    import argparse
    ap = argparse.ArgumentParser(prog='chlide.notify', description=main.__doc__)
    ap.add_argument('text', nargs='*', help='message body; joined with spaces')
    ap.add_argument('--photo', nargs='*', type=Path, default=(),
                    help='send these images as an album instead of / as well as the text')
    ap.add_argument('--caption', default='')
    ap.add_argument('--plain', action='store_true', help='no HTML parsing')
    a = ap.parse_args(argv)
    ok = True
    if a.text:
        ok = send_message(' '.join(a.text), parse_mode=None if a.plain else 'HTML')
    if a.photo:
        existing = [p for p in a.photo if Path(p).is_file()]
        if existing:
            ok = send_photo_album(existing, caption=a.caption) and ok
    # Never fail the caller: a harvest must not die because Telegram is unreachable.
    if not ok:
        _log('send failed; continuing anyway')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
