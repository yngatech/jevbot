#!/usr/bin/env python3
"""
A web page for looking into Rocky's logs (rocky_harness): each channel's log as Rocky sees it — people's messages,
reactions, edits and deletions, and its turns with their thinking, tool calls and results — what it remembers, and
every turn it has taken, with what each cost, from the daily logs in logs/ (those go back past a log's jumps).

It only reads the files the bot writes, so it shows what's saved: a log a couple of seconds after it changes
(rocky_harness.SAVE_AFTER), a turn once it's finished. The page checks for changes every few seconds.

    python rocky_viewer.py                      # http://127.0.0.1:8350, this checkout's files
    python rocky_viewer.py --root ../jevbot     # another checkout's

The bot also serves it while it runs, when ROCKY_VIEWER_PORT is set in .env. It listens on 127.0.0.1 only: the logs
hold what people said in the server, so it's for looking at on this machine (or through a tunnel that checks who's
asking first), never for opening to the internet as it is.
"""

import argparse
import json
from pathlib import Path

from aiohttp import web

HERE = Path(__file__).resolve().parent
STATIC = HERE / "viewer"
PORT = 8350
TURN_PAGE = 50  # turns per page in the Turns view

# A turn's line in logs/, for the page: what woke it, what it did, and what it cost. history and transcript are left
# out — the log already shows what it saw, and they're most of a line's size.
TURN_FIELDS = ["at", "channel", "channel_id", "message_id", "author", "message", "reply", "sent", "reply_id", "reaction",
               "cost", "byok_cost", "seconds", "requests", "llm_model", "llm_prompt_tokens", "llm_cached_tokens",
               "llm_completion_tokens", "harness_requests", "memory_kept", "rocky_events", "error"]


class Files:
    def __init__(self, root):
        self.root = Path(root)
        self.logs = self.root / "rocky_logs"
        self.memories = self.root / "rocky_memories.json"
        self.traces = self.root / "logs"
        self.read = {}  # a day's file: (bytes read so far, its Rocky turns)

    def channels(self):
        found = []
        for path in sorted(self.logs.glob("*.json")) if self.logs.is_dir() else []:
            if not path.stem.isdigit():
                continue
            try:
                saved = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            events = saved.get("events") or []
            found.append({"id": path.stem, "channel": saved.get("channel"), "space": saved.get("space"),
                          "events": len(events), "last": events[-1].get("at") if events else None})
        return sorted(found, key=lambda c: c["last"] or "", reverse=True)

    def log(self, channel_id):
        try:
            return json.loads((self.logs / f"{int(channel_id)}.json").read_text())
        except (OSError, ValueError):
            return None

    def facts(self, space):
        try:
            return (json.loads(self.memories.read_text()).get(space) or {}).get("facts") or []
        except (OSError, ValueError):
            return []

    # Rocky's turns in the daily logs, oldest first. A day's file only grows, so each read picks up where the last ended.
    def turns(self):
        days = sorted(self.traces.glob("*.jsonl")) if self.traces.is_dir() else []
        for path in days:
            size = path.stat().st_size
            done, turns = self.read.get(path.name, (0, []))
            if size < done:  # rewritten — start it again
                done, turns = 0, []
            if size > done:
                with open(path, "rb") as f:
                    f.seek(done)
                    chunk = f.read(size - done)
                end = chunk.rfind(b"\n") + 1  # a line still being written waits for the next read
                for line in chunk[:end].splitlines():
                    try:
                        t = json.loads(line)
                    except ValueError:
                        continue
                    if t.get("llm") == "rocky":
                        turns.append({k: t[k] for k in TURN_FIELDS if k in t})
                self.read[path.name] = (done + end, turns)
        for name in [n for n in self.read if not (self.traces / n).exists()]:
            del self.read[name]
        return [t for path in days for t in self.read.get(path.name, (0, []))[1]]

    def version(self, channel_id=None):
        paths = [self.memories, *(self.traces.glob("*.jsonl") if self.traces.is_dir() else [])]
        paths += [self.logs / f"{int(channel_id)}.json"] if channel_id else list(self.logs.glob("*.json"))
        stamps = []
        for p in paths:
            try:
                s = p.stat()
                stamps.append(f"{p.name}:{s.st_mtime_ns}:{s.st_size}")
            except OSError:
                pass
        return str(hash(tuple(sorted(stamps))))


def answer(request, body):
    tag = f'"{request.app["files"].version(request.match_info.get("id"))}"'
    if request.headers.get("If-None-Match") == tag:
        return web.Response(status=304, headers={"ETag": tag})
    return web.json_response(body(), headers={"ETag": tag}, dumps=lambda v: json.dumps(v, ensure_ascii=False))


async def index(request):
    return web.FileResponse(STATIC / "index.html")

async def channels(request):
    return answer(request, lambda: {"channels": request.app["files"].channels()})

async def channel(request):
    files, channel_id = request.app["files"], request.match_info["id"]
    saved = files.log(channel_id)
    if saved is None:
        raise web.HTTPNotFound()

    def body():
        # Each turn's line, by when its first response was added to the log (as the log writes the time), so the log
        # can show what each of its turns cost
        turns = {}
        for t in files.turns():
            first = next((e for e in t.get("rocky_events") or [] if e.get("type") == "rocky"), None)
            if str(t.get("channel_id")) == channel_id and first and first.get("at"):
                turns[str(first["at"]).replace(" ", "T")] = t
        return {"id": channel_id, **saved, "facts": files.facts(saved.get("space")), "turns": turns}
    return answer(request, body)

async def turns(request):
    files, channel_id = request.app["files"], request.match_info["id"]
    try:
        page = max(0, int(request.query.get("page", 0)))
    except ValueError:
        raise web.HTTPBadRequest()

    def body():
        found = [t for t in files.turns() if str(t.get("channel_id")) == channel_id][::-1]
        return {"turns": found[page * TURN_PAGE:(page + 1) * TURN_PAGE], "total": len(found), "page_size": TURN_PAGE}
    return answer(request, body)


@web.middleware
async def headers(request, handler):
    response = await handler(request)
    # Nothing on the page comes from anywhere else, and what's said in the chat is only ever shown as text
    response.headers["Content-Security-Policy"] = ("default-src 'none'; script-src 'self'; style-src 'self'; "
                                                   "img-src 'self' data:; connect-src 'self'; base-uri 'none'; "
                                                   "form-action 'none'; frame-ancestors 'none'")
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Cache-Control"] = "no-cache"
    return response


def app(root=HERE):
    a = web.Application(middlewares=[headers])
    a["files"] = Files(root)
    a.router.add_get("/", index)
    a.router.add_get("/api/channels", channels)
    a.router.add_get(r"/api/channels/{id:\d+}", channel)
    a.router.add_get(r"/api/channels/{id:\d+}/turns", turns)
    a.router.add_static("/static/", STATIC)
    return a

# For the bot: serve it alongside, until the bot stops
async def start(port, root=HERE):
    runner = web.AppRunner(app(root), access_log=None)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", port).start()
    return runner


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=PORT, help=f"port on 127.0.0.1 (default {PORT})")
    ap.add_argument("--root", default=HERE, help="the bot's folder, with rocky_logs/ and logs/ (default: this one)")
    args = ap.parse_args()
    print(f"Rocky's logs from {Path(args.root).resolve()} on http://127.0.0.1:{args.port}")
    web.run_app(app(args.root), host="127.0.0.1", port=args.port, print=None, access_log=None)


if __name__ == "__main__":
    main()
