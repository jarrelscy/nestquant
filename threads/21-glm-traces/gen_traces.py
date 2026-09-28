"""[NOT RUN: Part A (paid generation) was cancelled by the user 2026-09-28; kept for reference only.]
Part A generation client: prompts.jsonl -> traces.jsonl (append-only checkpoint, one line per finished request).

  python gen_traces.py --model <endpoint model name> [--limit N] [--concurrency 64] [--deadline-min 110]
                       [--budget-usd 40] [--price-in 1.4 --price-out 4.4]

Requests Together chat completions at temperature 1.0 / top_p 0.95 with the prompt's reasoning_effort; TB prompts
carry the bash/read_file/write_file tools. Saves reasoning_content, content, tool_calls, finish_reason, usage and
matched_stop separately; truncated = finish_reason == "length". Already-finished ids are skipped on restart.
Concurrency is AIMD: halve on 429/503, +1 per 10 successes, up to --concurrency. Stops launching new requests at
--deadline-min or when the token-priced spend reaches --budget-usd (in-flight requests are allowed to finish unless
--hard-stop). The API key is read from the environment (TOGETHER_API_KEY) and never logged.
"""
import argparse, asyncio, json, os, random, sys, time
import aiohttp

W = "/tmp/nestquant/21-traces"
URL = "https://api.together.xyz/v1/chat/completions"
MAXTOK = dict(math=24576, competitive_programming=24576, agentic_tb=16384, short=4096)
MAXTOK_DEFAULT = 16384


def max_tokens_for(p, scale):
    return int(MAXTOK.get(p["cat"], MAXTOK_DEFAULT) * scale)


class Limiter:
    def __init__(self, hi, start):
        self.hi, self.cur, self.inflight, self.ok = hi, start, 0, 0
        self.cv = asyncio.Condition()

    async def acquire(self):
        async with self.cv:
            await self.cv.wait_for(lambda: self.inflight < self.cur)
            self.inflight += 1

    async def release(self, throttled):
        async with self.cv:
            self.inflight -= 1
            if throttled:
                self.cur = max(1, self.cur // 2); self.ok = 0
            else:
                self.ok += 1
                if self.ok >= 10 and self.cur < self.hi:
                    self.cur += 1; self.ok = 0
            self.cv.notify_all()


async def one(sess, a, p, lim, stats):
    body = dict(model=a.model, messages=p["messages"], temperature=1.0, top_p=0.95,
                max_tokens=max_tokens_for(p, a.max_tokens_scale), reasoning_effort=p["effort"], stream=False)
    if p.get("tools"):
        body["tools"] = p["tools"]
    t0 = time.time(); err = None
    for attempt in range(8):
        await lim.acquire(); throttled = False
        try:
            async with sess.post(URL, json=body, timeout=aiohttp.ClientTimeout(total=a.timeout)) as r:
                txt = await r.text()
                if r.status == 200:
                    j = json.loads(txt)
                    await lim.release(False)
                    return j, time.time() - t0, attempt, None
                err = f"http {r.status}: {txt[:300]}"
                throttled = r.status in (429, 503)
                if r.status in (400, 401, 403, 404, 422):
                    await lim.release(False)
                    return None, time.time() - t0, attempt, err
        except Exception as e:
            err = f"{type(e).__name__}: {str(e)[:200]}"
        await lim.release(throttled)
        stats["retries"] += 1
        await asyncio.sleep(min(90, 2 ** attempt + random.random() * 3))
    return None, time.time() - t0, attempt, err


def record(p, j, dt, attempts, err, model):
    rec = dict(id=p["id"], cat=p["cat"], src=p["src"], item=p["item"], sample=p["sample"], effort=p["effort"],
               model=model, t=time.time(), latency_s=round(dt, 1), attempts=attempts + 1)
    if j is None:
        rec.update(status="error", error=err); return rec
    ch = j["choices"][0]; m = ch.get("message") or {}
    rec.update(status="ok", finish_reason=ch.get("finish_reason"), truncated=ch.get("finish_reason") == "length",
               matched_stop=ch.get("matched_stop") or ch.get("stop_reason"),
               reasoning_content=m.get("reasoning_content", m.get("reasoning")), content=m.get("content"),
               tool_calls=m.get("tool_calls"), usage=j.get("usage"), response_id=j.get("id"))
    return rec


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", default=f"{W}/prompts.jsonl")
    ap.add_argument("--out", default=f"{W}/traces.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--only-cat", default=None)
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--start-concurrency", type=int, default=16)
    ap.add_argument("--deadline-min", type=float, default=None)
    ap.add_argument("--budget-usd", type=float, default=None)
    ap.add_argument("--price-in", type=float, default=1.4)
    ap.add_argument("--price-out", type=float, default=4.4)
    ap.add_argument("--max-tokens-scale", type=float, default=1.0)
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--hard-stop", action="store_true")
    a = ap.parse_args()
    key = os.environ.get("TOGETHER_API_KEY")
    assert key, "TOGETHER_API_KEY not set"
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out):
            r = json.loads(l)
            if r["status"] == "ok":
                done.add(r["id"])
    P = [json.loads(l) for l in open(a.prompts)]
    P = [p for p in P if p["id"] not in done and (a.only_cat is None or p["cat"] == a.only_cat)]
    if a.limit:
        P = P[:a.limit]
    print(f"todo {len(P)} (done {len(done)})", flush=True)
    stats = dict(ok=0, err=0, trunc=0, tin=0, tout=0, retries=0)
    t_start = time.time(); lim = Limiter(a.concurrency, min(a.start_concurrency, a.concurrency))
    out = open(a.out, "a"); stop = asyncio.Event()

    def spend():
        return stats["tin"] * a.price_in / 1e6 + stats["tout"] * a.price_out / 1e6

    async def worker(q, sess):
        while not q.empty() and not stop.is_set():
            p = q.get_nowait()
            j, dt, att, err = await one(sess, a, p, lim, stats)
            rec = record(p, j, dt, att, err, a.model)
            out.write(json.dumps(rec, ensure_ascii=False) + "\n"); out.flush()
            if rec["status"] == "ok":
                u = rec["usage"] or {}
                stats["ok"] += 1; stats["trunc"] += rec["truncated"]
                stats["tin"] += u.get("prompt_tokens", 0); stats["tout"] += u.get("completion_tokens", 0)
            else:
                stats["err"] += 1
            n = stats["ok"] + stats["err"]
            if n % 25 == 0 or n == len(P):
                el = time.time() - t_start
                print(f"[{el/60:6.1f}m] ok {stats['ok']} err {stats['err']} trunc {stats['trunc']} "
                      f"in {stats['tin']} out {stats['tout']} (~${spend():.2f}) out_tok/s {stats['tout']/max(el,1):.0f} "
                      f"conc {lim.cur}/{lim.inflight} retries {stats['retries']}", flush=True)
            if a.deadline_min and time.time() - t_start > a.deadline_min * 60:
                stop.set()
            if a.budget_usd and spend() >= a.budget_usd:
                stop.set()

    q = asyncio.Queue()
    for p in P:
        q.put_nowait(p)
    conn = aiohttp.TCPConnector(limit=a.concurrency + 8)
    async with aiohttp.ClientSession(connector=conn, headers={"Authorization": f"Bearer {key}"}) as sess:
        tasks = [asyncio.create_task(worker(q, sess)) for _ in range(a.concurrency)]
        if a.hard_stop:
            async def watch():
                await stop.wait()
                for t in tasks: t.cancel()
            asyncio.create_task(watch())
        await asyncio.gather(*tasks, return_exceptions=True)
    el = time.time() - t_start
    print("FINAL", json.dumps(dict(stats, elapsed_min=round(el / 60, 1), token_priced_usd=round(spend(), 2),
                                   stopped_early=stop.is_set())), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
