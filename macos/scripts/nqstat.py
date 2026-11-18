#!/usr/bin/env python3
"""Show the NestQuant server's live status: requests, current progress, decode and prefill speed, GPU and memory.

usage: nqstat.py            one snapshot
       nqstat.py -w [secs]  refresh every 2 s (or the given interval) until Ctrl-C
       nqstat.py --json     raw metrics

Environment: NQ_HOST (127.0.0.1), NQ_PORT (8000), NQ_API_KEY (if the server requires one).
"""

import json
import os
import sys
import time
import urllib.request


def fetch():
    url = f"http://{os.environ.get('NQ_HOST', '127.0.0.1')}:{os.environ.get('NQ_PORT', '8000')}/metrics"
    headers = {}
    if os.environ.get('NQ_API_KEY'):
        headers['Authorization'] = f"Bearer {os.environ['NQ_API_KEY']}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=5) as response:
        return json.loads(response.read())


def duration(seconds):
    seconds = int(seconds or 0)
    if seconds >= 3600:
        return f'{seconds // 3600}h{seconds % 3600 // 60:02d}m'
    if seconds >= 60:
        return f'{seconds // 60}m{seconds % 60:02d}s'
    return f'{seconds}s'


def number(value, unit=''):
    return '-' if value is None else f'{value:,}{unit}'


def render(m):
    lines = []
    r, gpu, mem = m['requests'], m.get('gpu') or {}, m.get('memory') or {}
    lines.append(f"NestQuant server   up {duration(m['uptime_seconds'])}   models: {', '.join(m['models'])}")
    st = m.get('settings') or {}
    lines.append(f"Requests   running {r['running']}   queued {r['queued']}   served {r['served']}   failed {r['failed']}"
                 + (f"   cached conversations {st['cached_conversations']} of {st['cache_slots']}"
                    if st.get('cache_slots') is not None else ''))
    c = m.get('current')
    if c:
        if c['phase'] == 'prefill':
            lines.append(f"Current    prefill of {number(c['prompt_tokens'])} prompt tokens, {duration(c['elapsed_seconds'])} so far"
                         + ('   (thinking)' if c.get('thinking') else ''))
        else:
            lines.append(f"Current    decoding: {number(c['generated_tokens'])} tokens at {number(c.get('decode_tok_s'))} tok/s"
                         f"   (prompt {number(c['prompt_tokens'])}, prefill {c.get('prefill_seconds')} s,"
                         f" {duration(c['elapsed_seconds'])} total)" + ('   thinking' if c.get('thinking') else ''))
    else:
        lines.append('Current    idle')
    last = m.get('last')
    if last:
        lines.append(f"Last       prefill {number(last.get('prefilled_tokens'))} new tokens in {last.get('prefill_seconds')} s"
                     f" ({number(last.get('prefill_tok_s'))} tok/s), decode {number(last['completion_tokens'])} tokens"
                     f" at {number(last.get('decode_tok_s'))} tok/s, {last['elapsed_seconds']} s total")
    rec, tot = m['recent'], m['totals']
    lines.append(f"Recent {rec['requests']:<3} prefill avg {number(rec.get('prefill_tok_s'))} tok/s, decode avg"
                 f" {number(rec.get('decode_tok_s'))} tok/s")
    lines.append(f"Totals     {number(tot['completion_tokens'])} tokens generated at {number(tot.get('decode_tok_s'))} tok/s,"
                 f" {number(tot.get('cache_reuse_percent'))}% of prompt tokens from cache")
    cache = m.get('cache')
    if cache:
        sizes = ', '.join(f"{t:,}" for t in cache.get('cached_tokens', []))
        lines.append(f"Cache      {cache['cached']} of {cache['slots']} conversations cached" + (f" ({sizes} tokens)" if sizes else '')
                     + f", {cache.get('eviction_misses', 0)} re-read after eviction"
                     f" ({number(round(cache.get('eviction_miss_seconds', 0)))} s lost)"
                     + (f", shared start {', '.join(f'{n:,}' for n in cache['shared_starts'])} tokens"
                        f" ({cache.get('shared_start_gib')} GiB)" if cache.get('shared_starts') else ''))
    modes = rec.get('modes')
    if modes:
        lines.append('Reuse      ' + ', '.join(f'{k} {modes[k]}' for k in ('extend', 'rollback', 'replay', 'fresh') if k in modes)
                     + ' (last 50 requests)')
    if rec.get('time_to_first_token_seconds') is not None or rec.get('large_prefill_tok_s') is not None:
        lines.append(f"Prefill    first token avg {number(rec.get('time_to_first_token_seconds'))} s,"
                     f" large prompts avg {number(rec.get('large_prefill_tok_s'))} tok/s")
    if m.get('advice'):
        lines.append(f"Advice     {m['advice']}")
    lines.append(f"GPU        {number(gpu.get('utilization_percent'))}% busy, {number(gpu.get('memory_in_use_gib'))} GiB in use")
    lines.append(f"Memory     server {number(mem.get('process_gib'))} GiB (peak {number(mem.get('process_peak_gib'))}),"
                 f" system available {number(mem.get('system_available_gib'))} of {number(mem.get('system_total_gib'))} GiB,"
                 f" pressure {mem.get('pressure', '-')}")
    return '\n'.join(lines)


def main():
    args = sys.argv[1:]
    if '--json' in args:
        print(json.dumps(fetch(), indent=1))
        return
    if '-w' in args:
        rest = args[args.index('-w') + 1:]
        interval = float(rest[0]) if rest else 2.0
        try:
            while True:
                try:
                    text = render(fetch())
                except OSError as error:
                    text = f'NestQuant server not reachable: {error}'
                sys.stdout.write('\033[2J\033[H' + text + '\n')
                sys.stdout.flush()
                time.sleep(interval)
        except KeyboardInterrupt:
            return
    try:
        print(render(fetch()))
    except OSError as error:
        sys.exit(f'NestQuant server not reachable: {error}')


if __name__ == '__main__':
    main()
