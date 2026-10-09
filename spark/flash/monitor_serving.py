"""Read routing telemetry without issuing inference requests or exposing API keys."""
import argparse
import json
import time
from pathlib import Path


def snapshot(path):
    d = json.loads(Path(path).read_text())
    totals = d.get('total', {}).get('decode_committed', {})
    keys = ('token_rows', 'hit_rate', 'mean_hot_of_8', 'hot_salience_coverage',
            'desired_salience_coverage', 'desired_but_cold_rate')
    io = d.get('io', {})
    prefetch = io.get('prefetch', {})
    return {
        'age_seconds': round(time.time() - d['updated_at'], 2),
        'scope': 'decode totals since server start; TPS is scheduler estimate',
        'decode': {k: totals.get(k) for k in keys},
        'io': {k: io.get(k) for k in ('dt_s', 'ssd_GBps', 'delivered_GBps', 'drive_read_ms')},
        'prefetch': {k: prefetch.get(k) for k in (
            'mode', 'maximum_pending', 'maximum_demotions',
            'estimated_committed_tps', 'estimated_records_per_second', 'pending',
            'outstanding_reads', 'mailbox_pending', 'pending_demotions',
            'desired_not_yet_admitted')},
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('path')
    p.add_argument('--interval', type=float, default=10)
    p.add_argument('--once', action='store_true')
    args = p.parse_args()
    if args.interval <= 0:
        p.error('interval must be positive')
    while True:
        try:
            print(json.dumps(snapshot(args.path)), flush=True)
        except (OSError, ValueError, KeyError) as e:
            print(json.dumps({'telemetry_error': str(e)}), flush=True)
        if args.once:
            return
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
