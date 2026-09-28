"""T25 campaign status at a glance (read-only; never writes). Usage: ./status.sh [--alerts N] [--all]
Driver + watchdog liveness, gate holds, throughput (layers/hour, experts/hour, s/expert, ETA), per-layer state
(non-pending layers; --all for every layer), running workers, HF top-level uploads and the last N alerts."""
import os, sys, json, time, argparse, collections
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ["TZ"] = "Australia/Melbourne"; time.tzset()


def jl(p, d=None):
    try:
        return json.load(open(p))
    except Exception:
        return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alerts", type=int, default=8)
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    c = jl(f"{HERE}/campaign.json"); R = c["root"]
    import nq25_resume as RS, psutil
    drv = RS.driver_running(R)
    wd = None
    try:
        p = int(open(f"{R}/watchdog.pid").read()); wd = p if psutil.pid_exists(p) and "watchdog" in " ".join(psutil.Process(p).cmdline()) else None
    except Exception:
        pass
    cj = jl(f"{R}/campaign.json", {}); st = jl(f"{R}/status.json", {}); S = jl(f"{R}/state.json", {"layers": {}, "workers": {}})
    age = time.time() - os.path.getmtime(f"{R}/status.json") if os.path.exists(f"{R}/status.json") else None
    print(f"T25 campaign {R}  now {time.strftime('%a %d %b %H:%M %Z')}  status.json age {age and round(age)}s")
    print(f"driver {'pid ' + str(drv) if drv else 'NOT RUNNING'}   watchdog {'pid ' + str(wd) if wd else 'NOT RUNNING'}"
          f"{'   STOPPED (manual stop; ./resume.sh to restart)' if os.path.exists(f'{R}/STOPPED') else ''}")
    f = cj.get("frozen", {})
    print(f"config_id {cj.get('config_id')}  encoder {f.get('encoder')}  vision_weight {f.get('vision_weight')}  "
          f"code {cj.get('code', {}).get('git_head')}")
    gates = [(n, c["sched"].get(k)) for n, k in (("MMW_GO", "mmw_go"), ("T23_GO", "t23_go"))]
    fin = f"{f.get('stats_root')}/final"
    nfin = len([x for x in os.listdir(fin) if x.startswith("L") and x.endswith(".json")]) if os.path.isdir(fin) else 0
    print("gates: " + "  ".join(f"{n} {'present ' + open(p).read().strip()[:12] if p and os.path.exists(p) else 'ABSENT'}" for n, p in gates)
          + f"  T19 final markers {nfin}/75")
    for why, Ls in (st.get("holding") or {}).items():
        print(f"  holding {Ls}: {why}")
    print(f"layers {st.get('layers')}  encoded {st.get('layers_encoded')}  first layer start {st.get('first_layer_start')}")
    print(f"throughput: {st.get('layers_per_hour')} layers/h  {st.get('experts_per_hour')} experts/h  "
          f"worker s/expert {st.get('worker_s_per_expert')}  experts {st.get('experts_done')}/{st.get('experts_total')}  ETA {st.get('eta_all_layers')}")
    print(f"running {st.get('running')}  disk out {st.get('disk_out_gb')} GB free {st.get('disk_free_tb')} TB  "
          f"flags {st.get('flags')}  failed chunks {st.get('failed_chunks')}")
    by = collections.defaultdict(list)
    for w in S.get("workers", {}).values():
        if "end" not in w:
            by[w.get("gpu")].append(f"{w['kind']}:L{w['layer']}:{w.get('extra', '')}")
    if by:
        print("workers: " + "  ".join(f"gpu{g}[{', '.join(v)}]" for g, v in sorted(by.items(), key=lambda x: str(x[0]))))
    print(f"{'layer':<6}{'state':<13}{'done':>8}  enc      fin   chk   spot  upload")
    for L in sorted(S["layers"], key=int):
        ly = S["layers"][L]
        if ly["state"] == "pending" and not ly.get("n_done") and not a.all:
            continue
        up = ly.get("upload") or {}
        print(f"L{L:<5}{ly['state']:<13}{ly.get('n_done', 0):>4}/256  {ly.get('encoder', '-'):<8} "
              f"{str((ly.get('finalize') or {}).get('ok', '-')):<5} {str((ly.get('check_decode') or {}).get('ok', '-')):<5} "
              f"{str((ly.get('spot') or {}).get('flag', '-')):<5} {up.get('status', '-')}")
    top = jl(f"{R}/../25-campaign/uploads/top.json") or jl("/tmp/nestquant/25-campaign/uploads/top.json")
    if top:
        print(f"top-level upload: last {top.get('status')} {top.get('commit')} {top.get('time')} UTC")
    try:
        al = open(f"{R}/ALERTS.jsonl").read().splitlines()
    except FileNotFoundError:
        al = []
    print(f"alerts: {len(al)} total" + (f", last {min(a.alerts, len(al))}:" if al else ""))
    for x in al[-a.alerts:]:
        try:
            d = json.loads(x); print(f"  {d['time']} {d['kind']} L{d.get('layer')}: {d['msg'][:160]}")
        except Exception:
            print("  " + x[:180])


if __name__ == "__main__":
    main()
