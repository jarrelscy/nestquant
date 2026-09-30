"""Write fp8dec_src/sources.json: name, revision, licence, URL, full counts + GLM-token stats (from stats.json)."""
import glob, json, os, sys
os.environ.setdefault("RAYON_NUM_THREADS", "1")
import pyarrow as pa, pyarrow.parquet as pq
pa.set_cpu_count(1)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import convert as C
D = C.SRC
HF = "https://huggingface.co/datasets/"
META = {  # sub: (name, revision, licence, url, note)
 "r2egym_subset": ("R2E-Gym/R2E-Gym-Subset", "2e8108ff942f24fcb5686badfaf7f9a8808566d5", "apache-2.0", HF + "R2E-Gym/R2E-Gym-Subset", "DeepSWE RL training set (4.5k problems); used as turn-1 prompts"),
 "r2egym_sft_traj": ("R2E-Gym/R2EGym-SFT-Trajectories", "63ab4eb37668f8be0104133c21d896bedbcf8404", "none on card (R2E-Gym code: MIT)", HF + "R2E-Gym/R2EGym-SFT-Trajectories", "Claude-3.5 R2E-scaffold SFT trajs; text <function=> actions; no instance id (mapped via problem statement)"),
 "deepswe_kimik2_traj": ("SWE-Factory/DeepSWE-Agent-Kimi-K2-Trajectories-2.8K", "2a2175e291a0606376592af08b64c25afabe72a9", "mit", HF + "SWE-Factory/DeepSWE-Agent-Kimi-K2-Trajectories-2.8K", "Kimi-K2 on DeepSWE/R2E scaffold; text <function=> actions; no instance id"),
 "swebv": ("princeton-nlp/SWE-bench_Verified", "c104f840cc67f8b6eec6f759ebc8b2693d585d4a", "none on card (SWE-bench code: MIT)", HF + "princeton-nlp/SWE-bench_Verified", "500 problem statements; turn-1 prompts only; mirror SWE-bench/SWE-bench_Verified@78f471bf"),
 "swegym": ("SWE-Gym/SWE-Gym", "bb94ed9e39bbeb96a7fcbfb533b80f25a7fd59cb", "mit", HF + "SWE-Gym/SWE-Gym", "2438 tasks (id map for OH-SFT)"),
 "swegym_oh_sft": ("SWE-Gym/OpenHands-SFT-Trajectories", "4aaa5a4a4b5861f4799d2336908760c190ac3b17", "mit", HF + "SWE-Gym/OpenHands-SFT-Trajectories", "OpenHands CodeAct text <function=> actions; train.success.oss split"),
 "nebius_sweagent": ("nebius/SWE-agent-trajectories", "68195a1450865274106246d0d0296a1d6807b88e", "cc-by-4.0", HF + "nebius/SWE-agent-trajectories", "SWE-agent text actions; downloaded 2/12 shards (13,340 of ~80k rows)"),
 "swesmith_traj": ("SWE-bench/SWE-smith-trajectories", "08e109b4a59eaeebf80e4675cd125d42e7ac99a4", "mit", HF + "SWE-bench/SWE-smith-trajectories", "tool split (native tool_calls), downloaded 1/8 shards"),
 "nebius_rebench_oh": ("nebius/SWE-rebench-openhands-trajectories", "35455389ab51bf5e2306bfd436ef72d0f98bf882", "cc-by-4.0", HF + "nebius/SWE-rebench-openhands-trajectories", "OpenHands native tool_calls + tools schema"),
 "tb2_gpt5_traj": ("DCAgent/GPT-5-terminal-bench-2", "b4a75c10478a229016a31349eb849d33089fc932", "none on card", HF + "DCAgent/GPT-5-terminal-bench-2", "terminus-2 JSON actions on TB2.0 tasks"),
 "tb2_sonnet45_traj": ("DCAgent/claude-sonnet-4-5-terminal-bench-2", "739abfe6999e998ffcc9bab9a8755951b717841b", "none on card", HF + "DCAgent/claude-sonnet-4-5-terminal-bench-2", "terminus-2 JSON actions on TB2.0 tasks"),
 "tb2_glm47_traj": ("DCAgent2/terminal_bench_2__together_ai_zai-org_GLM-4.7_20260203", "0156c8e4940638a6ad7c25fe68b725b5a525199e", "none on card", HF + "DCAgent2/terminal_bench_2__together_ai_zai-org_GLM-4.7_20260203", "terminus-2, <think> + <tool_call> JSON; 3 trials x 89 tasks"),
 "tb2_kimik25_traj": ("DCAgent2/terminal_bench_2__together_ai_moonshotai_Kimi-K2.5_20260203", "383980b224002f1349208fef0c0e385fcff619f2", "none on card", HF + "DCAgent2/terminal_bench_2__together_ai_moonshotai_Kimi-K2.5_20260203", "terminus-2; 3 trials x 89 tasks"),
 "tb2_deepswe_traj": ("DCAgent2/terminal_bench_2_DeepSWE_Preview_20260502_173733-traces", "edf0b6afe34dc4cc08058b37ffbd91d26ecf0d4d", "none on card", HF + "DCAgent2/terminal_bench_2_DeepSWE_Preview_20260502_173733-traces", "EXCLUDED: task instruction missing from trace (R2E scaffold looking for /testbed)"),
 "tb21_registry": ("harborframework/terminal-bench-2.1", "3e235dff6880252a587fa479c09bcb1e16edf2eb", "apache-2.0", HF + "harborframework/terminal-bench-2.1", "mirror of github harbor-framework/terminal-bench-2-1@7131e437; 89 tasks; canary: never in training corpora"),
 "tb20_registry": ("harborframework/terminal-bench-2.0", "f2e8c75e23add71613117eecc9498f53bcd7e04e", "apache-2.0", HF + "harborframework/terminal-bench-2.0", "89 tasks (reference only)"),
 "tb2_zai_verified": ("zai-org/terminal-bench-2-verified", "2d28949d016330454060a57343405330633622b8", "apache-2.0", HF + "zai-org/terminal-bench-2-verified", "Z.ai-fixed TB2.0 (reference only)"),
 "tb_harbor_gh": ("github.com/harbor-framework/terminal-bench (tasks/)", "1dcda8716784493721921c23e4bc7f7d988b4494", "apache-2.0", "https://github.com/harbor-framework/terminal-bench", "continuous TB (dataset.toml says terminal-bench-3; leaderboard runs 'tb-4-0-0'); 67 tasks incl. all 14 test names; archive/ = 90 TB2 tasks"),
 "math500": ("HuggingFaceH4/MATH-500", "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be", "none on card (MATH: MIT)", HF + "HuggingFaceH4/MATH-500", "general/maths slot"),
}


def counts(sub):
    fs = sorted(glob.glob(f"{D}/{sub}/**/*.parquet", recursive=True))
    if sub in ("nebius_sweagent", "nebius_rebench_oh", "swesmith_traj"):
        ids = []
        for f in fs if sub != "swesmith_traj" else [f for f in fs if "/tool-" in f]:
            ids += pq.read_table(f, columns=["instance_id"]).column(0).to_pylist()
        return len(set(ids)), len(ids)
    if sub.startswith("tb2_") and sub != "tb2_zai_verified":
        t = pq.read_table(fs[0], columns=["task"]).column(0).to_pylist()
        return len(set(t)), len(t)
    if sub in ("r2egym_sft_traj", "swegym_oh_sft"):
        ms = pq.read_table(fs[0]).column("messages").to_pylist()
        return len({C.norm_ps(C.issue_key(m)) for m in ms}), len(ms)
    if sub == "deepswe_kimik2_traj":
        ms = [json.loads(l)["messages"] for l in open(glob.glob(f"{D}/{sub}/*.jsonl")[0])]
        return len({C.norm_ps(C.issue_key(m)) for m in ms}), len(ms)
    if sub in ("r2egym_subset", "swegym", "swebv"):
        return sum(pq.ParquetFile(f).metadata.num_rows for f in fs), 0
    if sub == "math500":
        return sum(1 for _ in open(f"{D}/math500/test.jsonl")), 0
    if sub == "tb21_registry":
        return len(os.listdir(f"{D}/tb21_registry/tasks")), 0
    if sub == "tb_harbor_gh":
        return len([x for x in os.listdir(f"{D}/tb_harbor_gh/tasks") if os.path.isdir(f"{D}/tb_harbor_gh/tasks/{x}")]), 0
    if sub in ("tb20_registry", "tb2_zai_verified"):
        return len([x for x in os.listdir(f"{D}/{sub}") if os.path.exists(f"{D}/{sub}/{x}/instruction.md")]), 0
    return None, None


st = json.load(open(f"{D}/stats.json"))["sources"]
out = {}
for sub, (name, rev, lic, url, note) in META.items():
    nt, ntr = counts(sub)
    e = dict(name=name, revision=rev, licence=lic, url=url, local=f"{D}/{sub}", note=note, tasks_total=nt, trajectories_total=ntr)
    for k in [sub, sub + "_t1", sub.replace("r2egym_subset", "r2egym_subset_t1").replace("swebv", "swebv_t1")]:
        if k in st:
            e["glm_stats"] = st[k]
            break
    out[sub] = e
json.dump(out, open(f"{D}/sources.json", "w"), indent=1)
for k, e in out.items():
    g = e.get("glm_stats", {})
    print(f"{k:22s} {e['name'][:60]:60s} {e['revision'][:8]} {e['licence'][:14]:14s} tasks={e['tasks_total']} trajs={e['trajectories_total']} "
          f"turns={g.get('mean_turns')}/{g.get('median_turns')} fulltok={g.get('full_traj_tokens_mean')}/{g.get('full_traj_tokens_median')} >30k={g.get('full_traj_frac_over_30k')} n={g.get('full_traj_sample')}")
