"""Download public agentic-coding sources (pinned revisions) for T33l fp8dec corpus."""
import os, sys
from huggingface_hub import snapshot_download
D = "/tmp/nestquant/33-search/ceiling/fp8dec_src"
SRC = {  # sub: (repo, rev, allow_patterns)
 "r2egym_subset": ("R2E-Gym/R2E-Gym-Subset", "2e8108ff942f24fcb5686badfaf7f9a8808566d5", None),
 "r2egym_sft_traj": ("R2E-Gym/R2EGym-SFT-Trajectories", "63ab4eb37668f8be0104133c21d896bedbcf8404", None),
 "deepswe_kimik2_traj": ("SWE-Factory/DeepSWE-Agent-Kimi-K2-Trajectories-2.8K", "2a2175e291a0606376592af08b64c25afabe72a9", None),
 "swebv": ("princeton-nlp/SWE-bench_Verified", "c104f840cc67f8b6eec6f759ebc8b2693d585d4a", None),
 "swegym": ("SWE-Gym/SWE-Gym", "bb94ed9e39bbeb96a7fcbfb533b80f25a7fd59cb", None),
 "swegym_oh_sft": ("SWE-Gym/OpenHands-SFT-Trajectories", "4aaa5a4a4b5861f4799d2336908760c190ac3b17", None),
 "nebius_sweagent": ("nebius/SWE-agent-trajectories", "68195a1450865274106246d0d0296a1d6807b88e", ["README.md","data/train-0000[0-1]-*"]),
 "swesmith_traj": ("SWE-bench/SWE-smith-trajectories", "08e109b4a59eaeebf80e4675cd125d42e7ac99a4", ["README.md","data/tool-00000-*"]),
 "nebius_rebench_oh": ("nebius/SWE-rebench-openhands-trajectories", "35455389ab51bf5e2306bfd436ef72d0f98bf882", None),
 "tb2_gpt5_traj": ("DCAgent/GPT-5-terminal-bench-2", "b4a75c10478a229016a31349eb849d33089fc932", None),
 "tb2_sonnet45_traj": ("DCAgent/claude-sonnet-4-5-terminal-bench-2", "739abfe6999e998ffcc9bab9a8755951b717841b", None),
 "tb2_glm47_traj": ("DCAgent2/terminal_bench_2__together_ai_zai-org_GLM-4.7_20260203", "0156c8e4940638a6ad7c25fe68b725b5a525199e", None),
 "tb2_kimik25_traj": ("DCAgent2/terminal_bench_2__together_ai_moonshotai_Kimi-K2.5_20260203", "383980b224002f1349208fef0c0e385fcff619f2", None),
 "tb2_deepswe_traj": ("DCAgent2/terminal_bench_2_DeepSWE_Preview_20260502_173733-traces", "edf0b6afe34dc4cc08058b37ffbd91d26ecf0d4d", None),
 "math500": ("HuggingFaceH4/MATH-500", "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be", None),
}
for sub in (sys.argv[1:] or SRC):
    rid, rev, pat = SRC[sub]
    try:
        snapshot_download(rid, repo_type="dataset", revision=rev, local_dir=f"{D}/{sub}", allow_patterns=pat, max_workers=2)
        print("OK", sub, flush=True)
    except Exception as e:
        print("FAIL", sub, type(e).__name__, str(e)[:200], flush=True)
