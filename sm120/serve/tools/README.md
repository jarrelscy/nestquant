# serve tools

These are client-side scripts for the NQ serve on port 8001. They read the API key from `/home/jarrelscy/homeassistant/.env`.

- `nq_step.py`: single-stream decode benchmark that reports tok/s, accepted tokens per step and ms/step from the spec-decode counters.
- `ab.sh`, `ab2.sh`, `kab2.sh`: reboot the serve once per config and append the decode benchmark results to ab.jsonl.
- `needle.py`: needle-in-a-haystack test at up to ~1M tokens.
- `coh.py`: coherence check.
- `soak.py`: long-running soak.
- `kld.py`: dump served logprobs and compare two dumps (KLD, ppl, top-1).
- `verify_repack.py`: check a repack directory against its manifest hashes.
- `nq_tb40_start2.sh`: boot the prod serve at 1M context and start the tb4 watcher.
