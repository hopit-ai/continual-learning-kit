# Combined launch inventory

Message-3 joins only on Opus SEND by 2026-11-02; otherwise send core alone. No launch authority.

For v6b use `build-v6b` and README-v6b: its own SEND record authorizes PD/SP;
K/CR and REF-m3 rows in this full combined example are reused, never resubmitted.
Non-m3 REF rows stay on the first configured core host. All watcher attempts go
to shared `control_root/utilisation/<group>/`; REPORT-text uses `text_out/<nonce>/`.
SP panel engines are declared to the pool: qualified Qwen3-8B uses K=3; an
unqualified engine falls back to K=1 without changing scoring numerics.

Generated from every source task. Runtime locations stay in the local JSON; this table contains portable IDs only.

| Job | Group | Prerequisites | Dependency | gpus | nodes | cpus | share node | commands / concurrency (PD K≤2) | GPU-h estimate / basis |
|---|---|---|---|---|---|---|---|---|---|
| PF-node-0 | core | none | first wave | 0 | 1 | 4 | yes | CPU | [0, 0]; CPU/handoff; no GPU compute |
| PF-ARC | core | INPUTS | first wave | 0 | 1 | 4 | yes | CPU | [0, 0]; CPU/handoff; no GPU compute |
| PF-tier-0 | preflight | none | first wave | 0 | 1 | 8 | yes | CPU | [0, 0]; CPU dependency/cgroup checks; UUID checked at GPU admission |
| PF-tier-1 | preflight | none | first wave | 0 | 1 | 8 | yes | CPU | [0, 0]; CPU dependency/cgroup checks; UUID checked at GPU admission |
| INPUTS | core | none | afterok | 0 | 1 | 4 | yes | 1 | [0, 0]; CPU/handoff; no GPU compute |
| M2 | core | INPUTS | afterok | 0 | 1 | 4 | yes | 1 | [0, 0]; CPU/handoff; no GPU compute |
| S1 | core | INPUTS | afterok | 0 | 1 | 4 | yes | 1 | [0, 0]; CPU/handoff; no GPU compute |
| M3b-stage | core | INPUTS, PF-ARC | afterok | 0 | 1 | 8 | yes | 3 | None; core measured scoring/preparation/qualification allocation pending |
| M3 | core | M2 | afterok | 0 | 1 | 4 | yes | 1 | [0, 0]; CPU/handoff; no GPU compute |
| RA0 | core | M2 | afterok | 0 | 1 | 4 | yes | 1 | [0, 0]; CPU/handoff; no GPU compute |
| S2 | core | S1, M2 | afterok | 0 | 1 | 4 | yes | 1 | [0, 0]; CPU/handoff; no GPU compute |
| M3b-gpu | core | M3b-stage | afterok | 1 | 1 | 8 | yes | 3 / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| RA101-stage | core | RA0 | afterok | 0 | 1 | 8 | yes | 12 | [0, 0]; CPU/handoff; no GPU compute |
| RA102-stage | core | RA0 | afterok | 0 | 1 | 8 | yes | 12 | [0, 0]; CPU/handoff; no GPU compute |
| RA103-stage | core | RA0 | afterok | 0 | 1 | 8 | yes | 17 | [0, 0]; CPU/handoff; no GPU compute |
| M5-stage | core | M3 | afterok | 0 | 1 | 8 | yes | 1 | None; core measured scoring/preparation/qualification allocation pending |
| S3 | core | S2 | afterok | 0 | 1 | 4 | yes | 1 | [0, 0]; CPU/handoff; no GPU compute |
| S5 | core | S2 | afterok | 0 | 1 | 4 | yes | 1 | [0, 0]; CPU/handoff; no GPU compute |
| S4-D-finqa-alone-s101-stage | core | S2 | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| S4-S-finqa-alone-s101-stage | core | S2 | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| S4-F-finqa-alone-s101-stage | core | S2 | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| S4-R-finqa-alone-s101-stage | core | S2 | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| S4-S-finqa-alone-s102-stage | core | S2 | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| S4-F-finqa-alone-s102-stage | core | S2 | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| S4-R-finqa-alone-s102-stage | core | S2 | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| S4-D-finqa-alone-s102-stage | core | S2 | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| S4-S-finqa-alone-s103-stage | core | S2 | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| S4-F-finqa-alone-s103-stage | core | S2 | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| S4-R-finqa-alone-s103-stage | core | S2 | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| S4-D-finqa-alone-s103-stage | core | S2 | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| S4-D-R-finqa-alone-s101-stage | core | S2 | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| S4-D-R-finqa-alone-s102-stage | core | S2 | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| S4-D-R-finqa-alone-s103-stage | core | S2 | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| FV-core-101-stage | core | M3 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| FV-core-102-stage | core | M3 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| FV-core-103-stage | core | M3 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| FV-reprobe-101-stage | core | M3 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| FV-reprobe-102-stage | core | M3 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| FV-reprobe-103-stage | core | M3 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| FV-x-tier1-101-stage | core | M3 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| FV-x-tier1-102-stage | core | M3 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| FV-x-tier1-103-stage | core | M3 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| FV-x-tier2.b-101-stage | core | M3 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| FV-x-tier2.b-102-stage | core | M3 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| FV-x-tier2.b-103-stage | core | M3 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| FV-x-tier2.b-p2-101-stage | core | M3 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| FV-x-tier2.b-p2-102-stage | core | M3 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| FV-x-tier2.b-p2-103-stage | core | M3 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| FV-x-tier2.incremental-101-stage | core | M3 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| FV-x-tier2.incremental-102-stage | core | M3 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| FV-x-tier2.incremental-103-stage | core | M3 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| FV-x-tier2.incremental-p2-101-stage | core | M3 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| FV-x-tier2.incremental-p2-102-stage | core | M3 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| FV-x-tier2.incremental-p2-103-stage | core | M3 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| FV-x-tier2.stack-drift-101-stage | core | M3 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| FV-x-tier2.stack-drift-102-stage | core | M3 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| FV-x-tier2.stack-drift-103-stage | core | M3 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| FV-x-tier2.stack-plasticity-101-stage | core | M3 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| FV-x-tier2.stack-plasticity-102-stage | core | M3 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| FV-x-tier2.stack-plasticity-103-stage | core | M3 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| M3b | core | M3b-gpu | afterok | 0 | 1 | 8 | yes | 3 | None; core measured scoring/preparation/qualification allocation pending |
| RA101-gpu | core | RA101-stage | afterok | 1 | 1 | 8 | yes | 12 / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| RA102-gpu | core | RA102-stage | afterok | 1 | 1 | 8 | yes | 12 / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| RA103-gpu | core | RA103-stage | afterok | 1 | 1 | 8 | yes | 17 / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| M5-gpu | core | M5-stage | afterok | 2 | 1 | 8 | yes | 1 | None; core measured scoring/preparation/qualification allocation pending |
| S6 | core | S5 | afterok | 0 | 1 | 4 | yes | 1 | [0, 0]; CPU/handoff; no GPU compute |
| S4-D-finqa-alone-s101-gpu | core | S4-D-finqa-alone-s101-stage | afterok | 1 | 1 | 8 | yes | 2 / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| S4-S-finqa-alone-s101-gpu | core | S4-S-finqa-alone-s101-stage | afterok | 1 | 1 | 8 | yes | 2 / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| S4-F-finqa-alone-s101-gpu | core | S4-F-finqa-alone-s101-stage | afterok | 1 | 1 | 8 | yes | 2 / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| S4-R-finqa-alone-s101-gpu | core | S4-R-finqa-alone-s101-stage | afterok | 1 | 1 | 8 | yes | 2 / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| S4-S-finqa-alone-s102-gpu | core | S4-S-finqa-alone-s102-stage | afterok | 1 | 1 | 8 | yes | 2 / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| S4-F-finqa-alone-s102-gpu | core | S4-F-finqa-alone-s102-stage | afterok | 1 | 1 | 8 | yes | 2 / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| S4-R-finqa-alone-s102-gpu | core | S4-R-finqa-alone-s102-stage | afterok | 1 | 1 | 8 | yes | 2 / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| S4-D-finqa-alone-s102-gpu | core | S4-D-finqa-alone-s102-stage | afterok | 1 | 1 | 8 | yes | 2 / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| S4-S-finqa-alone-s103-gpu | core | S4-S-finqa-alone-s103-stage | afterok | 1 | 1 | 8 | yes | 2 / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| S4-F-finqa-alone-s103-gpu | core | S4-F-finqa-alone-s103-stage | afterok | 1 | 1 | 8 | yes | 2 / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| S4-R-finqa-alone-s103-gpu | core | S4-R-finqa-alone-s103-stage | afterok | 1 | 1 | 8 | yes | 2 / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| S4-D-finqa-alone-s103-gpu | core | S4-D-finqa-alone-s103-stage | afterok | 1 | 1 | 8 | yes | 2 / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| S4-D-R-finqa-alone-s101-gpu | core | S4-D-R-finqa-alone-s101-stage | afterok | 1 | 1 | 8 | yes | 2 / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| S4-D-R-finqa-alone-s102-gpu | core | S4-D-R-finqa-alone-s102-stage | afterok | 1 | 1 | 8 | yes | 2 / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| S4-D-R-finqa-alone-s103-gpu | core | S4-D-R-finqa-alone-s103-stage | afterok | 1 | 1 | 8 | yes | 2 / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| FV-core-101-gpu | core | FV-core-101-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| FV-core-102-gpu | core | FV-core-102-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| FV-core-103-gpu | core | FV-core-103-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| FV-reprobe-101-gpu | core | FV-reprobe-101-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| FV-reprobe-102-gpu | core | FV-reprobe-102-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| FV-reprobe-103-gpu | core | FV-reprobe-103-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| FV-x-tier1-101-gpu | core | FV-x-tier1-101-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| FV-x-tier1-102-gpu | core | FV-x-tier1-102-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| FV-x-tier1-103-gpu | core | FV-x-tier1-103-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| FV-x-tier2.b-101-gpu | core | FV-x-tier2.b-101-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| FV-x-tier2.b-102-gpu | core | FV-x-tier2.b-102-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| FV-x-tier2.b-103-gpu | core | FV-x-tier2.b-103-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| FV-x-tier2.b-p2-101-gpu | core | FV-x-tier2.b-p2-101-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| FV-x-tier2.b-p2-102-gpu | core | FV-x-tier2.b-p2-102-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| FV-x-tier2.b-p2-103-gpu | core | FV-x-tier2.b-p2-103-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| FV-x-tier2.incremental-101-gpu | core | FV-x-tier2.incremental-101-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| FV-x-tier2.incremental-102-gpu | core | FV-x-tier2.incremental-102-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| FV-x-tier2.incremental-103-gpu | core | FV-x-tier2.incremental-103-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| FV-x-tier2.incremental-p2-101-gpu | core | FV-x-tier2.incremental-p2-101-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| FV-x-tier2.incremental-p2-102-gpu | core | FV-x-tier2.incremental-p2-102-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| FV-x-tier2.incremental-p2-103-gpu | core | FV-x-tier2.incremental-p2-103-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| FV-x-tier2.stack-drift-101-gpu | core | FV-x-tier2.stack-drift-101-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| FV-x-tier2.stack-drift-102-gpu | core | FV-x-tier2.stack-drift-102-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| FV-x-tier2.stack-drift-103-gpu | core | FV-x-tier2.stack-drift-103-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| FV-x-tier2.stack-plasticity-101-gpu | core | FV-x-tier2.stack-plasticity-101-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| FV-x-tier2.stack-plasticity-102-gpu | core | FV-x-tier2.stack-plasticity-102-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| FV-x-tier2.stack-plasticity-103-gpu | core | FV-x-tier2.stack-plasticity-103-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| M-pools | core | M3, M3b | afterok | 0 | 1 | 4 | yes | 1 | [0, 0]; CPU/handoff; no GPU compute |
| RA101 | core | RA101-gpu | afterok | 0 | 1 | 8 | yes | 12 | [0, 0]; CPU/handoff; no GPU compute |
| RA102 | core | RA102-gpu | afterok | 0 | 1 | 8 | yes | 12 | [0, 0]; CPU/handoff; no GPU compute |
| RA103 | core | RA103-gpu | afterok | 0 | 1 | 8 | yes | 17 | [0, 0]; CPU/handoff; no GPU compute |
| M5 | core | M5-gpu | afterok | 0 | 1 | 8 | yes | 1 | [10.0, 10.0]; core 500-item bounded 27B target allowance; charge once |
| S4-D-finqa-alone-s101 | core | S4-D-finqa-alone-s101-gpu | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| S4-S-finqa-alone-s101 | core | S4-S-finqa-alone-s101-gpu | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| S4-F-finqa-alone-s101 | core | S4-F-finqa-alone-s101-gpu | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| S4-R-finqa-alone-s101 | core | S4-R-finqa-alone-s101-gpu | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| S4-S-finqa-alone-s102 | core | S4-S-finqa-alone-s102-gpu | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| S4-F-finqa-alone-s102 | core | S4-F-finqa-alone-s102-gpu | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| S4-R-finqa-alone-s102 | core | S4-R-finqa-alone-s102-gpu | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| S4-D-finqa-alone-s102 | core | S4-D-finqa-alone-s102-gpu | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| S4-S-finqa-alone-s103 | core | S4-S-finqa-alone-s103-gpu | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| S4-F-finqa-alone-s103 | core | S4-F-finqa-alone-s103-gpu | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| S4-R-finqa-alone-s103 | core | S4-R-finqa-alone-s103-gpu | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| S4-D-finqa-alone-s103 | core | S4-D-finqa-alone-s103-gpu | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| S4-D-R-finqa-alone-s101 | core | S4-D-R-finqa-alone-s101-gpu | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| S4-D-R-finqa-alone-s102 | core | S4-D-R-finqa-alone-s102-gpu | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| S4-D-R-finqa-alone-s103 | core | S4-D-R-finqa-alone-s103-gpu | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| S7-base-stage | core | S6 | afterok | 0 | 1 | 8 | yes | 6 | [0, 0]; CPU/handoff; no GPU compute |
| S7-t1-D-s101-40-stage | core | S6 | afterok | 0 | 1 | 8 | yes | 8 | [0, 0]; CPU/handoff; no GPU compute |
| S7-t1-D-s102-40-stage | core | S6 | afterok | 0 | 1 | 8 | yes | 8 | [0, 0]; CPU/handoff; no GPU compute |
| S7-t1-D-s103-40-stage | core | S6 | afterok | 0 | 1 | 8 | yes | 8 | [0, 0]; CPU/handoff; no GPU compute |
| S7-t1-F-s101-40-stage | core | S6 | afterok | 0 | 1 | 8 | yes | 8 | [0, 0]; CPU/handoff; no GPU compute |
| S7-t1-F-s102-40-stage | core | S6 | afterok | 0 | 1 | 8 | yes | 8 | [0, 0]; CPU/handoff; no GPU compute |
| S7-t1-F-s103-40-stage | core | S6 | afterok | 0 | 1 | 8 | yes | 8 | [0, 0]; CPU/handoff; no GPU compute |
| FV-core-101 | core | FV-core-101-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| FV-core-102 | core | FV-core-102-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| FV-core-103 | core | FV-core-103-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| FV-reprobe-101 | core | FV-reprobe-101-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| FV-reprobe-102 | core | FV-reprobe-102-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| FV-reprobe-103 | core | FV-reprobe-103-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| FV-x-tier1-101 | core | FV-x-tier1-101-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| FV-x-tier1-102 | core | FV-x-tier1-102-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| FV-x-tier1-103 | core | FV-x-tier1-103-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| FV-x-tier2.b-101 | core | FV-x-tier2.b-101-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| FV-x-tier2.b-102 | core | FV-x-tier2.b-102-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| FV-x-tier2.b-103 | core | FV-x-tier2.b-103-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| FV-x-tier2.b-p2-101 | core | FV-x-tier2.b-p2-101-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| FV-x-tier2.b-p2-102 | core | FV-x-tier2.b-p2-102-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| FV-x-tier2.b-p2-103 | core | FV-x-tier2.b-p2-103-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| FV-x-tier2.incremental-101 | core | FV-x-tier2.incremental-101-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| FV-x-tier2.incremental-102 | core | FV-x-tier2.incremental-102-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| FV-x-tier2.incremental-103 | core | FV-x-tier2.incremental-103-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| FV-x-tier2.incremental-p2-101 | core | FV-x-tier2.incremental-p2-101-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| FV-x-tier2.incremental-p2-102 | core | FV-x-tier2.incremental-p2-102-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| FV-x-tier2.incremental-p2-103 | core | FV-x-tier2.incremental-p2-103-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| FV-x-tier2.stack-drift-101 | core | FV-x-tier2.stack-drift-101-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| FV-x-tier2.stack-drift-102 | core | FV-x-tier2.stack-drift-102-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| FV-x-tier2.stack-drift-103 | core | FV-x-tier2.stack-drift-103-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| FV-x-tier2.stack-plasticity-101 | core | FV-x-tier2.stack-plasticity-101-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| FV-x-tier2.stack-plasticity-102 | core | FV-x-tier2.stack-plasticity-102-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| FV-x-tier2.stack-plasticity-103 | core | FV-x-tier2.stack-plasticity-103-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| R0 | core | M-pools | afterok | 0 | 1 | 4 | yes | 1 | [0, 0]; CPU/handoff; no GPU compute |
| RA-collect | core | RA101, RA102, RA103 | afterany | 0 | 1 | 4 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| M4-101-stage | core | M3, FV-core-101 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| M4-102-stage | core | M3, FV-core-102 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| M4-103-stage | core | M3, FV-core-103 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| S7-base-gpu | core | S7-base-stage | afterok | 1 | 1 | 8 | yes | 6 / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| S7-t1-D-s101-40-gpu | core | S7-t1-D-s101-40-stage | afterok | 1 | 1 | 8 | yes | 8 / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| S7-t1-D-s102-40-gpu | core | S7-t1-D-s102-40-stage | afterok | 1 | 1 | 8 | yes | 8 / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| S7-t1-D-s103-40-gpu | core | S7-t1-D-s103-40-stage | afterok | 1 | 1 | 8 | yes | 8 / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| S7-t1-F-s101-40-gpu | core | S7-t1-F-s101-40-stage | afterok | 1 | 1 | 8 | yes | 8 / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| S7-t1-F-s102-40-gpu | core | S7-t1-F-s102-40-stage | afterok | 1 | 1 | 8 | yes | 8 / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| S7-t1-F-s103-40-gpu | core | S7-t1-F-s103-40-stage | afterok | 1 | 1 | 8 | yes | 8 / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| R-incoming-101-stage | core | R0, FV-reprobe-101 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| R-incoming-102-stage | core | R0, FV-reprobe-102 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| R-incoming-103-stage | core | R0, FV-reprobe-103 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| M4-101-gpu | core | M4-101-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| M4-102-gpu | core | M4-102-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| M4-103-gpu | core | M4-103-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| S7-base | core | S7-base-gpu | afterok | 0 | 1 | 8 | yes | 6 | [0, 0]; CPU/handoff; no GPU compute |
| S7-t1-D-s101-40 | core | S7-t1-D-s101-40-gpu | afterok | 0 | 1 | 8 | yes | 8 | [0, 0]; CPU/handoff; no GPU compute |
| S7-t1-D-s102-40 | core | S7-t1-D-s102-40-gpu | afterok | 0 | 1 | 8 | yes | 8 | [0, 0]; CPU/handoff; no GPU compute |
| S7-t1-D-s103-40 | core | S7-t1-D-s103-40-gpu | afterok | 0 | 1 | 8 | yes | 8 | [0, 0]; CPU/handoff; no GPU compute |
| S7-t1-F-s101-40 | core | S7-t1-F-s101-40-gpu | afterok | 0 | 1 | 8 | yes | 8 | [0, 0]; CPU/handoff; no GPU compute |
| S7-t1-F-s102-40 | core | S7-t1-F-s102-40-gpu | afterok | 0 | 1 | 8 | yes | 8 | [0, 0]; CPU/handoff; no GPU compute |
| S7-t1-F-s103-40 | core | S7-t1-F-s103-40-gpu | afterok | 0 | 1 | 8 | yes | 8 | [0, 0]; CPU/handoff; no GPU compute |
| R-incoming-101-gpu | core | R-incoming-101-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| R-incoming-102-gpu | core | R-incoming-102-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| R-incoming-103-gpu | core | R-incoming-103-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| M4-101 | core | M4-101-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| M4-102 | core | M4-102-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| M4-103 | core | M4-103-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| S10-base-ifeval-train-stage | core | S7-base | afterok | 0 | 1 | 8 | yes | 1 | None; core measured scoring/preparation/qualification allocation pending |
| S10-base-gsm8k-train-stage | core | S7-base | afterok | 0 | 1 | 8 | yes | 1 | None; core measured scoring/preparation/qualification allocation pending |
| S10-base-arc-train-stage | core | S7-base | afterok | 0 | 1 | 8 | yes | 1 | None; core measured scoring/preparation/qualification allocation pending |
| S10-t1-D-s101-40-ifeval-train-stage | core | S7-t1-D-s101-40 | afterok | 0 | 1 | 8 | yes | 1 | None; core measured scoring/preparation/qualification allocation pending |
| S10-t1-D-s101-40-gsm8k-train-stage | core | S7-t1-D-s101-40 | afterok | 0 | 1 | 8 | yes | 1 | None; core measured scoring/preparation/qualification allocation pending |
| S10-t1-D-s101-40-arc-train-stage | core | S7-t1-D-s101-40 | afterok | 0 | 1 | 8 | yes | 1 | None; core measured scoring/preparation/qualification allocation pending |
| S10-t1-D-s101-40-finqa-train-stage | core | S7-t1-D-s101-40 | afterok | 0 | 1 | 8 | yes | 1 | None; core measured scoring/preparation/qualification allocation pending |
| S10-t1-D-s102-40-ifeval-train-stage | core | S7-t1-D-s102-40 | afterok | 0 | 1 | 8 | yes | 1 | None; core measured scoring/preparation/qualification allocation pending |
| S10-t1-D-s102-40-gsm8k-train-stage | core | S7-t1-D-s102-40 | afterok | 0 | 1 | 8 | yes | 1 | None; core measured scoring/preparation/qualification allocation pending |
| S10-t1-D-s102-40-arc-train-stage | core | S7-t1-D-s102-40 | afterok | 0 | 1 | 8 | yes | 1 | None; core measured scoring/preparation/qualification allocation pending |
| S10-t1-D-s102-40-finqa-train-stage | core | S7-t1-D-s102-40 | afterok | 0 | 1 | 8 | yes | 1 | None; core measured scoring/preparation/qualification allocation pending |
| S10-t1-D-s103-40-ifeval-train-stage | core | S7-t1-D-s103-40 | afterok | 0 | 1 | 8 | yes | 1 | None; core measured scoring/preparation/qualification allocation pending |
| S10-t1-D-s103-40-gsm8k-train-stage | core | S7-t1-D-s103-40 | afterok | 0 | 1 | 8 | yes | 1 | None; core measured scoring/preparation/qualification allocation pending |
| S10-t1-D-s103-40-arc-train-stage | core | S7-t1-D-s103-40 | afterok | 0 | 1 | 8 | yes | 1 | None; core measured scoring/preparation/qualification allocation pending |
| S10-t1-D-s103-40-finqa-train-stage | core | S7-t1-D-s103-40 | afterok | 0 | 1 | 8 | yes | 1 | None; core measured scoring/preparation/qualification allocation pending |
| S10-t1-F-s101-40-ifeval-train-stage | core | S7-t1-F-s101-40 | afterok | 0 | 1 | 8 | yes | 1 | None; core measured scoring/preparation/qualification allocation pending |
| S10-t1-F-s101-40-gsm8k-train-stage | core | S7-t1-F-s101-40 | afterok | 0 | 1 | 8 | yes | 1 | None; core measured scoring/preparation/qualification allocation pending |
| S10-t1-F-s101-40-arc-train-stage | core | S7-t1-F-s101-40 | afterok | 0 | 1 | 8 | yes | 1 | None; core measured scoring/preparation/qualification allocation pending |
| S10-t1-F-s101-40-finqa-train-stage | core | S7-t1-F-s101-40 | afterok | 0 | 1 | 8 | yes | 1 | None; core measured scoring/preparation/qualification allocation pending |
| S10-t1-F-s102-40-ifeval-train-stage | core | S7-t1-F-s102-40 | afterok | 0 | 1 | 8 | yes | 1 | None; core measured scoring/preparation/qualification allocation pending |
| S10-t1-F-s102-40-gsm8k-train-stage | core | S7-t1-F-s102-40 | afterok | 0 | 1 | 8 | yes | 1 | None; core measured scoring/preparation/qualification allocation pending |
| S10-t1-F-s102-40-arc-train-stage | core | S7-t1-F-s102-40 | afterok | 0 | 1 | 8 | yes | 1 | None; core measured scoring/preparation/qualification allocation pending |
| S10-t1-F-s102-40-finqa-train-stage | core | S7-t1-F-s102-40 | afterok | 0 | 1 | 8 | yes | 1 | None; core measured scoring/preparation/qualification allocation pending |
| S10-t1-F-s103-40-ifeval-train-stage | core | S7-t1-F-s103-40 | afterok | 0 | 1 | 8 | yes | 1 | None; core measured scoring/preparation/qualification allocation pending |
| S10-t1-F-s103-40-gsm8k-train-stage | core | S7-t1-F-s103-40 | afterok | 0 | 1 | 8 | yes | 1 | None; core measured scoring/preparation/qualification allocation pending |
| S10-t1-F-s103-40-arc-train-stage | core | S7-t1-F-s103-40 | afterok | 0 | 1 | 8 | yes | 1 | None; core measured scoring/preparation/qualification allocation pending |
| S10-t1-F-s103-40-finqa-train-stage | core | S7-t1-F-s103-40 | afterok | 0 | 1 | 8 | yes | 1 | None; core measured scoring/preparation/qualification allocation pending |
| R-incoming-101 | core | R-incoming-101-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| R-incoming-102 | core | R-incoming-102-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| R-incoming-103 | core | R-incoming-103-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| M6-101-incoming-stage | core | M4-101, M5, FV-core-101 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| M6-102-incoming-stage | core | M4-102, M5, FV-core-102 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| M6-103-incoming-stage | core | M4-103, M5, FV-core-103 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| M7-101-incoming-stage | core | M-pools, M4-101, FV-core-101 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| M7-102-incoming-stage | core | M-pools, M4-102, FV-core-102 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| M7-103-incoming-stage | core | M-pools, M4-103, FV-core-103 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| S10-base-ifeval-train-gpu | core | S10-base-ifeval-train-stage | afterok | 8 | 1 | 32 | no | 1 | [0.75, 1.4]; core Screen S four-update probe; included in S envelope |
| S10-base-gsm8k-train-gpu | core | S10-base-gsm8k-train-stage | afterok | 8 | 1 | 32 | no | 1 | [0.75, 1.4]; core Screen S four-update probe; included in S envelope |
| S10-base-arc-train-gpu | core | S10-base-arc-train-stage | afterok | 8 | 1 | 32 | no | 1 | [0.75, 1.4]; core Screen S four-update probe; included in S envelope |
| S10-t1-D-s101-40-ifeval-train-gpu | core | S10-t1-D-s101-40-ifeval-train-stage | afterok | 8 | 1 | 32 | no | 1 | [0.75, 1.4]; core Screen S four-update probe; included in S envelope |
| S10-t1-D-s101-40-gsm8k-train-gpu | core | S10-t1-D-s101-40-gsm8k-train-stage | afterok | 8 | 1 | 32 | no | 1 | [0.75, 1.4]; core Screen S four-update probe; included in S envelope |
| S10-t1-D-s101-40-arc-train-gpu | core | S10-t1-D-s101-40-arc-train-stage | afterok | 8 | 1 | 32 | no | 1 | [0.75, 1.4]; core Screen S four-update probe; included in S envelope |
| S10-t1-D-s101-40-finqa-train-gpu | core | S10-t1-D-s101-40-finqa-train-stage | afterok | 8 | 1 | 32 | no | 1 | [0.75, 1.4]; core Screen S four-update probe; included in S envelope |
| S10-t1-D-s102-40-ifeval-train-gpu | core | S10-t1-D-s102-40-ifeval-train-stage | afterok | 8 | 1 | 32 | no | 1 | [0.75, 1.4]; core Screen S four-update probe; included in S envelope |
| S10-t1-D-s102-40-gsm8k-train-gpu | core | S10-t1-D-s102-40-gsm8k-train-stage | afterok | 8 | 1 | 32 | no | 1 | [0.75, 1.4]; core Screen S four-update probe; included in S envelope |
| S10-t1-D-s102-40-arc-train-gpu | core | S10-t1-D-s102-40-arc-train-stage | afterok | 8 | 1 | 32 | no | 1 | [0.75, 1.4]; core Screen S four-update probe; included in S envelope |
| S10-t1-D-s102-40-finqa-train-gpu | core | S10-t1-D-s102-40-finqa-train-stage | afterok | 8 | 1 | 32 | no | 1 | [0.75, 1.4]; core Screen S four-update probe; included in S envelope |
| S10-t1-D-s103-40-ifeval-train-gpu | core | S10-t1-D-s103-40-ifeval-train-stage | afterok | 8 | 1 | 32 | no | 1 | [0.75, 1.4]; core Screen S four-update probe; included in S envelope |
| S10-t1-D-s103-40-gsm8k-train-gpu | core | S10-t1-D-s103-40-gsm8k-train-stage | afterok | 8 | 1 | 32 | no | 1 | [0.75, 1.4]; core Screen S four-update probe; included in S envelope |
| S10-t1-D-s103-40-arc-train-gpu | core | S10-t1-D-s103-40-arc-train-stage | afterok | 8 | 1 | 32 | no | 1 | [0.75, 1.4]; core Screen S four-update probe; included in S envelope |
| S10-t1-D-s103-40-finqa-train-gpu | core | S10-t1-D-s103-40-finqa-train-stage | afterok | 8 | 1 | 32 | no | 1 | [0.75, 1.4]; core Screen S four-update probe; included in S envelope |
| S10-t1-F-s101-40-ifeval-train-gpu | core | S10-t1-F-s101-40-ifeval-train-stage | afterok | 8 | 1 | 32 | no | 1 | [0.75, 1.4]; core Screen S four-update probe; included in S envelope |
| S10-t1-F-s101-40-gsm8k-train-gpu | core | S10-t1-F-s101-40-gsm8k-train-stage | afterok | 8 | 1 | 32 | no | 1 | [0.75, 1.4]; core Screen S four-update probe; included in S envelope |
| S10-t1-F-s101-40-arc-train-gpu | core | S10-t1-F-s101-40-arc-train-stage | afterok | 8 | 1 | 32 | no | 1 | [0.75, 1.4]; core Screen S four-update probe; included in S envelope |
| S10-t1-F-s101-40-finqa-train-gpu | core | S10-t1-F-s101-40-finqa-train-stage | afterok | 8 | 1 | 32 | no | 1 | [0.75, 1.4]; core Screen S four-update probe; included in S envelope |
| S10-t1-F-s102-40-ifeval-train-gpu | core | S10-t1-F-s102-40-ifeval-train-stage | afterok | 8 | 1 | 32 | no | 1 | [0.75, 1.4]; core Screen S four-update probe; included in S envelope |
| S10-t1-F-s102-40-gsm8k-train-gpu | core | S10-t1-F-s102-40-gsm8k-train-stage | afterok | 8 | 1 | 32 | no | 1 | [0.75, 1.4]; core Screen S four-update probe; included in S envelope |
| S10-t1-F-s102-40-arc-train-gpu | core | S10-t1-F-s102-40-arc-train-stage | afterok | 8 | 1 | 32 | no | 1 | [0.75, 1.4]; core Screen S four-update probe; included in S envelope |
| S10-t1-F-s102-40-finqa-train-gpu | core | S10-t1-F-s102-40-finqa-train-stage | afterok | 8 | 1 | 32 | no | 1 | [0.75, 1.4]; core Screen S four-update probe; included in S envelope |
| S10-t1-F-s103-40-ifeval-train-gpu | core | S10-t1-F-s103-40-ifeval-train-stage | afterok | 8 | 1 | 32 | no | 1 | [0.75, 1.4]; core Screen S four-update probe; included in S envelope |
| S10-t1-F-s103-40-gsm8k-train-gpu | core | S10-t1-F-s103-40-gsm8k-train-stage | afterok | 8 | 1 | 32 | no | 1 | [0.75, 1.4]; core Screen S four-update probe; included in S envelope |
| S10-t1-F-s103-40-arc-train-gpu | core | S10-t1-F-s103-40-arc-train-stage | afterok | 8 | 1 | 32 | no | 1 | [0.75, 1.4]; core Screen S four-update probe; included in S envelope |
| S10-t1-F-s103-40-finqa-train-gpu | core | S10-t1-F-s103-40-finqa-train-stage | afterok | 8 | 1 | 32 | no | 1 | [0.75, 1.4]; core Screen S four-update probe; included in S envelope |
| R-generic-101-stage | core | R-incoming-101 | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; CPU/handoff; no GPU compute |
| R-broad-101-stage | core | R-incoming-101 | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; CPU/handoff; no GPU compute |
| R-earlier-101-stage | core | R-incoming-101 | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; CPU/handoff; no GPU compute |
| R-both-101-stage | core | R-incoming-101 | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; CPU/handoff; no GPU compute |
| R-generic-102-stage | core | R-incoming-102 | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; CPU/handoff; no GPU compute |
| R-broad-102-stage | core | R-incoming-102 | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; CPU/handoff; no GPU compute |
| R-earlier-102-stage | core | R-incoming-102 | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; CPU/handoff; no GPU compute |
| R-both-102-stage | core | R-incoming-102 | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; CPU/handoff; no GPU compute |
| R-generic-103-stage | core | R-incoming-103 | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; CPU/handoff; no GPU compute |
| R-broad-103-stage | core | R-incoming-103 | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; CPU/handoff; no GPU compute |
| R-earlier-103-stage | core | R-incoming-103 | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; CPU/handoff; no GPU compute |
| R-both-103-stage | core | R-incoming-103 | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; CPU/handoff; no GPU compute |
| M6-101-incoming-gpu | core | M6-101-incoming-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| M6-102-incoming-gpu | core | M6-102-incoming-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| M6-103-incoming-gpu | core | M6-103-incoming-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| M7-101-incoming-gpu | core | M7-101-incoming-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| M7-102-incoming-gpu | core | M7-102-incoming-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| M7-103-incoming-gpu | core | M7-103-incoming-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| S10-base-ifeval-train | core | S10-base-ifeval-train-gpu | afterok | 0 | 1 | 8 | yes | 1 | None; core measured scoring/preparation/qualification allocation pending |
| S10-base-gsm8k-train | core | S10-base-gsm8k-train-gpu | afterok | 0 | 1 | 8 | yes | 1 | None; core measured scoring/preparation/qualification allocation pending |
| S10-base-arc-train | core | S10-base-arc-train-gpu | afterok | 0 | 1 | 8 | yes | 1 | None; core measured scoring/preparation/qualification allocation pending |
| S10-t1-D-s101-40-ifeval-train | core | S10-t1-D-s101-40-ifeval-train-gpu | afterok | 0 | 1 | 8 | yes | 1 | None; core measured scoring/preparation/qualification allocation pending |
| S10-t1-D-s101-40-gsm8k-train | core | S10-t1-D-s101-40-gsm8k-train-gpu | afterok | 0 | 1 | 8 | yes | 1 | None; core measured scoring/preparation/qualification allocation pending |
| S10-t1-D-s101-40-arc-train | core | S10-t1-D-s101-40-arc-train-gpu | afterok | 0 | 1 | 8 | yes | 1 | None; core measured scoring/preparation/qualification allocation pending |
| S10-t1-D-s101-40-finqa-train | core | S10-t1-D-s101-40-finqa-train-gpu | afterok | 0 | 1 | 8 | yes | 1 | None; core measured scoring/preparation/qualification allocation pending |
| S10-t1-D-s102-40-ifeval-train | core | S10-t1-D-s102-40-ifeval-train-gpu | afterok | 0 | 1 | 8 | yes | 1 | None; core measured scoring/preparation/qualification allocation pending |
| S10-t1-D-s102-40-gsm8k-train | core | S10-t1-D-s102-40-gsm8k-train-gpu | afterok | 0 | 1 | 8 | yes | 1 | None; core measured scoring/preparation/qualification allocation pending |
| S10-t1-D-s102-40-arc-train | core | S10-t1-D-s102-40-arc-train-gpu | afterok | 0 | 1 | 8 | yes | 1 | None; core measured scoring/preparation/qualification allocation pending |
| S10-t1-D-s102-40-finqa-train | core | S10-t1-D-s102-40-finqa-train-gpu | afterok | 0 | 1 | 8 | yes | 1 | None; core measured scoring/preparation/qualification allocation pending |
| S10-t1-D-s103-40-ifeval-train | core | S10-t1-D-s103-40-ifeval-train-gpu | afterok | 0 | 1 | 8 | yes | 1 | None; core measured scoring/preparation/qualification allocation pending |
| S10-t1-D-s103-40-gsm8k-train | core | S10-t1-D-s103-40-gsm8k-train-gpu | afterok | 0 | 1 | 8 | yes | 1 | None; core measured scoring/preparation/qualification allocation pending |
| S10-t1-D-s103-40-arc-train | core | S10-t1-D-s103-40-arc-train-gpu | afterok | 0 | 1 | 8 | yes | 1 | None; core measured scoring/preparation/qualification allocation pending |
| S10-t1-D-s103-40-finqa-train | core | S10-t1-D-s103-40-finqa-train-gpu | afterok | 0 | 1 | 8 | yes | 1 | None; core measured scoring/preparation/qualification allocation pending |
| S10-t1-F-s101-40-ifeval-train | core | S10-t1-F-s101-40-ifeval-train-gpu | afterok | 0 | 1 | 8 | yes | 1 | None; core measured scoring/preparation/qualification allocation pending |
| S10-t1-F-s101-40-gsm8k-train | core | S10-t1-F-s101-40-gsm8k-train-gpu | afterok | 0 | 1 | 8 | yes | 1 | None; core measured scoring/preparation/qualification allocation pending |
| S10-t1-F-s101-40-arc-train | core | S10-t1-F-s101-40-arc-train-gpu | afterok | 0 | 1 | 8 | yes | 1 | None; core measured scoring/preparation/qualification allocation pending |
| S10-t1-F-s101-40-finqa-train | core | S10-t1-F-s101-40-finqa-train-gpu | afterok | 0 | 1 | 8 | yes | 1 | None; core measured scoring/preparation/qualification allocation pending |
| S10-t1-F-s102-40-ifeval-train | core | S10-t1-F-s102-40-ifeval-train-gpu | afterok | 0 | 1 | 8 | yes | 1 | None; core measured scoring/preparation/qualification allocation pending |
| S10-t1-F-s102-40-gsm8k-train | core | S10-t1-F-s102-40-gsm8k-train-gpu | afterok | 0 | 1 | 8 | yes | 1 | None; core measured scoring/preparation/qualification allocation pending |
| S10-t1-F-s102-40-arc-train | core | S10-t1-F-s102-40-arc-train-gpu | afterok | 0 | 1 | 8 | yes | 1 | None; core measured scoring/preparation/qualification allocation pending |
| S10-t1-F-s102-40-finqa-train | core | S10-t1-F-s102-40-finqa-train-gpu | afterok | 0 | 1 | 8 | yes | 1 | None; core measured scoring/preparation/qualification allocation pending |
| S10-t1-F-s103-40-ifeval-train | core | S10-t1-F-s103-40-ifeval-train-gpu | afterok | 0 | 1 | 8 | yes | 1 | None; core measured scoring/preparation/qualification allocation pending |
| S10-t1-F-s103-40-gsm8k-train | core | S10-t1-F-s103-40-gsm8k-train-gpu | afterok | 0 | 1 | 8 | yes | 1 | None; core measured scoring/preparation/qualification allocation pending |
| S10-t1-F-s103-40-arc-train | core | S10-t1-F-s103-40-arc-train-gpu | afterok | 0 | 1 | 8 | yes | 1 | None; core measured scoring/preparation/qualification allocation pending |
| S10-t1-F-s103-40-finqa-train | core | S10-t1-F-s103-40-finqa-train-gpu | afterok | 0 | 1 | 8 | yes | 1 | None; core measured scoring/preparation/qualification allocation pending |
| R-generic-101-gpu | core | R-generic-101-stage | afterok | 8 | 1 | 32 | no | 1 | [7.733333333333333, 7.733333333333333]; core 58 min x 8 GPU proxy/full stage |
| R-broad-101-gpu | core | R-broad-101-stage | afterok | 8 | 1 | 32 | no | 1 | [7.733333333333333, 7.733333333333333]; core 58 min x 8 GPU proxy/full stage |
| R-earlier-101-gpu | core | R-earlier-101-stage | afterok | 8 | 1 | 32 | no | 1 | [7.733333333333333, 7.733333333333333]; core 58 min x 8 GPU proxy/full stage |
| R-both-101-gpu | core | R-both-101-stage | afterok | 8 | 1 | 32 | no | 1 | [7.733333333333333, 7.733333333333333]; core 58 min x 8 GPU proxy/full stage |
| R-generic-102-gpu | core | R-generic-102-stage | afterok | 8 | 1 | 32 | no | 1 | [7.733333333333333, 7.733333333333333]; core 58 min x 8 GPU proxy/full stage |
| R-broad-102-gpu | core | R-broad-102-stage | afterok | 8 | 1 | 32 | no | 1 | [7.733333333333333, 7.733333333333333]; core 58 min x 8 GPU proxy/full stage |
| R-earlier-102-gpu | core | R-earlier-102-stage | afterok | 8 | 1 | 32 | no | 1 | [7.733333333333333, 7.733333333333333]; core 58 min x 8 GPU proxy/full stage |
| R-both-102-gpu | core | R-both-102-stage | afterok | 8 | 1 | 32 | no | 1 | [7.733333333333333, 7.733333333333333]; core 58 min x 8 GPU proxy/full stage |
| R-generic-103-gpu | core | R-generic-103-stage | afterok | 8 | 1 | 32 | no | 1 | [7.733333333333333, 7.733333333333333]; core 58 min x 8 GPU proxy/full stage |
| R-broad-103-gpu | core | R-broad-103-stage | afterok | 8 | 1 | 32 | no | 1 | [7.733333333333333, 7.733333333333333]; core 58 min x 8 GPU proxy/full stage |
| R-earlier-103-gpu | core | R-earlier-103-stage | afterok | 8 | 1 | 32 | no | 1 | [7.733333333333333, 7.733333333333333]; core 58 min x 8 GPU proxy/full stage |
| R-both-103-gpu | core | R-both-103-stage | afterok | 8 | 1 | 32 | no | 1 | [7.733333333333333, 7.733333333333333]; core 58 min x 8 GPU proxy/full stage |
| M6-101-incoming | core | M6-101-incoming-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| M6-102-incoming | core | M6-102-incoming-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| M6-103-incoming | core | M6-103-incoming-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| M7-101-incoming | core | M7-101-incoming-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| M7-102-incoming | core | M7-102-incoming-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| M7-103-incoming | core | M7-103-incoming-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| S8-S10-101-stage | core | S6, S7-base, S7-t1-D-s101-40, S7-t1-F-s101-40, S10-base-ifeval-train, S10-base-gsm8k-train, S10-base-arc-train, S10-t1-D-s101-40-ifeval-train, S10-t1-D-s101-40-gsm8k-train, S10-t1-D-s101-40-arc-train, S10-t1-D-s101-40-finqa-train, S10-t1-F-s101-40-ifeval-train, S10-t1-F-s101-40-gsm8k-train, S10-t1-F-s101-40-arc-train, S10-t1-F-s101-40-finqa-train | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| S8-S10-102-stage | core | S6, S7-base, S7-t1-D-s102-40, S7-t1-F-s102-40, S10-base-ifeval-train, S10-base-gsm8k-train, S10-base-arc-train, S10-t1-D-s102-40-ifeval-train, S10-t1-D-s102-40-gsm8k-train, S10-t1-D-s102-40-arc-train, S10-t1-D-s102-40-finqa-train, S10-t1-F-s102-40-ifeval-train, S10-t1-F-s102-40-gsm8k-train, S10-t1-F-s102-40-arc-train, S10-t1-F-s102-40-finqa-train | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| S8-S10-103-stage | core | S6, S7-base, S7-t1-D-s103-40, S7-t1-F-s103-40, S10-base-ifeval-train, S10-base-gsm8k-train, S10-base-arc-train, S10-t1-D-s103-40-ifeval-train, S10-t1-D-s103-40-gsm8k-train, S10-t1-D-s103-40-arc-train, S10-t1-D-s103-40-finqa-train, S10-t1-F-s103-40-ifeval-train, S10-t1-F-s103-40-gsm8k-train, S10-t1-F-s103-40-arc-train, S10-t1-F-s103-40-finqa-train | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| R-generic-101 | core | R-generic-101-gpu | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; CPU/handoff; no GPU compute |
| R-broad-101 | core | R-broad-101-gpu | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; CPU/handoff; no GPU compute |
| R-earlier-101 | core | R-earlier-101-gpu | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; CPU/handoff; no GPU compute |
| R-both-101 | core | R-both-101-gpu | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; CPU/handoff; no GPU compute |
| R-generic-102 | core | R-generic-102-gpu | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; CPU/handoff; no GPU compute |
| R-broad-102 | core | R-broad-102-gpu | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; CPU/handoff; no GPU compute |
| R-earlier-102 | core | R-earlier-102-gpu | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; CPU/handoff; no GPU compute |
| R-both-102 | core | R-both-102-gpu | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; CPU/handoff; no GPU compute |
| R-generic-103 | core | R-generic-103-gpu | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; CPU/handoff; no GPU compute |
| R-broad-103 | core | R-broad-103-gpu | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; CPU/handoff; no GPU compute |
| R-earlier-103 | core | R-earlier-103-gpu | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; CPU/handoff; no GPU compute |
| R-both-103 | core | R-both-103-gpu | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; CPU/handoff; no GPU compute |
| M6-101-train-stage | core | M6-101-incoming | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; CPU/handoff; no GPU compute |
| M6-102-train-stage | core | M6-102-incoming | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; CPU/handoff; no GPU compute |
| M6-103-train-stage | core | M6-103-incoming | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; CPU/handoff; no GPU compute |
| M7-101-train-stage | core | M7-101-incoming | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; CPU/handoff; no GPU compute |
| M7-102-train-stage | core | M7-102-incoming | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; CPU/handoff; no GPU compute |
| M7-103-train-stage | core | M7-103-incoming | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; CPU/handoff; no GPU compute |
| S8-S10-101-gpu | core | S8-S10-101-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| S8-S10-102-gpu | core | S8-S10-102-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| S8-S10-103-gpu | core | S8-S10-103-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| R-generic-101-score-stage | core | R-generic-101, FV-reprobe-101 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| R-broad-101-score-stage | core | R-broad-101, FV-reprobe-101 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| R-earlier-101-score-stage | core | R-earlier-101, FV-reprobe-101 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| R-both-101-score-stage | core | R-both-101, FV-reprobe-101 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| R-generic-102-score-stage | core | R-generic-102, FV-reprobe-102 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| R-broad-102-score-stage | core | R-broad-102, FV-reprobe-102 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| R-earlier-102-score-stage | core | R-earlier-102, FV-reprobe-102 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| R-both-102-score-stage | core | R-both-102, FV-reprobe-102 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| R-generic-103-score-stage | core | R-generic-103, FV-reprobe-103 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| R-broad-103-score-stage | core | R-broad-103, FV-reprobe-103 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| R-earlier-103-score-stage | core | R-earlier-103, FV-reprobe-103 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| R-both-103-score-stage | core | R-both-103, FV-reprobe-103 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| M6-101-train-gpu | core | M6-101-train-stage | afterok | 8 | 1 | 32 | no | 1 | [7.733333333333333, 7.733333333333333]; core 58 min x 8 GPU proxy/full stage |
| M6-102-train-gpu | core | M6-102-train-stage | afterok | 8 | 1 | 32 | no | 1 | [7.733333333333333, 7.733333333333333]; core 58 min x 8 GPU proxy/full stage |
| M6-103-train-gpu | core | M6-103-train-stage | afterok | 8 | 1 | 32 | no | 1 | [7.733333333333333, 7.733333333333333]; core 58 min x 8 GPU proxy/full stage |
| M7-101-train-gpu | core | M7-101-train-stage | afterok | 8 | 1 | 32 | no | 1 | [7.733333333333333, 7.733333333333333]; core 58 min x 8 GPU proxy/full stage |
| M7-102-train-gpu | core | M7-102-train-stage | afterok | 8 | 1 | 32 | no | 1 | [7.733333333333333, 7.733333333333333]; core 58 min x 8 GPU proxy/full stage |
| M7-103-train-gpu | core | M7-103-train-stage | afterok | 8 | 1 | 32 | no | 1 | [7.733333333333333, 7.733333333333333]; core 58 min x 8 GPU proxy/full stage |
| S8-S10-101 | core | S8-S10-101-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| S8-S10-102 | core | S8-S10-102-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| S8-S10-103 | core | S8-S10-103-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| R-generic-101-score-gpu | core | R-generic-101-score-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| R-broad-101-score-gpu | core | R-broad-101-score-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| R-earlier-101-score-gpu | core | R-earlier-101-score-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| R-both-101-score-gpu | core | R-both-101-score-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| R-generic-102-score-gpu | core | R-generic-102-score-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| R-broad-102-score-gpu | core | R-broad-102-score-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| R-earlier-102-score-gpu | core | R-earlier-102-score-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| R-both-102-score-gpu | core | R-both-102-score-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| R-generic-103-score-gpu | core | R-generic-103-score-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| R-broad-103-score-gpu | core | R-broad-103-score-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| R-earlier-103-score-gpu | core | R-earlier-103-score-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| R-both-103-score-gpu | core | R-both-103-score-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| M6-101-train | core | M6-101-train-gpu | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; CPU/handoff; no GPU compute |
| M6-102-train | core | M6-102-train-gpu | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; CPU/handoff; no GPU compute |
| M6-103-train | core | M6-103-train-gpu | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; CPU/handoff; no GPU compute |
| M7-101-train | core | M7-101-train-gpu | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; CPU/handoff; no GPU compute |
| M7-102-train | core | M7-102-train-gpu | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; CPU/handoff; no GPU compute |
| M7-103-train | core | M7-103-train-gpu | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; CPU/handoff; no GPU compute |
| S12 | core | S3, S4-D-finqa-alone-s101, S4-S-finqa-alone-s101, S4-F-finqa-alone-s101, S4-R-finqa-alone-s101, S4-S-finqa-alone-s102, S4-F-finqa-alone-s102, S4-R-finqa-alone-s102, S4-D-finqa-alone-s102, S4-S-finqa-alone-s103, S4-F-finqa-alone-s103, S4-R-finqa-alone-s103, S4-D-finqa-alone-s103, S4-D-R-finqa-alone-s101, S4-D-R-finqa-alone-s102, S4-D-R-finqa-alone-s103, S8-S10-101, S8-S10-102, S8-S10-103 | afterany | 0 | 1 | 4 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| R-generic-101-score | core | R-generic-101-score-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| R-broad-101-score | core | R-broad-101-score-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| R-earlier-101-score | core | R-earlier-101-score-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| R-both-101-score | core | R-both-101-score-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| R-generic-102-score | core | R-generic-102-score-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| R-broad-102-score | core | R-broad-102-score-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| R-earlier-102-score | core | R-earlier-102-score-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| R-both-102-score | core | R-both-102-score-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| R-generic-103-score | core | R-generic-103-score-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| R-broad-103-score | core | R-broad-103-score-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| R-earlier-103-score | core | R-earlier-103-score-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| R-both-103-score | core | R-both-103-score-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| M6-101-score-stage | core | M6-101-train, FV-core-101 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| M6-102-score-stage | core | M6-102-train, FV-core-102 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| M6-103-score-stage | core | M6-103-train, FV-core-103 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| M7-101-score-stage | core | M7-101-train, FV-core-101 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| M7-102-score-stage | core | M7-102-train, FV-core-102 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| M7-103-score-stage | core | M7-103-train, FV-core-103 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| M6-101-score-gpu | core | M6-101-score-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| M6-102-score-gpu | core | M6-102-score-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| M6-103-score-gpu | core | M6-103-score-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| M7-101-score-gpu | core | M7-101-score-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| M7-102-score-gpu | core | M7-102-score-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| M7-103-score-gpu | core | M7-103-score-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| R-compare-broad-101-stage | core | R-generic-101-score, R-broad-101-score, FV-reprobe-101 | afterok | 0 | 1 | 8 | yes | 43 | [0, 0]; CPU/handoff; no GPU compute |
| R-compare-full-101-stage | core | R-generic-101-score, R-broad-101-score, R-earlier-101-score, R-both-101-score, FV-reprobe-101 | afterok | 0 | 1 | 8 | yes | 81 | [0, 0]; CPU/handoff; no GPU compute |
| R-compare-broad-102-stage | core | R-generic-102-score, R-broad-102-score, FV-reprobe-102 | afterok | 0 | 1 | 8 | yes | 43 | [0, 0]; CPU/handoff; no GPU compute |
| R-compare-full-102-stage | core | R-generic-102-score, R-broad-102-score, R-earlier-102-score, R-both-102-score, FV-reprobe-102 | afterok | 0 | 1 | 8 | yes | 81 | [0, 0]; CPU/handoff; no GPU compute |
| R-compare-broad-103-stage | core | R-generic-103-score, R-broad-103-score, FV-reprobe-103 | afterok | 0 | 1 | 8 | yes | 43 | [0, 0]; CPU/handoff; no GPU compute |
| R-compare-full-103-stage | core | R-generic-103-score, R-broad-103-score, R-earlier-103-score, R-both-103-score, FV-reprobe-103 | afterok | 0 | 1 | 8 | yes | 81 | [0, 0]; CPU/handoff; no GPU compute |
| M6-101-score | core | M6-101-score-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| M6-102-score | core | M6-102-score-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| M6-103-score | core | M6-103-score-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| M7-101-score | core | M7-101-score-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| M7-102-score | core | M7-102-score-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| M7-103-score | core | M7-103-score-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| R-compare-broad-101-gpu | core | R-compare-broad-101-stage | afterok | 1 | 1 | 8 | yes | 43 / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| R-compare-full-101-gpu | core | R-compare-full-101-stage | afterok | 1 | 1 | 8 | yes | 81 / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| R-compare-broad-102-gpu | core | R-compare-broad-102-stage | afterok | 1 | 1 | 8 | yes | 43 / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| R-compare-full-102-gpu | core | R-compare-full-102-stage | afterok | 1 | 1 | 8 | yes | 81 / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| R-compare-broad-103-gpu | core | R-compare-broad-103-stage | afterok | 1 | 1 | 8 | yes | 43 / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| R-compare-full-103-gpu | core | R-compare-full-103-stage | afterok | 1 | 1 | 8 | yes | 81 / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| R-compare-broad-101 | core | R-compare-broad-101-gpu | afterok | 0 | 1 | 8 | yes | 43 | [0, 0]; CPU/handoff; no GPU compute |
| R-compare-full-101 | core | R-compare-full-101-gpu | afterok | 0 | 1 | 8 | yes | 81 | [0, 0]; CPU/handoff; no GPU compute |
| R-compare-broad-102 | core | R-compare-broad-102-gpu | afterok | 0 | 1 | 8 | yes | 43 | [0, 0]; CPU/handoff; no GPU compute |
| R-compare-full-102 | core | R-compare-full-102-gpu | afterok | 0 | 1 | 8 | yes | 81 | [0, 0]; CPU/handoff; no GPU compute |
| R-compare-broad-103 | core | R-compare-broad-103-gpu | afterok | 0 | 1 | 8 | yes | 43 | [0, 0]; CPU/handoff; no GPU compute |
| R-compare-full-103 | core | R-compare-full-103-gpu | afterok | 0 | 1 | 8 | yes | 81 | [0, 0]; CPU/handoff; no GPU compute |
| R-broad | core | R0, R-compare-broad-101, R-compare-broad-102, R-compare-broad-103 | afterany | 0 | 1 | 4 | yes | 1 | [0, 0]; CPU/handoff; no GPU compute |
| R-nominate | core | R0, R-compare-full-101, R-compare-full-102, R-compare-full-103 | afterany | 0 | 1 | 4 | yes | 1 | [0, 0]; CPU/handoff; no GPU compute |
| R-collect | core | R0, R-compare-full-101, R-compare-full-102, R-compare-full-103 | afterany | 0 | 1 | 4 | yes | 1 | [0, 0]; CPU/handoff; no GPU compute |
| M-nomination | core | M-pools, R-broad | afterok | 0 | 1 | 4 | yes | 1 | [0, 0]; CPU/handoff; no GPU compute |
| M-final-nomination | core | M-pools, R-nominate | afterok | 0 | 1 | 4 | yes | 1 | [0, 0]; CPU/handoff; no GPU compute |
| M8-101-incoming-stage | core | M-nomination, M4-101, FV-core-101 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| M8-102-incoming-stage | core | M-nomination, M4-102, FV-core-102 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| M8-103-incoming-stage | core | M-nomination, M4-103, FV-core-103 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| M8-101-incoming-gpu | core | M8-101-incoming-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| M8-102-incoming-gpu | core | M8-102-incoming-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| M8-103-incoming-gpu | core | M8-103-incoming-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| M8-101-incoming | core | M8-101-incoming-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| M8-102-incoming | core | M8-102-incoming-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| M8-103-incoming | core | M8-103-incoming-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| M8-101-train-stage | core | M8-101-incoming | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; CPU/handoff; no GPU compute |
| M8-102-train-stage | core | M8-102-incoming | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; CPU/handoff; no GPU compute |
| M8-103-train-stage | core | M8-103-incoming | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; CPU/handoff; no GPU compute |
| M8-101-train-gpu | core | M8-101-train-stage | afterok | 8 | 1 | 32 | no | 1 | [7.733333333333333, 7.733333333333333]; core 58 min x 8 GPU proxy/full stage |
| M8-102-train-gpu | core | M8-102-train-stage | afterok | 8 | 1 | 32 | no | 1 | [7.733333333333333, 7.733333333333333]; core 58 min x 8 GPU proxy/full stage |
| M8-103-train-gpu | core | M8-103-train-stage | afterok | 8 | 1 | 32 | no | 1 | [7.733333333333333, 7.733333333333333]; core 58 min x 8 GPU proxy/full stage |
| M8-101-train | core | M8-101-train-gpu | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; CPU/handoff; no GPU compute |
| M8-102-train | core | M8-102-train-gpu | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; CPU/handoff; no GPU compute |
| M8-103-train | core | M8-103-train-gpu | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; CPU/handoff; no GPU compute |
| M8-101-score-stage | core | M8-101-train, FV-core-101 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| M8-102-score-stage | core | M8-102-train, FV-core-102 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| M8-103-score-stage | core | M8-103-train, FV-core-103 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| M8-101-score-gpu | core | M8-101-score-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| M8-102-score-gpu | core | M8-102-score-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| M8-103-score-gpu | core | M8-103-score-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| M8-101-score | core | M8-101-score-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| M8-102-score | core | M8-102-score-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| M8-103-score | core | M8-103-score-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| M8-compare-101-stage | core | M7-101-score, M8-101-score, FV-core-101 | afterok | 0 | 1 | 8 | yes | 62 if broad nominated; 45 otherwise | [0, 0]; CPU/handoff; no GPU compute |
| M8-compare-102-stage | core | M7-102-score, M8-102-score, FV-core-102 | afterok | 0 | 1 | 8 | yes | 62 if broad nominated; 45 otherwise | [0, 0]; CPU/handoff; no GPU compute |
| M8-compare-103-stage | core | M7-103-score, M8-103-score, FV-core-103 | afterok | 0 | 1 | 8 | yes | 62 if broad nominated; 45 otherwise | [0, 0]; CPU/handoff; no GPU compute |
| M8-compare-101-gpu | core | M8-compare-101-stage | afterok | 1 | 1 | 8 | yes | 62 if broad nominated; 45 otherwise / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| M8-compare-102-gpu | core | M8-compare-102-stage | afterok | 1 | 1 | 8 | yes | 62 if broad nominated; 45 otherwise / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| M8-compare-103-gpu | core | M8-compare-103-stage | afterok | 1 | 1 | 8 | yes | 62 if broad nominated; 45 otherwise / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| M8-compare-101 | core | M8-compare-101-gpu | afterok | 0 | 1 | 8 | yes | 62 if broad nominated; 45 otherwise | [0, 0]; CPU/handoff; no GPU compute |
| M8-compare-102 | core | M8-compare-102-gpu | afterok | 0 | 1 | 8 | yes | 62 if broad nominated; 45 otherwise | [0, 0]; CPU/handoff; no GPU compute |
| M8-compare-103 | core | M8-compare-103-gpu | afterok | 0 | 1 | 8 | yes | 62 if broad nominated; 45 otherwise | [0, 0]; CPU/handoff; no GPU compute |
| M8-read | core | M8-compare-101, M8-compare-102, M8-compare-103 | afterok | 0 | 1 | 4 | yes | 1 | [0, 0]; CPU/handoff; no GPU compute |
| M9-101-incoming-stage | core | M8-read, M-final-nomination, M4-101, FV-core-101 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| M9-102-incoming-stage | core | M8-read, M-final-nomination, M4-102, FV-core-102 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| M9-103-incoming-stage | core | M8-read, M-final-nomination, M4-103, FV-core-103 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| M9-101-incoming-gpu | core | M9-101-incoming-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| M9-102-incoming-gpu | core | M9-102-incoming-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| M9-103-incoming-gpu | core | M9-103-incoming-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| M9-101-incoming | core | M9-101-incoming-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| M9-102-incoming | core | M9-102-incoming-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| M9-103-incoming | core | M9-103-incoming-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| M9-101-train-stage | core | M9-101-incoming | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; CPU/handoff; no GPU compute |
| M9-102-train-stage | core | M9-102-incoming | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; CPU/handoff; no GPU compute |
| M9-103-train-stage | core | M9-103-incoming | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; CPU/handoff; no GPU compute |
| M9-101-train-gpu | core | M9-101-train-stage | afterok | 8 | 1 | 32 | no | 1 | [7.733333333333333, 7.733333333333333]; core 58 min x 8 GPU proxy/full stage |
| M9-102-train-gpu | core | M9-102-train-stage | afterok | 8 | 1 | 32 | no | 1 | [7.733333333333333, 7.733333333333333]; core 58 min x 8 GPU proxy/full stage |
| M9-103-train-gpu | core | M9-103-train-stage | afterok | 8 | 1 | 32 | no | 1 | [7.733333333333333, 7.733333333333333]; core 58 min x 8 GPU proxy/full stage |
| M9-101-train | core | M9-101-train-gpu | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; CPU/handoff; no GPU compute |
| M9-102-train | core | M9-102-train-gpu | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; CPU/handoff; no GPU compute |
| M9-103-train | core | M9-103-train-gpu | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; CPU/handoff; no GPU compute |
| M9-101-score-stage | core | M9-101-train, FV-core-101 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| M9-102-score-stage | core | M9-102-train, FV-core-102 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| M9-103-score-stage | core | M9-103-train, FV-core-103 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| M9-101-score-gpu | core | M9-101-score-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| M9-102-score-gpu | core | M9-102-score-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| M9-103-score-gpu | core | M9-103-score-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| M9-101-score | core | M9-101-score-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| M9-102-score | core | M9-102-score-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| M9-103-score | core | M9-103-score-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| M9-compare-101-stage | core | M9-101-score, M8-compare-101, FV-core-101 | afterok | 0 | 1 | 8 | yes | up to 79, conditional earlier arm | [0, 0]; CPU/handoff; no GPU compute |
| M9-compare-102-stage | core | M9-102-score, M8-compare-102, FV-core-102 | afterok | 0 | 1 | 8 | yes | up to 79, conditional earlier arm | [0, 0]; CPU/handoff; no GPU compute |
| M9-compare-103-stage | core | M9-103-score, M8-compare-103, FV-core-103 | afterok | 0 | 1 | 8 | yes | up to 79, conditional earlier arm | [0, 0]; CPU/handoff; no GPU compute |
| M9-compare-101-gpu | core | M9-compare-101-stage | afterok | 1 | 1 | 8 | yes | up to 79, conditional earlier arm / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| M9-compare-102-gpu | core | M9-compare-102-stage | afterok | 1 | 1 | 8 | yes | up to 79, conditional earlier arm / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| M9-compare-103-gpu | core | M9-compare-103-stage | afterok | 1 | 1 | 8 | yes | up to 79, conditional earlier arm / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| M9-compare-101 | core | M9-compare-101-gpu | afterok | 0 | 1 | 8 | yes | up to 79, conditional earlier arm | [0, 0]; CPU/handoff; no GPU compute |
| M9-compare-102 | core | M9-compare-102-gpu | afterok | 0 | 1 | 8 | yes | up to 79, conditional earlier arm | [0, 0]; CPU/handoff; no GPU compute |
| M9-compare-103 | core | M9-compare-103-gpu | afterok | 0 | 1 | 8 | yes | up to 79, conditional earlier arm | [0, 0]; CPU/handoff; no GPU compute |
| M9 | core | M8-read, M9-compare-101, M9-compare-102, M9-compare-103 | afterok | 0 | 1 | 4 | yes | 1 | [0, 0]; CPU/handoff; no GPU compute |
| M10-101-incoming-stage | core | M9, M5, FV-core-101 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| M10-102-incoming-stage | core | M9, M5, FV-core-102 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| M10-103-incoming-stage | core | M9, M5, FV-core-103 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| M10-101-incoming-gpu | core | M10-101-incoming-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| M10-102-incoming-gpu | core | M10-102-incoming-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| M10-103-incoming-gpu | core | M10-103-incoming-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| M10-101-incoming | core | M10-101-incoming-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| M10-102-incoming | core | M10-102-incoming-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| M10-103-incoming | core | M10-103-incoming-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| M10-101-train-stage | core | M10-101-incoming | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; CPU/handoff; no GPU compute |
| M10-102-train-stage | core | M10-102-incoming | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; CPU/handoff; no GPU compute |
| M10-103-train-stage | core | M10-103-incoming | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; CPU/handoff; no GPU compute |
| M10-101-train-gpu | core | M10-101-train-stage | afterok | 8 | 1 | 32 | no | 1 | [7.733333333333333, 7.733333333333333]; core 58 min x 8 GPU proxy/full stage |
| M10-102-train-gpu | core | M10-102-train-stage | afterok | 8 | 1 | 32 | no | 1 | [7.733333333333333, 7.733333333333333]; core 58 min x 8 GPU proxy/full stage |
| M10-103-train-gpu | core | M10-103-train-stage | afterok | 8 | 1 | 32 | no | 1 | [7.733333333333333, 7.733333333333333]; core 58 min x 8 GPU proxy/full stage |
| M10-101-train | core | M10-101-train-gpu | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; CPU/handoff; no GPU compute |
| M10-102-train | core | M10-102-train-gpu | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; CPU/handoff; no GPU compute |
| M10-103-train | core | M10-103-train-gpu | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; CPU/handoff; no GPU compute |
| M10-101-score-stage | core | M10-101-train, FV-core-101 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| M10-102-score-stage | core | M10-102-train, FV-core-102 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| M10-103-score-stage | core | M10-103-train, FV-core-103 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| M10-101-score-gpu | core | M10-101-score-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| M10-102-score-gpu | core | M10-102-score-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| M10-103-score-gpu | core | M10-103-score-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| M10-101-score | core | M10-101-score-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| M10-102-score | core | M10-102-score-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| M10-103-score | core | M10-103-score-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| M10 | core | M10-101-score, M10-102-score, M10-103-score | afterok | 0 | 1 | 4 | yes | 1 | [0, 0]; CPU/handoff; no GPU compute |
| M-final-score-101-stage | core | M10-101-score, M6-101-score, FV-core-101 | afterok | 0 | 1 | 8 | yes | up to 119, frozen nomination/admission branches | [0, 0]; CPU/handoff; no GPU compute |
| M-final-score-102-stage | core | M10-102-score, M6-102-score, FV-core-102 | afterok | 0 | 1 | 8 | yes | up to 119, frozen nomination/admission branches | [0, 0]; CPU/handoff; no GPU compute |
| M-final-score-103-stage | core | M10-103-score, M6-103-score, FV-core-103 | afterok | 0 | 1 | 8 | yes | up to 119, frozen nomination/admission branches | [0, 0]; CPU/handoff; no GPU compute |
| X1 | core | M10, S6 | afterok | 0 | 1 | 4 | yes | 1 | [0, 0]; CPU/handoff; no GPU compute |
| M-final-score-101-gpu | core | M-final-score-101-stage | afterok | 1 | 1 | 8 | yes | up to 119, frozen nomination/admission branches / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| M-final-score-102-gpu | core | M-final-score-102-stage | afterok | 1 | 1 | 8 | yes | up to 119, frozen nomination/admission branches / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| M-final-score-103-gpu | core | M-final-score-103-stage | afterok | 1 | 1 | 8 | yes | up to 119, frozen nomination/admission branches / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X2 | core | X1 | afterok | 0 | 1 | 4 | yes | 1 | [0, 0]; CPU/handoff; no GPU compute |
| M-final-score-101 | core | M-final-score-101-gpu | afterok | 0 | 1 | 8 | yes | up to 119, frozen nomination/admission branches | [0, 0]; CPU/handoff; no GPU compute |
| M-final-score-102 | core | M-final-score-102-gpu | afterok | 0 | 1 | 8 | yes | up to 119, frozen nomination/admission branches | [0, 0]; CPU/handoff; no GPU compute |
| M-final-score-103 | core | M-final-score-103-gpu | afterok | 0 | 1 | 8 | yes | up to 119, frozen nomination/admission branches | [0, 0]; CPU/handoff; no GPU compute |
| M-core-collect | core | M3, M-final-score-101, M-final-score-102, M-final-score-103, M9, M10 | afterany | 0 | 1 | 4 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X3-101-stage | core | X2 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X3-102-stage | core | X2 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X3-103-stage | core | X2 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X3-101-gpu | core | X3-101-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X3-102-gpu | core | X3-102-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X3-103-gpu | core | X3-103-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X3-101 | core | X3-101-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X3-102 | core | X3-102-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X3-103 | core | X3-103-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X3 | core | X3-101, X3-102, X3-103 | afterok | 0 | 1 | 4 | yes | 1 | [0, 0]; CPU/handoff; no GPU compute |
| X4-101-stage | core | X3 | afterok | 0 | 1 | 8 | yes | 1 | None; core X4 all-three-seed feasibility/calibration envelope 5–7 GPU-h; per-job measurement pending |
| X4-102-stage | core | X3 | afterok | 0 | 1 | 8 | yes | 1 | None; core X4 all-three-seed feasibility/calibration envelope 5–7 GPU-h; per-job measurement pending |
| X4-103-stage | core | X3 | afterok | 0 | 1 | 8 | yes | 1 | None; core X4 all-three-seed feasibility/calibration envelope 5–7 GPU-h; per-job measurement pending |
| X9-probes-101-train-stage | core | X3 | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; CPU/handoff; no GPU compute |
| X9-probes-102-train-stage | core | X3 | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; CPU/handoff; no GPU compute |
| X9-probes-103-train-stage | core | X3 | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; CPU/handoff; no GPU compute |
| X4-101-gpu | core | X4-101-stage | afterok | 8 | 1 | 32 | no | 1 | None; core X4 all-three-seed feasibility/calibration envelope 5–7 GPU-h; per-job measurement pending |
| X4-102-gpu | core | X4-102-stage | afterok | 8 | 1 | 32 | no | 1 | None; core X4 all-three-seed feasibility/calibration envelope 5–7 GPU-h; per-job measurement pending |
| X4-103-gpu | core | X4-103-stage | afterok | 8 | 1 | 32 | no | 1 | None; core X4 all-three-seed feasibility/calibration envelope 5–7 GPU-h; per-job measurement pending |
| X9-probes-101-train-gpu | core | X9-probes-101-train-stage | afterok | 8 | 1 | 32 | no | 1 | [1.5, 2.8]; two core probes x .75–1.4; included in X probe envelope |
| X9-probes-102-train-gpu | core | X9-probes-102-train-stage | afterok | 8 | 1 | 32 | no | 1 | [1.5, 2.8]; two core probes x .75–1.4; included in X probe envelope |
| X9-probes-103-train-gpu | core | X9-probes-103-train-stage | afterok | 8 | 1 | 32 | no | 1 | [1.5, 2.8]; two core probes x .75–1.4; included in X probe envelope |
| X4-101 | core | X4-101-gpu | afterok | 0 | 1 | 8 | yes | 1 | None; core X4 all-three-seed feasibility/calibration envelope 5–7 GPU-h; per-job measurement pending |
| X4-102 | core | X4-102-gpu | afterok | 0 | 1 | 8 | yes | 1 | None; core X4 all-three-seed feasibility/calibration envelope 5–7 GPU-h; per-job measurement pending |
| X4-103 | core | X4-103-gpu | afterok | 0 | 1 | 8 | yes | 1 | None; core X4 all-three-seed feasibility/calibration envelope 5–7 GPU-h; per-job measurement pending |
| X9-probes-101-train | core | X9-probes-101-train-gpu | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; CPU/handoff; no GPU compute |
| X9-probes-102-train | core | X9-probes-102-train-gpu | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; CPU/handoff; no GPU compute |
| X9-probes-103-train | core | X9-probes-103-train-gpu | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; CPU/handoff; no GPU compute |
| X4 | core | X4-101, X4-102, X4-103 | afterok | 0 | 1 | 4 | yes | 1 | None; core X4 all-three-seed feasibility/calibration envelope 5–7 GPU-h; per-job measurement pending |
| X9-probes-101-score-stage | core | X9-probes-101-train, FV-x-tier1-101 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X9-probes-102-score-stage | core | X9-probes-102-train, FV-x-tier1-102 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X9-probes-103-score-stage | core | X9-probes-103-train, FV-x-tier1-103 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X-incoming-101-stage | core | X4, FV-x-tier1-101 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X9-probes-101-score-gpu | core | X9-probes-101-score-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X-incoming-102-stage | core | X4, FV-x-tier1-102 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X9-probes-102-score-gpu | core | X9-probes-102-score-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X-incoming-103-stage | core | X4, FV-x-tier1-103 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X9-probes-103-score-gpu | core | X9-probes-103-score-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X-incoming-101-gpu | core | X-incoming-101-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X9-probes-101-score | core | X9-probes-101-score-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X-incoming-102-gpu | core | X-incoming-102-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X9-probes-102-score | core | X9-probes-102-score-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X-incoming-103-gpu | core | X-incoming-103-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X9-probes-103-score | core | X9-probes-103-score-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X-incoming-101 | core | X-incoming-101-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X-incoming-102 | core | X-incoming-102-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X-incoming-103 | core | X-incoming-103-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| P2-qualify-101-stage | core | X-incoming-101 | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; CPU/handoff; no GPU compute |
| X5-control-101-train-stage | core | X-incoming-101 | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X7-drift-101-train-stage | core | X-incoming-101 | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X9-plasticity-101-train-stage | core | X-incoming-101, X9-probes-101-score | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| P2-qualify-102-stage | core | X-incoming-102 | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; CPU/handoff; no GPU compute |
| X5-control-102-train-stage | core | X-incoming-102 | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X7-drift-102-train-stage | core | X-incoming-102 | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X9-plasticity-102-train-stage | core | X-incoming-102, X9-probes-102-score | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| P2-qualify-103-stage | core | X-incoming-103 | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; CPU/handoff; no GPU compute |
| X5-control-103-train-stage | core | X-incoming-103 | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X7-drift-103-train-stage | core | X-incoming-103 | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X9-plasticity-103-train-stage | core | X-incoming-103, X9-probes-103-score | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| P2-qualify-101-gpu | core | P2-qualify-101-stage | afterok | 8 | 1 | 32 | no | 1 | [7.733333333333333, 7.733333333333333]; core 58 min x 8 GPU proxy/full stage |
| X5-control-101-train-gpu | core | X5-control-101-train-stage | afterok | 8 | 1 | 32 | no | 2 | [7.733333333333333, 7.733333333333333]; core 58 min x 8 GPU proxy/full stage |
| X7-drift-101-train-gpu | core | X7-drift-101-train-stage | afterok | 8 | 1 | 32 | no | 2 | None; split stage; included in full-stage envelope, measured split pending |
| X9-plasticity-101-train-gpu | core | X9-plasticity-101-train-stage | afterok | 8 | 1 | 32 | no | 2 | [7.733333333333333, 7.733333333333333]; core 58 min x 8 GPU proxy/full stage |
| P2-qualify-102-gpu | core | P2-qualify-102-stage | afterok | 8 | 1 | 32 | no | 1 | [7.733333333333333, 7.733333333333333]; core 58 min x 8 GPU proxy/full stage |
| X5-control-102-train-gpu | core | X5-control-102-train-stage | afterok | 8 | 1 | 32 | no | 2 | [7.733333333333333, 7.733333333333333]; core 58 min x 8 GPU proxy/full stage |
| X7-drift-102-train-gpu | core | X7-drift-102-train-stage | afterok | 8 | 1 | 32 | no | 2 | None; split stage; included in full-stage envelope, measured split pending |
| X9-plasticity-102-train-gpu | core | X9-plasticity-102-train-stage | afterok | 8 | 1 | 32 | no | 2 | [7.733333333333333, 7.733333333333333]; core 58 min x 8 GPU proxy/full stage |
| P2-qualify-103-gpu | core | P2-qualify-103-stage | afterok | 8 | 1 | 32 | no | 1 | [7.733333333333333, 7.733333333333333]; core 58 min x 8 GPU proxy/full stage |
| X5-control-103-train-gpu | core | X5-control-103-train-stage | afterok | 8 | 1 | 32 | no | 2 | [7.733333333333333, 7.733333333333333]; core 58 min x 8 GPU proxy/full stage |
| X7-drift-103-train-gpu | core | X7-drift-103-train-stage | afterok | 8 | 1 | 32 | no | 2 | None; split stage; included in full-stage envelope, measured split pending |
| X9-plasticity-103-train-gpu | core | X9-plasticity-103-train-stage | afterok | 8 | 1 | 32 | no | 2 | [7.733333333333333, 7.733333333333333]; core 58 min x 8 GPU proxy/full stage |
| P2-qualify-101 | core | P2-qualify-101-gpu | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; CPU/handoff; no GPU compute |
| X5-control-101-train | core | X5-control-101-train-gpu | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X7-drift-101-train | core | X7-drift-101-train-gpu | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X9-plasticity-101-train | core | X9-plasticity-101-train-gpu | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| P2-qualify-102 | core | P2-qualify-102-gpu | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; CPU/handoff; no GPU compute |
| X5-control-102-train | core | X5-control-102-train-gpu | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X7-drift-102-train | core | X7-drift-102-train-gpu | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X9-plasticity-102-train | core | X9-plasticity-102-train-gpu | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| P2-qualify-103 | core | P2-qualify-103-gpu | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; CPU/handoff; no GPU compute |
| X5-control-103-train | core | X5-control-103-train-gpu | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X7-drift-103-train | core | X7-drift-103-train-gpu | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X9-plasticity-103-train | core | X9-plasticity-103-train-gpu | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X5-control-101-train-score-stage | core | X5-control-101-train, FV-x-tier1-101 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X6-p-101-train-stage | core | X-incoming-101, P2-qualify-101 | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; CPU/handoff; no GPU compute |
| X8-101-signal-stage | core | X7-drift-101-train, FV-x-tier1-101 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X9-plasticity-101-train-score-stage | core | X9-plasticity-101-train, FV-x-tier1-101 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X5-control-102-train-score-stage | core | X5-control-102-train, FV-x-tier1-102 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X6-p-102-train-stage | core | X-incoming-102, P2-qualify-102 | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; CPU/handoff; no GPU compute |
| X8-102-signal-stage | core | X7-drift-102-train, FV-x-tier1-102 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X9-plasticity-102-train-score-stage | core | X9-plasticity-102-train, FV-x-tier1-102 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X5-control-103-train-score-stage | core | X5-control-103-train, FV-x-tier1-103 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X6-p-103-train-stage | core | X-incoming-103, P2-qualify-103 | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; CPU/handoff; no GPU compute |
| X8-103-signal-stage | core | X7-drift-103-train, FV-x-tier1-103 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X9-plasticity-103-train-score-stage | core | X9-plasticity-103-train, FV-x-tier1-103 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X5-control-101-train-score-gpu | core | X5-control-101-train-score-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X6-p-101-train-gpu | core | X6-p-101-train-stage | afterok | 8 | 1 | 32 | no | 1 | [7.733333333333333, 7.733333333333333]; core 58 min x 8 GPU proxy/full stage |
| X8-101-signal-gpu | core | X8-101-signal-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X9-plasticity-101-train-score-gpu | core | X9-plasticity-101-train-score-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X5-control-102-train-score-gpu | core | X5-control-102-train-score-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X6-p-102-train-gpu | core | X6-p-102-train-stage | afterok | 8 | 1 | 32 | no | 1 | [7.733333333333333, 7.733333333333333]; core 58 min x 8 GPU proxy/full stage |
| X8-102-signal-gpu | core | X8-102-signal-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X9-plasticity-102-train-score-gpu | core | X9-plasticity-102-train-score-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X5-control-103-train-score-gpu | core | X5-control-103-train-score-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X6-p-103-train-gpu | core | X6-p-103-train-stage | afterok | 8 | 1 | 32 | no | 1 | [7.733333333333333, 7.733333333333333]; core 58 min x 8 GPU proxy/full stage |
| X8-103-signal-gpu | core | X8-103-signal-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X9-plasticity-103-train-score-gpu | core | X9-plasticity-103-train-score-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X5-control-101-train-score | core | X5-control-101-train-score-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X6-p-101-train | core | X6-p-101-train-gpu | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; CPU/handoff; no GPU compute |
| X8-101-signal | core | X8-101-signal-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X9-plasticity-101-train-score | core | X9-plasticity-101-train-score-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X5-control-102-train-score | core | X5-control-102-train-score-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X6-p-102-train | core | X6-p-102-train-gpu | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; CPU/handoff; no GPU compute |
| X8-102-signal | core | X8-102-signal-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X9-plasticity-102-train-score | core | X9-plasticity-102-train-score-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X5-control-103-train-score | core | X5-control-103-train-score-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X6-p-103-train | core | X6-p-103-train-gpu | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; CPU/handoff; no GPU compute |
| X8-103-signal | core | X8-103-signal-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X9-plasticity-103-train-score | core | X9-plasticity-103-train-score-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X6-p-101-train-score-stage | core | X6-p-101-train, FV-x-tier1-101 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X8-101-continue-stage | core | X8-101-signal | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X6-p-102-train-score-stage | core | X6-p-102-train, FV-x-tier1-102 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X8-102-continue-stage | core | X8-102-signal | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X6-p-103-train-score-stage | core | X6-p-103-train, FV-x-tier1-103 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X8-103-continue-stage | core | X8-103-signal | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X6-p-101-train-score-gpu | core | X6-p-101-train-score-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X8-101-continue-gpu | core | X8-101-continue-stage | afterok | 8 | 1 | 32 | no | 2 | None; split stage; included in full-stage envelope, measured split pending |
| X6-p-102-train-score-gpu | core | X6-p-102-train-score-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X8-102-continue-gpu | core | X8-102-continue-stage | afterok | 8 | 1 | 32 | no | 2 | None; split stage; included in full-stage envelope, measured split pending |
| X6-p-103-train-score-gpu | core | X6-p-103-train-score-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X8-103-continue-gpu | core | X8-103-continue-stage | afterok | 8 | 1 | 32 | no | 2 | None; split stage; included in full-stage envelope, measured split pending |
| X6-p-101-train-score | core | X6-p-101-train-score-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X8-101-continue | core | X8-101-continue-gpu | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X6-p-102-train-score | core | X6-p-102-train-score-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X8-102-continue | core | X8-102-continue-gpu | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X6-p-103-train-score | core | X6-p-103-train-score-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X8-103-continue | core | X8-103-continue-gpu | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X8-101-continue-score-stage | core | X8-101-continue, FV-x-tier1-101 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X8-102-continue-score-stage | core | X8-102-continue, FV-x-tier1-102 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X8-103-continue-score-stage | core | X8-103-continue, FV-x-tier1-103 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X8-101-continue-score-gpu | core | X8-101-continue-score-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X8-102-continue-score-gpu | core | X8-102-continue-score-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X8-103-continue-score-gpu | core | X8-103-continue-score-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X8-101-continue-score | core | X8-101-continue-score-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X8-102-continue-score | core | X8-102-continue-score-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X8-103-continue-score | core | X8-103-continue-score-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X-complete-101-stage | core | X5-control-101-train-score, X6-p-101-train-score, X8-101-continue-score, X9-plasticity-101-train-score, FV-x-tier1-101 | afterok | 0 | 1 | 8 | yes | 63 with all four arms READY; disabled arms SKIP | [0, 0]; CPU/handoff; no GPU compute |
| X-complete-102-stage | core | X5-control-102-train-score, X6-p-102-train-score, X8-102-continue-score, X9-plasticity-102-train-score, FV-x-tier1-102 | afterok | 0 | 1 | 8 | yes | 63 with all four arms READY; disabled arms SKIP | [0, 0]; CPU/handoff; no GPU compute |
| X-complete-103-stage | core | X5-control-103-train-score, X6-p-103-train-score, X8-103-continue-score, X9-plasticity-103-train-score, FV-x-tier1-103 | afterok | 0 | 1 | 8 | yes | 63 with all four arms READY; disabled arms SKIP | [0, 0]; CPU/handoff; no GPU compute |
| X-complete-101-gpu | core | X-complete-101-stage | afterok | 1 | 1 | 8 | yes | 63 with all four arms READY; disabled arms SKIP / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X-complete-102-gpu | core | X-complete-102-stage | afterok | 1 | 1 | 8 | yes | 63 with all four arms READY; disabled arms SKIP / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X-complete-103-gpu | core | X-complete-103-stage | afterok | 1 | 1 | 8 | yes | 63 with all four arms READY; disabled arms SKIP / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X-complete-101 | core | X-complete-101-gpu | afterok | 0 | 1 | 8 | yes | 63 with all four arms READY; disabled arms SKIP | [0, 0]; CPU/handoff; no GPU compute |
| X-complete-102 | core | X-complete-102-gpu | afterok | 0 | 1 | 8 | yes | 63 with all four arms READY; disabled arms SKIP | [0, 0]; CPU/handoff; no GPU compute |
| X-complete-103 | core | X-complete-103-gpu | afterok | 0 | 1 | 8 | yes | 63 with all four arms READY; disabled arms SKIP | [0, 0]; CPU/handoff; no GPU compute |
| X12 | core | X-complete-101, X-complete-102, X-complete-103 | afterany | 0 | 1 | 4 | yes | 3 | [0, 0]; CPU/handoff; no GPU compute |
| X14 | core | X12 | afterok | 0 | 1 | 4 | yes | 1 | [0, 0]; CPU/handoff; no GPU compute |
| X15-101-incoming-stage | core | X14, FV-x-tier2.stack-drift-101 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X15-101-probe-score-stage | core | X14, X9-probes-101-train, FV-x-tier2.stack-drift-101 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X15-102-incoming-stage | core | X14, FV-x-tier2.stack-drift-102 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X15-102-probe-score-stage | core | X14, X9-probes-102-train, FV-x-tier2.stack-drift-102 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X15-103-incoming-stage | core | X14, FV-x-tier2.stack-drift-103 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X15-103-probe-score-stage | core | X14, X9-probes-103-train, FV-x-tier2.stack-drift-103 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X16-101-incoming-stage | core | X14, FV-x-tier2.stack-plasticity-101 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X16-101-probe-score-stage | core | X14, X9-probes-101-train, FV-x-tier2.stack-plasticity-101 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X16-102-incoming-stage | core | X14, FV-x-tier2.stack-plasticity-102 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X16-102-probe-score-stage | core | X14, X9-probes-102-train, FV-x-tier2.stack-plasticity-102 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X16-103-incoming-stage | core | X14, FV-x-tier2.stack-plasticity-103 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X16-103-probe-score-stage | core | X14, X9-probes-103-train, FV-x-tier2.stack-plasticity-103 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X17-101-incoming-stage | core | X14, FV-x-tier2.b-101 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X17-101-probe-score-stage | core | X14, X9-probes-101-train, FV-x-tier2.b-101 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X17-102-incoming-stage | core | X14, FV-x-tier2.b-102 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X17-102-probe-score-stage | core | X14, X9-probes-102-train, FV-x-tier2.b-102 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X17-103-incoming-stage | core | X14, FV-x-tier2.b-103 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X17-103-probe-score-stage | core | X14, X9-probes-103-train, FV-x-tier2.b-103 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X18-101-incoming-stage | core | X14, FV-x-tier2.b-p2-101 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X18-101-probe-score-stage | core | X14, X9-probes-101-train, FV-x-tier2.b-p2-101 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X18-102-incoming-stage | core | X14, FV-x-tier2.b-p2-102 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X18-102-probe-score-stage | core | X14, X9-probes-102-train, FV-x-tier2.b-p2-102 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X18-103-incoming-stage | core | X14, FV-x-tier2.b-p2-103 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X18-103-probe-score-stage | core | X14, X9-probes-103-train, FV-x-tier2.b-p2-103 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X19-101-incoming-stage | core | X14, FV-x-tier2.incremental-101 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X19-101-probe-score-stage | core | X14, X9-probes-101-train, FV-x-tier2.incremental-101 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X19-102-incoming-stage | core | X14, FV-x-tier2.incremental-102 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X19-102-probe-score-stage | core | X14, X9-probes-102-train, FV-x-tier2.incremental-102 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X19-103-incoming-stage | core | X14, FV-x-tier2.incremental-103 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X19-103-probe-score-stage | core | X14, X9-probes-103-train, FV-x-tier2.incremental-103 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X20-101-incoming-stage | core | X14, FV-x-tier2.incremental-p2-101 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X20-101-probe-score-stage | core | X14, X9-probes-101-train, FV-x-tier2.incremental-p2-101 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X20-102-incoming-stage | core | X14, FV-x-tier2.incremental-p2-102 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X20-102-probe-score-stage | core | X14, X9-probes-102-train, FV-x-tier2.incremental-p2-102 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X20-103-incoming-stage | core | X14, FV-x-tier2.incremental-p2-103 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X20-103-probe-score-stage | core | X14, X9-probes-103-train, FV-x-tier2.incremental-p2-103 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X15-101-incoming-gpu | core | X15-101-incoming-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X15-101-probe-score-gpu | core | X15-101-probe-score-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X15-102-incoming-gpu | core | X15-102-incoming-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X15-102-probe-score-gpu | core | X15-102-probe-score-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X15-103-incoming-gpu | core | X15-103-incoming-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X15-103-probe-score-gpu | core | X15-103-probe-score-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X16-101-incoming-gpu | core | X16-101-incoming-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X16-101-probe-score-gpu | core | X16-101-probe-score-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X16-102-incoming-gpu | core | X16-102-incoming-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X16-102-probe-score-gpu | core | X16-102-probe-score-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X16-103-incoming-gpu | core | X16-103-incoming-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X16-103-probe-score-gpu | core | X16-103-probe-score-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X17-101-incoming-gpu | core | X17-101-incoming-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X17-101-probe-score-gpu | core | X17-101-probe-score-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X17-102-incoming-gpu | core | X17-102-incoming-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X17-102-probe-score-gpu | core | X17-102-probe-score-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X17-103-incoming-gpu | core | X17-103-incoming-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X17-103-probe-score-gpu | core | X17-103-probe-score-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X18-101-incoming-gpu | core | X18-101-incoming-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X18-101-probe-score-gpu | core | X18-101-probe-score-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X18-102-incoming-gpu | core | X18-102-incoming-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X18-102-probe-score-gpu | core | X18-102-probe-score-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X18-103-incoming-gpu | core | X18-103-incoming-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X18-103-probe-score-gpu | core | X18-103-probe-score-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X19-101-incoming-gpu | core | X19-101-incoming-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X19-101-probe-score-gpu | core | X19-101-probe-score-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X19-102-incoming-gpu | core | X19-102-incoming-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X19-102-probe-score-gpu | core | X19-102-probe-score-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X19-103-incoming-gpu | core | X19-103-incoming-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X19-103-probe-score-gpu | core | X19-103-probe-score-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X20-101-incoming-gpu | core | X20-101-incoming-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X20-101-probe-score-gpu | core | X20-101-probe-score-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X20-102-incoming-gpu | core | X20-102-incoming-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X20-102-probe-score-gpu | core | X20-102-probe-score-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X20-103-incoming-gpu | core | X20-103-incoming-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X20-103-probe-score-gpu | core | X20-103-probe-score-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X15-101-incoming | core | X15-101-incoming-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X15-101-probe-score | core | X15-101-probe-score-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X15-102-incoming | core | X15-102-incoming-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X15-102-probe-score | core | X15-102-probe-score-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X15-103-incoming | core | X15-103-incoming-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X15-103-probe-score | core | X15-103-probe-score-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X16-101-incoming | core | X16-101-incoming-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X16-101-probe-score | core | X16-101-probe-score-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X16-102-incoming | core | X16-102-incoming-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X16-102-probe-score | core | X16-102-probe-score-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X16-103-incoming | core | X16-103-incoming-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X16-103-probe-score | core | X16-103-probe-score-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X17-101-incoming | core | X17-101-incoming-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X17-101-probe-score | core | X17-101-probe-score-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X17-102-incoming | core | X17-102-incoming-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X17-102-probe-score | core | X17-102-probe-score-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X17-103-incoming | core | X17-103-incoming-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X17-103-probe-score | core | X17-103-probe-score-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X18-101-incoming | core | X18-101-incoming-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X18-101-probe-score | core | X18-101-probe-score-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X18-102-incoming | core | X18-102-incoming-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X18-102-probe-score | core | X18-102-probe-score-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X18-103-incoming | core | X18-103-incoming-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X18-103-probe-score | core | X18-103-probe-score-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X19-101-incoming | core | X19-101-incoming-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X19-101-probe-score | core | X19-101-probe-score-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X19-102-incoming | core | X19-102-incoming-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X19-102-probe-score | core | X19-102-probe-score-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X19-103-incoming | core | X19-103-incoming-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X19-103-probe-score | core | X19-103-probe-score-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X20-101-incoming | core | X20-101-incoming-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X20-101-probe-score | core | X20-101-probe-score-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X20-102-incoming | core | X20-102-incoming-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X20-102-probe-score | core | X20-102-probe-score-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X20-103-incoming | core | X20-103-incoming-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X20-103-probe-score | core | X20-103-probe-score-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X15-control-101-train-stage | core | X15-101-incoming, X15-101-probe-score | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X15-challenger-101-train-stage | core | X15-101-incoming, X15-101-probe-score | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X15-control-102-train-stage | core | X15-102-incoming, X15-102-probe-score | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X15-challenger-102-train-stage | core | X15-102-incoming, X15-102-probe-score | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X15-control-103-train-stage | core | X15-103-incoming, X15-103-probe-score | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X15-challenger-103-train-stage | core | X15-103-incoming, X15-103-probe-score | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X16-control-101-train-stage | core | X16-101-incoming, X16-101-probe-score | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X16-challenger-101-train-stage | core | X16-101-incoming, X16-101-probe-score | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X16-control-102-train-stage | core | X16-102-incoming, X16-102-probe-score | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X16-challenger-102-train-stage | core | X16-102-incoming, X16-102-probe-score | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X16-control-103-train-stage | core | X16-103-incoming, X16-103-probe-score | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X16-challenger-103-train-stage | core | X16-103-incoming, X16-103-probe-score | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X17-control-101-train-stage | core | X17-101-incoming, X17-101-probe-score | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X17-challenger-101-train-stage | core | X17-101-incoming, X17-101-probe-score | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X17-control-102-train-stage | core | X17-102-incoming, X17-102-probe-score | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X17-challenger-102-train-stage | core | X17-102-incoming, X17-102-probe-score | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X17-control-103-train-stage | core | X17-103-incoming, X17-103-probe-score | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X17-challenger-103-train-stage | core | X17-103-incoming, X17-103-probe-score | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X18-control-101-train-stage | core | X18-101-incoming, X18-101-probe-score | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X18-challenger-101-train-stage | core | X18-101-incoming, X18-101-probe-score | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X18-control-102-train-stage | core | X18-102-incoming, X18-102-probe-score | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X18-challenger-102-train-stage | core | X18-102-incoming, X18-102-probe-score | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X18-control-103-train-stage | core | X18-103-incoming, X18-103-probe-score | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X18-challenger-103-train-stage | core | X18-103-incoming, X18-103-probe-score | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X19-control-101-train-stage | core | X19-101-incoming, X19-101-probe-score | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X19-challenger-101-train-stage | core | X19-101-incoming, X19-101-probe-score | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X19-control-102-train-stage | core | X19-102-incoming, X19-102-probe-score | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X19-challenger-102-train-stage | core | X19-102-incoming, X19-102-probe-score | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X19-control-103-train-stage | core | X19-103-incoming, X19-103-probe-score | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X19-challenger-103-train-stage | core | X19-103-incoming, X19-103-probe-score | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X20-control-101-train-stage | core | X20-101-incoming, X20-101-probe-score | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X20-challenger-101-train-stage | core | X20-101-incoming, X20-101-probe-score | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X20-control-102-train-stage | core | X20-102-incoming, X20-102-probe-score | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X20-challenger-102-train-stage | core | X20-102-incoming, X20-102-probe-score | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X20-control-103-train-stage | core | X20-103-incoming, X20-103-probe-score | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X20-challenger-103-train-stage | core | X20-103-incoming, X20-103-probe-score | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X15-control-101-train-gpu | core | X15-control-101-train-stage | afterok | 8 | 1 | 32 | no | 2 | None; split stage; included in full-stage envelope, measured split pending |
| X15-challenger-101-train-gpu | core | X15-challenger-101-train-stage | afterok | 8 | 1 | 32 | no | 2 | None; split stage; included in full-stage envelope, measured split pending |
| X15-control-102-train-gpu | core | X15-control-102-train-stage | afterok | 8 | 1 | 32 | no | 2 | None; split stage; included in full-stage envelope, measured split pending |
| X15-challenger-102-train-gpu | core | X15-challenger-102-train-stage | afterok | 8 | 1 | 32 | no | 2 | None; split stage; included in full-stage envelope, measured split pending |
| X15-control-103-train-gpu | core | X15-control-103-train-stage | afterok | 8 | 1 | 32 | no | 2 | None; split stage; included in full-stage envelope, measured split pending |
| X15-challenger-103-train-gpu | core | X15-challenger-103-train-stage | afterok | 8 | 1 | 32 | no | 2 | None; split stage; included in full-stage envelope, measured split pending |
| X16-control-101-train-gpu | core | X16-control-101-train-stage | afterok | 8 | 1 | 32 | no | 2 | [7.733333333333333, 7.733333333333333]; core 58 min x 8 GPU proxy/full stage |
| X16-challenger-101-train-gpu | core | X16-challenger-101-train-stage | afterok | 8 | 1 | 32 | no | 2 | [7.733333333333333, 7.733333333333333]; core 58 min x 8 GPU proxy/full stage |
| X16-control-102-train-gpu | core | X16-control-102-train-stage | afterok | 8 | 1 | 32 | no | 2 | [7.733333333333333, 7.733333333333333]; core 58 min x 8 GPU proxy/full stage |
| X16-challenger-102-train-gpu | core | X16-challenger-102-train-stage | afterok | 8 | 1 | 32 | no | 2 | [7.733333333333333, 7.733333333333333]; core 58 min x 8 GPU proxy/full stage |
| X16-control-103-train-gpu | core | X16-control-103-train-stage | afterok | 8 | 1 | 32 | no | 2 | [7.733333333333333, 7.733333333333333]; core 58 min x 8 GPU proxy/full stage |
| X16-challenger-103-train-gpu | core | X16-challenger-103-train-stage | afterok | 8 | 1 | 32 | no | 2 | [7.733333333333333, 7.733333333333333]; core 58 min x 8 GPU proxy/full stage |
| X17-control-101-train-gpu | core | X17-control-101-train-stage | afterok | 8 | 1 | 32 | no | 2 | None; split stage; included in full-stage envelope, measured split pending |
| X17-challenger-101-train-gpu | core | X17-challenger-101-train-stage | afterok | 8 | 1 | 32 | no | 2 | None; split stage; included in full-stage envelope, measured split pending |
| X17-control-102-train-gpu | core | X17-control-102-train-stage | afterok | 8 | 1 | 32 | no | 2 | None; split stage; included in full-stage envelope, measured split pending |
| X17-challenger-102-train-gpu | core | X17-challenger-102-train-stage | afterok | 8 | 1 | 32 | no | 2 | None; split stage; included in full-stage envelope, measured split pending |
| X17-control-103-train-gpu | core | X17-control-103-train-stage | afterok | 8 | 1 | 32 | no | 2 | None; split stage; included in full-stage envelope, measured split pending |
| X17-challenger-103-train-gpu | core | X17-challenger-103-train-stage | afterok | 8 | 1 | 32 | no | 2 | None; split stage; included in full-stage envelope, measured split pending |
| X18-control-101-train-gpu | core | X18-control-101-train-stage | afterok | 8 | 1 | 32 | no | 2 | None; split stage; included in full-stage envelope, measured split pending |
| X18-challenger-101-train-gpu | core | X18-challenger-101-train-stage | afterok | 8 | 1 | 32 | no | 2 | None; split stage; included in full-stage envelope, measured split pending |
| X18-control-102-train-gpu | core | X18-control-102-train-stage | afterok | 8 | 1 | 32 | no | 2 | None; split stage; included in full-stage envelope, measured split pending |
| X18-challenger-102-train-gpu | core | X18-challenger-102-train-stage | afterok | 8 | 1 | 32 | no | 2 | None; split stage; included in full-stage envelope, measured split pending |
| X18-control-103-train-gpu | core | X18-control-103-train-stage | afterok | 8 | 1 | 32 | no | 2 | None; split stage; included in full-stage envelope, measured split pending |
| X18-challenger-103-train-gpu | core | X18-challenger-103-train-stage | afterok | 8 | 1 | 32 | no | 2 | None; split stage; included in full-stage envelope, measured split pending |
| X19-control-101-train-gpu | core | X19-control-101-train-stage | afterok | 8 | 1 | 32 | no | 2 | None; split stage; included in full-stage envelope, measured split pending |
| X19-challenger-101-train-gpu | core | X19-challenger-101-train-stage | afterok | 8 | 1 | 32 | no | 2 | None; split stage; included in full-stage envelope, measured split pending |
| X19-control-102-train-gpu | core | X19-control-102-train-stage | afterok | 8 | 1 | 32 | no | 2 | None; split stage; included in full-stage envelope, measured split pending |
| X19-challenger-102-train-gpu | core | X19-challenger-102-train-stage | afterok | 8 | 1 | 32 | no | 2 | None; split stage; included in full-stage envelope, measured split pending |
| X19-control-103-train-gpu | core | X19-control-103-train-stage | afterok | 8 | 1 | 32 | no | 2 | None; split stage; included in full-stage envelope, measured split pending |
| X19-challenger-103-train-gpu | core | X19-challenger-103-train-stage | afterok | 8 | 1 | 32 | no | 2 | None; split stage; included in full-stage envelope, measured split pending |
| X20-control-101-train-gpu | core | X20-control-101-train-stage | afterok | 8 | 1 | 32 | no | 2 | None; split stage; included in full-stage envelope, measured split pending |
| X20-challenger-101-train-gpu | core | X20-challenger-101-train-stage | afterok | 8 | 1 | 32 | no | 2 | None; split stage; included in full-stage envelope, measured split pending |
| X20-control-102-train-gpu | core | X20-control-102-train-stage | afterok | 8 | 1 | 32 | no | 2 | None; split stage; included in full-stage envelope, measured split pending |
| X20-challenger-102-train-gpu | core | X20-challenger-102-train-stage | afterok | 8 | 1 | 32 | no | 2 | None; split stage; included in full-stage envelope, measured split pending |
| X20-control-103-train-gpu | core | X20-control-103-train-stage | afterok | 8 | 1 | 32 | no | 2 | None; split stage; included in full-stage envelope, measured split pending |
| X20-challenger-103-train-gpu | core | X20-challenger-103-train-stage | afterok | 8 | 1 | 32 | no | 2 | None; split stage; included in full-stage envelope, measured split pending |
| X15-control-101-train | core | X15-control-101-train-gpu | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X15-challenger-101-train | core | X15-challenger-101-train-gpu | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X15-control-102-train | core | X15-control-102-train-gpu | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X15-challenger-102-train | core | X15-challenger-102-train-gpu | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X15-control-103-train | core | X15-control-103-train-gpu | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X15-challenger-103-train | core | X15-challenger-103-train-gpu | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X16-control-101-train | core | X16-control-101-train-gpu | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X16-challenger-101-train | core | X16-challenger-101-train-gpu | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X16-control-102-train | core | X16-control-102-train-gpu | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X16-challenger-102-train | core | X16-challenger-102-train-gpu | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X16-control-103-train | core | X16-control-103-train-gpu | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X16-challenger-103-train | core | X16-challenger-103-train-gpu | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X17-control-101-train | core | X17-control-101-train-gpu | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X17-challenger-101-train | core | X17-challenger-101-train-gpu | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X17-control-102-train | core | X17-control-102-train-gpu | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X17-challenger-102-train | core | X17-challenger-102-train-gpu | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X17-control-103-train | core | X17-control-103-train-gpu | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X17-challenger-103-train | core | X17-challenger-103-train-gpu | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X18-control-101-train | core | X18-control-101-train-gpu | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X18-challenger-101-train | core | X18-challenger-101-train-gpu | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X18-control-102-train | core | X18-control-102-train-gpu | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X18-challenger-102-train | core | X18-challenger-102-train-gpu | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X18-control-103-train | core | X18-control-103-train-gpu | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X18-challenger-103-train | core | X18-challenger-103-train-gpu | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X19-control-101-train | core | X19-control-101-train-gpu | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X19-challenger-101-train | core | X19-challenger-101-train-gpu | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X19-control-102-train | core | X19-control-102-train-gpu | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X19-challenger-102-train | core | X19-challenger-102-train-gpu | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X19-control-103-train | core | X19-control-103-train-gpu | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X19-challenger-103-train | core | X19-challenger-103-train-gpu | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X20-control-101-train | core | X20-control-101-train-gpu | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X20-challenger-101-train | core | X20-challenger-101-train-gpu | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X20-control-102-train | core | X20-control-102-train-gpu | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X20-challenger-102-train | core | X20-challenger-102-train-gpu | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X20-control-103-train | core | X20-control-103-train-gpu | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X20-challenger-103-train | core | X20-challenger-103-train-gpu | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X15-control-101-train-measure10-stage | core | X15-control-101-train, FV-x-tier2.stack-drift-101 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X15-challenger-101-train-measure10-stage | core | X15-challenger-101-train, FV-x-tier2.stack-drift-101 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X15-control-102-train-measure10-stage | core | X15-control-102-train, FV-x-tier2.stack-drift-102 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X15-challenger-102-train-measure10-stage | core | X15-challenger-102-train, FV-x-tier2.stack-drift-102 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X15-control-103-train-measure10-stage | core | X15-control-103-train, FV-x-tier2.stack-drift-103 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X15-challenger-103-train-measure10-stage | core | X15-challenger-103-train, FV-x-tier2.stack-drift-103 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X16-control-101-train-score-stage | core | X16-control-101-train, FV-x-tier2.stack-plasticity-101 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X16-challenger-101-train-score-stage | core | X16-challenger-101-train, FV-x-tier2.stack-plasticity-101 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X16-control-102-train-score-stage | core | X16-control-102-train, FV-x-tier2.stack-plasticity-102 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X16-challenger-102-train-score-stage | core | X16-challenger-102-train, FV-x-tier2.stack-plasticity-102 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X16-control-103-train-score-stage | core | X16-control-103-train, FV-x-tier2.stack-plasticity-103 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X16-challenger-103-train-score-stage | core | X16-challenger-103-train, FV-x-tier2.stack-plasticity-103 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X17-control-101-train-measure10-stage | core | X17-control-101-train, FV-x-tier2.b-101 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X17-challenger-101-train-measure10-stage | core | X17-challenger-101-train, FV-x-tier2.b-101 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X17-control-102-train-measure10-stage | core | X17-control-102-train, FV-x-tier2.b-102 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X17-challenger-102-train-measure10-stage | core | X17-challenger-102-train, FV-x-tier2.b-102 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X17-control-103-train-measure10-stage | core | X17-control-103-train, FV-x-tier2.b-103 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X17-challenger-103-train-measure10-stage | core | X17-challenger-103-train, FV-x-tier2.b-103 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X18-control-101-train-measure10-stage | core | X18-control-101-train, FV-x-tier2.b-p2-101 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X18-challenger-101-train-measure10-stage | core | X18-challenger-101-train, FV-x-tier2.b-p2-101 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X18-control-102-train-measure10-stage | core | X18-control-102-train, FV-x-tier2.b-p2-102 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X18-challenger-102-train-measure10-stage | core | X18-challenger-102-train, FV-x-tier2.b-p2-102 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X18-control-103-train-measure10-stage | core | X18-control-103-train, FV-x-tier2.b-p2-103 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X18-challenger-103-train-measure10-stage | core | X18-challenger-103-train, FV-x-tier2.b-p2-103 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X19-control-101-train-measure10-stage | core | X19-control-101-train, FV-x-tier2.incremental-101 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X19-challenger-101-continue10-stage | core | X19-challenger-101-train | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X19-control-102-train-measure10-stage | core | X19-control-102-train, FV-x-tier2.incremental-102 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X19-challenger-102-continue10-stage | core | X19-challenger-102-train | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X19-control-103-train-measure10-stage | core | X19-control-103-train, FV-x-tier2.incremental-103 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X19-challenger-103-continue10-stage | core | X19-challenger-103-train | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X20-control-101-train-measure10-stage | core | X20-control-101-train, FV-x-tier2.incremental-p2-101 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X20-challenger-101-continue10-stage | core | X20-challenger-101-train | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X20-control-102-train-measure10-stage | core | X20-control-102-train, FV-x-tier2.incremental-p2-102 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X20-challenger-102-continue10-stage | core | X20-challenger-102-train | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X20-control-103-train-measure10-stage | core | X20-control-103-train, FV-x-tier2.incremental-p2-103 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X20-challenger-103-continue10-stage | core | X20-challenger-103-train | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X15-control-101-train-measure10-gpu | core | X15-control-101-train-measure10-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X15-challenger-101-train-measure10-gpu | core | X15-challenger-101-train-measure10-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X15-control-102-train-measure10-gpu | core | X15-control-102-train-measure10-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X15-challenger-102-train-measure10-gpu | core | X15-challenger-102-train-measure10-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X15-control-103-train-measure10-gpu | core | X15-control-103-train-measure10-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X15-challenger-103-train-measure10-gpu | core | X15-challenger-103-train-measure10-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X16-control-101-train-score-gpu | core | X16-control-101-train-score-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X16-challenger-101-train-score-gpu | core | X16-challenger-101-train-score-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X16-control-102-train-score-gpu | core | X16-control-102-train-score-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X16-challenger-102-train-score-gpu | core | X16-challenger-102-train-score-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X16-control-103-train-score-gpu | core | X16-control-103-train-score-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X16-challenger-103-train-score-gpu | core | X16-challenger-103-train-score-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X17-control-101-train-measure10-gpu | core | X17-control-101-train-measure10-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X17-challenger-101-train-measure10-gpu | core | X17-challenger-101-train-measure10-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X17-control-102-train-measure10-gpu | core | X17-control-102-train-measure10-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X17-challenger-102-train-measure10-gpu | core | X17-challenger-102-train-measure10-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X17-control-103-train-measure10-gpu | core | X17-control-103-train-measure10-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X17-challenger-103-train-measure10-gpu | core | X17-challenger-103-train-measure10-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X18-control-101-train-measure10-gpu | core | X18-control-101-train-measure10-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X18-challenger-101-train-measure10-gpu | core | X18-challenger-101-train-measure10-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X18-control-102-train-measure10-gpu | core | X18-control-102-train-measure10-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X18-challenger-102-train-measure10-gpu | core | X18-challenger-102-train-measure10-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X18-control-103-train-measure10-gpu | core | X18-control-103-train-measure10-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X18-challenger-103-train-measure10-gpu | core | X18-challenger-103-train-measure10-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X19-control-101-train-measure10-gpu | core | X19-control-101-train-measure10-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X19-challenger-101-continue10-gpu | core | X19-challenger-101-continue10-stage | afterok | 8 | 1 | 32 | no | 2 | None; split stage; included in full-stage envelope, measured split pending |
| X19-control-102-train-measure10-gpu | core | X19-control-102-train-measure10-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X19-challenger-102-continue10-gpu | core | X19-challenger-102-continue10-stage | afterok | 8 | 1 | 32 | no | 2 | None; split stage; included in full-stage envelope, measured split pending |
| X19-control-103-train-measure10-gpu | core | X19-control-103-train-measure10-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X19-challenger-103-continue10-gpu | core | X19-challenger-103-continue10-stage | afterok | 8 | 1 | 32 | no | 2 | None; split stage; included in full-stage envelope, measured split pending |
| X20-control-101-train-measure10-gpu | core | X20-control-101-train-measure10-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X20-challenger-101-continue10-gpu | core | X20-challenger-101-continue10-stage | afterok | 8 | 1 | 32 | no | 2 | None; split stage; included in full-stage envelope, measured split pending |
| X20-control-102-train-measure10-gpu | core | X20-control-102-train-measure10-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X20-challenger-102-continue10-gpu | core | X20-challenger-102-continue10-stage | afterok | 8 | 1 | 32 | no | 2 | None; split stage; included in full-stage envelope, measured split pending |
| X20-control-103-train-measure10-gpu | core | X20-control-103-train-measure10-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X20-challenger-103-continue10-gpu | core | X20-challenger-103-continue10-stage | afterok | 8 | 1 | 32 | no | 2 | None; split stage; included in full-stage envelope, measured split pending |
| X15-control-101-train-measure10 | core | X15-control-101-train-measure10-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X15-challenger-101-train-measure10 | core | X15-challenger-101-train-measure10-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X15-control-102-train-measure10 | core | X15-control-102-train-measure10-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X15-challenger-102-train-measure10 | core | X15-challenger-102-train-measure10-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X15-control-103-train-measure10 | core | X15-control-103-train-measure10-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X15-challenger-103-train-measure10 | core | X15-challenger-103-train-measure10-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X16-control-101-train-score | core | X16-control-101-train-score-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X16-challenger-101-train-score | core | X16-challenger-101-train-score-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X16-control-102-train-score | core | X16-control-102-train-score-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X16-challenger-102-train-score | core | X16-challenger-102-train-score-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X16-control-103-train-score | core | X16-control-103-train-score-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X16-challenger-103-train-score | core | X16-challenger-103-train-score-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X17-control-101-train-measure10 | core | X17-control-101-train-measure10-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X17-challenger-101-train-measure10 | core | X17-challenger-101-train-measure10-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X17-control-102-train-measure10 | core | X17-control-102-train-measure10-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X17-challenger-102-train-measure10 | core | X17-challenger-102-train-measure10-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X17-control-103-train-measure10 | core | X17-control-103-train-measure10-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X17-challenger-103-train-measure10 | core | X17-challenger-103-train-measure10-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X18-control-101-train-measure10 | core | X18-control-101-train-measure10-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X18-challenger-101-train-measure10 | core | X18-challenger-101-train-measure10-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X18-control-102-train-measure10 | core | X18-control-102-train-measure10-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X18-challenger-102-train-measure10 | core | X18-challenger-102-train-measure10-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X18-control-103-train-measure10 | core | X18-control-103-train-measure10-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X18-challenger-103-train-measure10 | core | X18-challenger-103-train-measure10-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X19-control-101-train-measure10 | core | X19-control-101-train-measure10-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X19-challenger-101-continue10 | core | X19-challenger-101-continue10-gpu | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X19-control-102-train-measure10 | core | X19-control-102-train-measure10-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X19-challenger-102-continue10 | core | X19-challenger-102-continue10-gpu | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X19-control-103-train-measure10 | core | X19-control-103-train-measure10-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X19-challenger-103-continue10 | core | X19-challenger-103-continue10-gpu | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X20-control-101-train-measure10 | core | X20-control-101-train-measure10-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X20-challenger-101-continue10 | core | X20-challenger-101-continue10-gpu | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X20-control-102-train-measure10 | core | X20-control-102-train-measure10-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X20-challenger-102-continue10 | core | X20-challenger-102-continue10-gpu | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X20-control-103-train-measure10 | core | X20-control-103-train-measure10-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X20-challenger-103-continue10 | core | X20-challenger-103-continue10-gpu | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X15-control-101-continue10-stage | core | X15-control-101-train-measure10 | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X15-challenger-101-continue10-stage | core | X15-challenger-101-train-measure10 | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X15-control-102-continue10-stage | core | X15-control-102-train-measure10 | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X15-challenger-102-continue10-stage | core | X15-challenger-102-train-measure10 | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X15-control-103-continue10-stage | core | X15-control-103-train-measure10 | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X15-challenger-103-continue10-stage | core | X15-challenger-103-train-measure10 | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X17-control-101-continue10-stage | core | X17-control-101-train-measure10 | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X17-challenger-101-continue10-stage | core | X17-challenger-101-train-measure10 | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X17-control-102-continue10-stage | core | X17-control-102-train-measure10 | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X17-challenger-102-continue10-stage | core | X17-challenger-102-train-measure10 | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X17-control-103-continue10-stage | core | X17-control-103-train-measure10 | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X17-challenger-103-continue10-stage | core | X17-challenger-103-train-measure10 | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X18-control-101-continue10-stage | core | X18-control-101-train-measure10 | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X18-challenger-101-continue10-stage | core | X18-challenger-101-train-measure10 | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X18-control-102-continue10-stage | core | X18-control-102-train-measure10 | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X18-challenger-102-continue10-stage | core | X18-challenger-102-train-measure10 | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X18-control-103-continue10-stage | core | X18-control-103-train-measure10 | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X18-challenger-103-continue10-stage | core | X18-challenger-103-train-measure10 | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X19-control-101-continue10-stage | core | X19-control-101-train-measure10 | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X19-challenger-101-continue10-score-stage | core | X19-challenger-101-continue10, FV-x-tier2.incremental-101 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X19-control-102-continue10-stage | core | X19-control-102-train-measure10 | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X19-challenger-102-continue10-score-stage | core | X19-challenger-102-continue10, FV-x-tier2.incremental-102 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X19-control-103-continue10-stage | core | X19-control-103-train-measure10 | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X19-challenger-103-continue10-score-stage | core | X19-challenger-103-continue10, FV-x-tier2.incremental-103 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X20-control-101-continue10-stage | core | X20-control-101-train-measure10 | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X20-challenger-101-continue10-score-stage | core | X20-challenger-101-continue10, FV-x-tier2.incremental-p2-101 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X20-control-102-continue10-stage | core | X20-control-102-train-measure10 | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X20-challenger-102-continue10-score-stage | core | X20-challenger-102-continue10, FV-x-tier2.incremental-p2-102 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X20-control-103-continue10-stage | core | X20-control-103-train-measure10 | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X20-challenger-103-continue10-score-stage | core | X20-challenger-103-continue10, FV-x-tier2.incremental-p2-103 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X16-complete-101-stage | core | X16-control-101-train-score, X16-challenger-101-train-score, FV-x-tier2.stack-plasticity-101 | afterok | 0 | 1 | 8 | yes | 39 with both arms READY | [0, 0]; CPU/handoff; no GPU compute |
| X16-complete-102-stage | core | X16-control-102-train-score, X16-challenger-102-train-score, FV-x-tier2.stack-plasticity-102 | afterok | 0 | 1 | 8 | yes | 39 with both arms READY | [0, 0]; CPU/handoff; no GPU compute |
| X16-complete-103-stage | core | X16-control-103-train-score, X16-challenger-103-train-score, FV-x-tier2.stack-plasticity-103 | afterok | 0 | 1 | 8 | yes | 39 with both arms READY | [0, 0]; CPU/handoff; no GPU compute |
| X15-control-101-continue10-gpu | core | X15-control-101-continue10-stage | afterok | 8 | 1 | 32 | no | 2 | None; split stage; included in full-stage envelope, measured split pending |
| X15-challenger-101-continue10-gpu | core | X15-challenger-101-continue10-stage | afterok | 8 | 1 | 32 | no | 2 | None; split stage; included in full-stage envelope, measured split pending |
| X15-control-102-continue10-gpu | core | X15-control-102-continue10-stage | afterok | 8 | 1 | 32 | no | 2 | None; split stage; included in full-stage envelope, measured split pending |
| X15-challenger-102-continue10-gpu | core | X15-challenger-102-continue10-stage | afterok | 8 | 1 | 32 | no | 2 | None; split stage; included in full-stage envelope, measured split pending |
| X15-control-103-continue10-gpu | core | X15-control-103-continue10-stage | afterok | 8 | 1 | 32 | no | 2 | None; split stage; included in full-stage envelope, measured split pending |
| X15-challenger-103-continue10-gpu | core | X15-challenger-103-continue10-stage | afterok | 8 | 1 | 32 | no | 2 | None; split stage; included in full-stage envelope, measured split pending |
| X17-control-101-continue10-gpu | core | X17-control-101-continue10-stage | afterok | 8 | 1 | 32 | no | 2 | None; split stage; included in full-stage envelope, measured split pending |
| X17-challenger-101-continue10-gpu | core | X17-challenger-101-continue10-stage | afterok | 8 | 1 | 32 | no | 2 | None; split stage; included in full-stage envelope, measured split pending |
| X17-control-102-continue10-gpu | core | X17-control-102-continue10-stage | afterok | 8 | 1 | 32 | no | 2 | None; split stage; included in full-stage envelope, measured split pending |
| X17-challenger-102-continue10-gpu | core | X17-challenger-102-continue10-stage | afterok | 8 | 1 | 32 | no | 2 | None; split stage; included in full-stage envelope, measured split pending |
| X17-control-103-continue10-gpu | core | X17-control-103-continue10-stage | afterok | 8 | 1 | 32 | no | 2 | None; split stage; included in full-stage envelope, measured split pending |
| X17-challenger-103-continue10-gpu | core | X17-challenger-103-continue10-stage | afterok | 8 | 1 | 32 | no | 2 | None; split stage; included in full-stage envelope, measured split pending |
| X18-control-101-continue10-gpu | core | X18-control-101-continue10-stage | afterok | 8 | 1 | 32 | no | 2 | None; split stage; included in full-stage envelope, measured split pending |
| X18-challenger-101-continue10-gpu | core | X18-challenger-101-continue10-stage | afterok | 8 | 1 | 32 | no | 2 | None; split stage; included in full-stage envelope, measured split pending |
| X18-control-102-continue10-gpu | core | X18-control-102-continue10-stage | afterok | 8 | 1 | 32 | no | 2 | None; split stage; included in full-stage envelope, measured split pending |
| X18-challenger-102-continue10-gpu | core | X18-challenger-102-continue10-stage | afterok | 8 | 1 | 32 | no | 2 | None; split stage; included in full-stage envelope, measured split pending |
| X18-control-103-continue10-gpu | core | X18-control-103-continue10-stage | afterok | 8 | 1 | 32 | no | 2 | None; split stage; included in full-stage envelope, measured split pending |
| X18-challenger-103-continue10-gpu | core | X18-challenger-103-continue10-stage | afterok | 8 | 1 | 32 | no | 2 | None; split stage; included in full-stage envelope, measured split pending |
| X19-control-101-continue10-gpu | core | X19-control-101-continue10-stage | afterok | 8 | 1 | 32 | no | 2 | None; split stage; included in full-stage envelope, measured split pending |
| X19-challenger-101-continue10-score-gpu | core | X19-challenger-101-continue10-score-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X19-control-102-continue10-gpu | core | X19-control-102-continue10-stage | afterok | 8 | 1 | 32 | no | 2 | None; split stage; included in full-stage envelope, measured split pending |
| X19-challenger-102-continue10-score-gpu | core | X19-challenger-102-continue10-score-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X19-control-103-continue10-gpu | core | X19-control-103-continue10-stage | afterok | 8 | 1 | 32 | no | 2 | None; split stage; included in full-stage envelope, measured split pending |
| X19-challenger-103-continue10-score-gpu | core | X19-challenger-103-continue10-score-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X20-control-101-continue10-gpu | core | X20-control-101-continue10-stage | afterok | 8 | 1 | 32 | no | 2 | None; split stage; included in full-stage envelope, measured split pending |
| X20-challenger-101-continue10-score-gpu | core | X20-challenger-101-continue10-score-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X20-control-102-continue10-gpu | core | X20-control-102-continue10-stage | afterok | 8 | 1 | 32 | no | 2 | None; split stage; included in full-stage envelope, measured split pending |
| X20-challenger-102-continue10-score-gpu | core | X20-challenger-102-continue10-score-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X20-control-103-continue10-gpu | core | X20-control-103-continue10-stage | afterok | 8 | 1 | 32 | no | 2 | None; split stage; included in full-stage envelope, measured split pending |
| X20-challenger-103-continue10-score-gpu | core | X20-challenger-103-continue10-score-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X16-complete-101-gpu | core | X16-complete-101-stage | afterok | 1 | 1 | 8 | yes | 39 with both arms READY / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X16-complete-102-gpu | core | X16-complete-102-stage | afterok | 1 | 1 | 8 | yes | 39 with both arms READY / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X16-complete-103-gpu | core | X16-complete-103-stage | afterok | 1 | 1 | 8 | yes | 39 with both arms READY / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X15-control-101-continue10 | core | X15-control-101-continue10-gpu | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X15-challenger-101-continue10 | core | X15-challenger-101-continue10-gpu | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X15-control-102-continue10 | core | X15-control-102-continue10-gpu | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X15-challenger-102-continue10 | core | X15-challenger-102-continue10-gpu | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X15-control-103-continue10 | core | X15-control-103-continue10-gpu | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X15-challenger-103-continue10 | core | X15-challenger-103-continue10-gpu | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X17-control-101-continue10 | core | X17-control-101-continue10-gpu | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X17-challenger-101-continue10 | core | X17-challenger-101-continue10-gpu | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X17-control-102-continue10 | core | X17-control-102-continue10-gpu | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X17-challenger-102-continue10 | core | X17-challenger-102-continue10-gpu | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X17-control-103-continue10 | core | X17-control-103-continue10-gpu | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X17-challenger-103-continue10 | core | X17-challenger-103-continue10-gpu | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X18-control-101-continue10 | core | X18-control-101-continue10-gpu | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X18-challenger-101-continue10 | core | X18-challenger-101-continue10-gpu | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X18-control-102-continue10 | core | X18-control-102-continue10-gpu | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X18-challenger-102-continue10 | core | X18-challenger-102-continue10-gpu | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X18-control-103-continue10 | core | X18-control-103-continue10-gpu | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X18-challenger-103-continue10 | core | X18-challenger-103-continue10-gpu | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X19-control-101-continue10 | core | X19-control-101-continue10-gpu | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X19-challenger-101-continue10-score | core | X19-challenger-101-continue10-score-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X19-control-102-continue10 | core | X19-control-102-continue10-gpu | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X19-challenger-102-continue10-score | core | X19-challenger-102-continue10-score-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X19-control-103-continue10 | core | X19-control-103-continue10-gpu | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X19-challenger-103-continue10-score | core | X19-challenger-103-continue10-score-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X20-control-101-continue10 | core | X20-control-101-continue10-gpu | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X20-challenger-101-continue10-score | core | X20-challenger-101-continue10-score-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X20-control-102-continue10 | core | X20-control-102-continue10-gpu | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X20-challenger-102-continue10-score | core | X20-challenger-102-continue10-score-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X20-control-103-continue10 | core | X20-control-103-continue10-gpu | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X20-challenger-103-continue10-score | core | X20-challenger-103-continue10-score-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X16-complete-101 | core | X16-complete-101-gpu | afterok | 0 | 1 | 8 | yes | 39 with both arms READY | [0, 0]; CPU/handoff; no GPU compute |
| X16-complete-102 | core | X16-complete-102-gpu | afterok | 0 | 1 | 8 | yes | 39 with both arms READY | [0, 0]; CPU/handoff; no GPU compute |
| X16-complete-103 | core | X16-complete-103-gpu | afterok | 0 | 1 | 8 | yes | 39 with both arms READY | [0, 0]; CPU/handoff; no GPU compute |
| X15-control-101-continue10-score-stage | core | X15-control-101-continue10, FV-x-tier2.stack-drift-101 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X15-challenger-101-continue10-score-stage | core | X15-challenger-101-continue10, FV-x-tier2.stack-drift-101 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X15-control-102-continue10-score-stage | core | X15-control-102-continue10, FV-x-tier2.stack-drift-102 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X15-challenger-102-continue10-score-stage | core | X15-challenger-102-continue10, FV-x-tier2.stack-drift-102 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X15-control-103-continue10-score-stage | core | X15-control-103-continue10, FV-x-tier2.stack-drift-103 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X15-challenger-103-continue10-score-stage | core | X15-challenger-103-continue10, FV-x-tier2.stack-drift-103 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X17-control-101-continue10-score-stage | core | X17-control-101-continue10, FV-x-tier2.b-101 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X17-challenger-101-continue10-measure20-stage | core | X17-challenger-101-continue10, FV-x-tier2.b-101 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X17-control-102-continue10-score-stage | core | X17-control-102-continue10, FV-x-tier2.b-102 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X17-challenger-102-continue10-measure20-stage | core | X17-challenger-102-continue10, FV-x-tier2.b-102 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X17-control-103-continue10-score-stage | core | X17-control-103-continue10, FV-x-tier2.b-103 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X17-challenger-103-continue10-measure20-stage | core | X17-challenger-103-continue10, FV-x-tier2.b-103 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X18-control-101-continue10-score-stage | core | X18-control-101-continue10, FV-x-tier2.b-p2-101 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X18-challenger-101-continue10-measure20-stage | core | X18-challenger-101-continue10, FV-x-tier2.b-p2-101 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X18-control-102-continue10-score-stage | core | X18-control-102-continue10, FV-x-tier2.b-p2-102 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X18-challenger-102-continue10-measure20-stage | core | X18-challenger-102-continue10, FV-x-tier2.b-p2-102 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X18-control-103-continue10-score-stage | core | X18-control-103-continue10, FV-x-tier2.b-p2-103 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X18-challenger-103-continue10-measure20-stage | core | X18-challenger-103-continue10, FV-x-tier2.b-p2-103 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X19-control-101-continue10-score-stage | core | X19-control-101-continue10, FV-x-tier2.incremental-101 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X19-control-102-continue10-score-stage | core | X19-control-102-continue10, FV-x-tier2.incremental-102 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X19-control-103-continue10-score-stage | core | X19-control-103-continue10, FV-x-tier2.incremental-103 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X20-control-101-continue10-score-stage | core | X20-control-101-continue10, FV-x-tier2.incremental-p2-101 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X20-control-102-continue10-score-stage | core | X20-control-102-continue10, FV-x-tier2.incremental-p2-102 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X20-control-103-continue10-score-stage | core | X20-control-103-continue10, FV-x-tier2.incremental-p2-103 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X15-control-101-continue10-score-gpu | core | X15-control-101-continue10-score-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X15-challenger-101-continue10-score-gpu | core | X15-challenger-101-continue10-score-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X15-control-102-continue10-score-gpu | core | X15-control-102-continue10-score-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X15-challenger-102-continue10-score-gpu | core | X15-challenger-102-continue10-score-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X15-control-103-continue10-score-gpu | core | X15-control-103-continue10-score-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X15-challenger-103-continue10-score-gpu | core | X15-challenger-103-continue10-score-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X17-control-101-continue10-score-gpu | core | X17-control-101-continue10-score-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X17-challenger-101-continue10-measure20-gpu | core | X17-challenger-101-continue10-measure20-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X17-control-102-continue10-score-gpu | core | X17-control-102-continue10-score-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X17-challenger-102-continue10-measure20-gpu | core | X17-challenger-102-continue10-measure20-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X17-control-103-continue10-score-gpu | core | X17-control-103-continue10-score-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X17-challenger-103-continue10-measure20-gpu | core | X17-challenger-103-continue10-measure20-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X18-control-101-continue10-score-gpu | core | X18-control-101-continue10-score-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X18-challenger-101-continue10-measure20-gpu | core | X18-challenger-101-continue10-measure20-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X18-control-102-continue10-score-gpu | core | X18-control-102-continue10-score-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X18-challenger-102-continue10-measure20-gpu | core | X18-challenger-102-continue10-measure20-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X18-control-103-continue10-score-gpu | core | X18-control-103-continue10-score-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X18-challenger-103-continue10-measure20-gpu | core | X18-challenger-103-continue10-measure20-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X19-control-101-continue10-score-gpu | core | X19-control-101-continue10-score-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X19-control-102-continue10-score-gpu | core | X19-control-102-continue10-score-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X19-control-103-continue10-score-gpu | core | X19-control-103-continue10-score-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X20-control-101-continue10-score-gpu | core | X20-control-101-continue10-score-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X20-control-102-continue10-score-gpu | core | X20-control-102-continue10-score-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X20-control-103-continue10-score-gpu | core | X20-control-103-continue10-score-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X15-control-101-continue10-score | core | X15-control-101-continue10-score-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X15-challenger-101-continue10-score | core | X15-challenger-101-continue10-score-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X15-control-102-continue10-score | core | X15-control-102-continue10-score-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X15-challenger-102-continue10-score | core | X15-challenger-102-continue10-score-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X15-control-103-continue10-score | core | X15-control-103-continue10-score-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X15-challenger-103-continue10-score | core | X15-challenger-103-continue10-score-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X17-control-101-continue10-score | core | X17-control-101-continue10-score-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X17-challenger-101-continue10-measure20 | core | X17-challenger-101-continue10-measure20-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X17-control-102-continue10-score | core | X17-control-102-continue10-score-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X17-challenger-102-continue10-measure20 | core | X17-challenger-102-continue10-measure20-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X17-control-103-continue10-score | core | X17-control-103-continue10-score-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X17-challenger-103-continue10-measure20 | core | X17-challenger-103-continue10-measure20-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X18-control-101-continue10-score | core | X18-control-101-continue10-score-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X18-challenger-101-continue10-measure20 | core | X18-challenger-101-continue10-measure20-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X18-control-102-continue10-score | core | X18-control-102-continue10-score-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X18-challenger-102-continue10-measure20 | core | X18-challenger-102-continue10-measure20-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X18-control-103-continue10-score | core | X18-control-103-continue10-score-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X18-challenger-103-continue10-measure20 | core | X18-challenger-103-continue10-measure20-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X19-control-101-continue10-score | core | X19-control-101-continue10-score-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X19-control-102-continue10-score | core | X19-control-102-continue10-score-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X19-control-103-continue10-score | core | X19-control-103-continue10-score-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X20-control-101-continue10-score | core | X20-control-101-continue10-score-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X20-control-102-continue10-score | core | X20-control-102-continue10-score-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X20-control-103-continue10-score | core | X20-control-103-continue10-score-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X17-challenger-101-continue20-stage | core | X17-challenger-101-continue10-measure20 | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X17-challenger-102-continue20-stage | core | X17-challenger-102-continue10-measure20 | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X17-challenger-103-continue20-stage | core | X17-challenger-103-continue10-measure20 | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X18-challenger-101-continue20-stage | core | X18-challenger-101-continue10-measure20 | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X18-challenger-102-continue20-stage | core | X18-challenger-102-continue10-measure20 | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X18-challenger-103-continue20-stage | core | X18-challenger-103-continue10-measure20 | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X15-complete-101-stage | core | X15-control-101-continue10-score, X15-challenger-101-continue10-score, FV-x-tier2.stack-drift-101 | afterok | 0 | 1 | 8 | yes | 39 with both arms READY | [0, 0]; CPU/handoff; no GPU compute |
| X19-complete-101-stage | core | X19-control-101-continue10-score, X19-challenger-101-continue10-score, FV-x-tier2.incremental-101 | afterok | 0 | 1 | 8 | yes | 39 with both arms READY | [0, 0]; CPU/handoff; no GPU compute |
| X20-complete-101-stage | core | X20-control-101-continue10-score, X20-challenger-101-continue10-score, FV-x-tier2.incremental-p2-101 | afterok | 0 | 1 | 8 | yes | 39 with both arms READY | [0, 0]; CPU/handoff; no GPU compute |
| X15-complete-102-stage | core | X15-control-102-continue10-score, X15-challenger-102-continue10-score, FV-x-tier2.stack-drift-102 | afterok | 0 | 1 | 8 | yes | 39 with both arms READY | [0, 0]; CPU/handoff; no GPU compute |
| X19-complete-102-stage | core | X19-control-102-continue10-score, X19-challenger-102-continue10-score, FV-x-tier2.incremental-102 | afterok | 0 | 1 | 8 | yes | 39 with both arms READY | [0, 0]; CPU/handoff; no GPU compute |
| X20-complete-102-stage | core | X20-control-102-continue10-score, X20-challenger-102-continue10-score, FV-x-tier2.incremental-p2-102 | afterok | 0 | 1 | 8 | yes | 39 with both arms READY | [0, 0]; CPU/handoff; no GPU compute |
| X15-complete-103-stage | core | X15-control-103-continue10-score, X15-challenger-103-continue10-score, FV-x-tier2.stack-drift-103 | afterok | 0 | 1 | 8 | yes | 39 with both arms READY | [0, 0]; CPU/handoff; no GPU compute |
| X19-complete-103-stage | core | X19-control-103-continue10-score, X19-challenger-103-continue10-score, FV-x-tier2.incremental-103 | afterok | 0 | 1 | 8 | yes | 39 with both arms READY | [0, 0]; CPU/handoff; no GPU compute |
| X20-complete-103-stage | core | X20-control-103-continue10-score, X20-challenger-103-continue10-score, FV-x-tier2.incremental-p2-103 | afterok | 0 | 1 | 8 | yes | 39 with both arms READY | [0, 0]; CPU/handoff; no GPU compute |
| X17-challenger-101-continue20-gpu | core | X17-challenger-101-continue20-stage | afterok | 8 | 1 | 32 | no | 2 | None; split stage; included in full-stage envelope, measured split pending |
| X17-challenger-102-continue20-gpu | core | X17-challenger-102-continue20-stage | afterok | 8 | 1 | 32 | no | 2 | None; split stage; included in full-stage envelope, measured split pending |
| X17-challenger-103-continue20-gpu | core | X17-challenger-103-continue20-stage | afterok | 8 | 1 | 32 | no | 2 | None; split stage; included in full-stage envelope, measured split pending |
| X18-challenger-101-continue20-gpu | core | X18-challenger-101-continue20-stage | afterok | 8 | 1 | 32 | no | 2 | None; split stage; included in full-stage envelope, measured split pending |
| X18-challenger-102-continue20-gpu | core | X18-challenger-102-continue20-stage | afterok | 8 | 1 | 32 | no | 2 | None; split stage; included in full-stage envelope, measured split pending |
| X18-challenger-103-continue20-gpu | core | X18-challenger-103-continue20-stage | afterok | 8 | 1 | 32 | no | 2 | None; split stage; included in full-stage envelope, measured split pending |
| X15-complete-101-gpu | core | X15-complete-101-stage | afterok | 1 | 1 | 8 | yes | 39 with both arms READY / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X19-complete-101-gpu | core | X19-complete-101-stage | afterok | 1 | 1 | 8 | yes | 39 with both arms READY / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X20-complete-101-gpu | core | X20-complete-101-stage | afterok | 1 | 1 | 8 | yes | 39 with both arms READY / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X15-complete-102-gpu | core | X15-complete-102-stage | afterok | 1 | 1 | 8 | yes | 39 with both arms READY / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X19-complete-102-gpu | core | X19-complete-102-stage | afterok | 1 | 1 | 8 | yes | 39 with both arms READY / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X20-complete-102-gpu | core | X20-complete-102-stage | afterok | 1 | 1 | 8 | yes | 39 with both arms READY / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X15-complete-103-gpu | core | X15-complete-103-stage | afterok | 1 | 1 | 8 | yes | 39 with both arms READY / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X19-complete-103-gpu | core | X19-complete-103-stage | afterok | 1 | 1 | 8 | yes | 39 with both arms READY / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X20-complete-103-gpu | core | X20-complete-103-stage | afterok | 1 | 1 | 8 | yes | 39 with both arms READY / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X17-challenger-101-continue20 | core | X17-challenger-101-continue20-gpu | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X17-challenger-102-continue20 | core | X17-challenger-102-continue20-gpu | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X17-challenger-103-continue20 | core | X17-challenger-103-continue20-gpu | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X18-challenger-101-continue20 | core | X18-challenger-101-continue20-gpu | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X18-challenger-102-continue20 | core | X18-challenger-102-continue20-gpu | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X18-challenger-103-continue20 | core | X18-challenger-103-continue20-gpu | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X15-complete-101 | core | X15-complete-101-gpu | afterok | 0 | 1 | 8 | yes | 39 with both arms READY | [0, 0]; CPU/handoff; no GPU compute |
| X19-complete-101 | core | X19-complete-101-gpu | afterok | 0 | 1 | 8 | yes | 39 with both arms READY | [0, 0]; CPU/handoff; no GPU compute |
| X20-complete-101 | core | X20-complete-101-gpu | afterok | 0 | 1 | 8 | yes | 39 with both arms READY | [0, 0]; CPU/handoff; no GPU compute |
| X15-complete-102 | core | X15-complete-102-gpu | afterok | 0 | 1 | 8 | yes | 39 with both arms READY | [0, 0]; CPU/handoff; no GPU compute |
| X19-complete-102 | core | X19-complete-102-gpu | afterok | 0 | 1 | 8 | yes | 39 with both arms READY | [0, 0]; CPU/handoff; no GPU compute |
| X20-complete-102 | core | X20-complete-102-gpu | afterok | 0 | 1 | 8 | yes | 39 with both arms READY | [0, 0]; CPU/handoff; no GPU compute |
| X15-complete-103 | core | X15-complete-103-gpu | afterok | 0 | 1 | 8 | yes | 39 with both arms READY | [0, 0]; CPU/handoff; no GPU compute |
| X19-complete-103 | core | X19-complete-103-gpu | afterok | 0 | 1 | 8 | yes | 39 with both arms READY | [0, 0]; CPU/handoff; no GPU compute |
| X20-complete-103 | core | X20-complete-103-gpu | afterok | 0 | 1 | 8 | yes | 39 with both arms READY | [0, 0]; CPU/handoff; no GPU compute |
| X17-challenger-101-continue20-measure30-stage | core | X17-challenger-101-continue20, FV-x-tier2.b-101 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X17-challenger-102-continue20-measure30-stage | core | X17-challenger-102-continue20, FV-x-tier2.b-102 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X17-challenger-103-continue20-measure30-stage | core | X17-challenger-103-continue20, FV-x-tier2.b-103 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X18-challenger-101-continue20-measure30-stage | core | X18-challenger-101-continue20, FV-x-tier2.b-p2-101 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X18-challenger-102-continue20-measure30-stage | core | X18-challenger-102-continue20, FV-x-tier2.b-p2-102 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X18-challenger-103-continue20-measure30-stage | core | X18-challenger-103-continue20, FV-x-tier2.b-p2-103 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X17-challenger-101-continue20-measure30-gpu | core | X17-challenger-101-continue20-measure30-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X17-challenger-102-continue20-measure30-gpu | core | X17-challenger-102-continue20-measure30-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X17-challenger-103-continue20-measure30-gpu | core | X17-challenger-103-continue20-measure30-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X18-challenger-101-continue20-measure30-gpu | core | X18-challenger-101-continue20-measure30-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X18-challenger-102-continue20-measure30-gpu | core | X18-challenger-102-continue20-measure30-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X18-challenger-103-continue20-measure30-gpu | core | X18-challenger-103-continue20-measure30-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X17-challenger-101-continue20-measure30 | core | X17-challenger-101-continue20-measure30-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X17-challenger-102-continue20-measure30 | core | X17-challenger-102-continue20-measure30-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X17-challenger-103-continue20-measure30 | core | X17-challenger-103-continue20-measure30-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X18-challenger-101-continue20-measure30 | core | X18-challenger-101-continue20-measure30-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X18-challenger-102-continue20-measure30 | core | X18-challenger-102-continue20-measure30-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X18-challenger-103-continue20-measure30 | core | X18-challenger-103-continue20-measure30-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X17-challenger-101-continue30-stage | core | X17-challenger-101-continue20-measure30 | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X17-challenger-102-continue30-stage | core | X17-challenger-102-continue20-measure30 | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X17-challenger-103-continue30-stage | core | X17-challenger-103-continue20-measure30 | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X18-challenger-101-continue30-stage | core | X18-challenger-101-continue20-measure30 | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X18-challenger-102-continue30-stage | core | X18-challenger-102-continue20-measure30 | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X18-challenger-103-continue30-stage | core | X18-challenger-103-continue20-measure30 | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X17-challenger-101-continue30-gpu | core | X17-challenger-101-continue30-stage | afterok | 8 | 1 | 32 | no | 2 | None; split stage; included in full-stage envelope, measured split pending |
| X17-challenger-102-continue30-gpu | core | X17-challenger-102-continue30-stage | afterok | 8 | 1 | 32 | no | 2 | None; split stage; included in full-stage envelope, measured split pending |
| X17-challenger-103-continue30-gpu | core | X17-challenger-103-continue30-stage | afterok | 8 | 1 | 32 | no | 2 | None; split stage; included in full-stage envelope, measured split pending |
| X18-challenger-101-continue30-gpu | core | X18-challenger-101-continue30-stage | afterok | 8 | 1 | 32 | no | 2 | None; split stage; included in full-stage envelope, measured split pending |
| X18-challenger-102-continue30-gpu | core | X18-challenger-102-continue30-stage | afterok | 8 | 1 | 32 | no | 2 | None; split stage; included in full-stage envelope, measured split pending |
| X18-challenger-103-continue30-gpu | core | X18-challenger-103-continue30-stage | afterok | 8 | 1 | 32 | no | 2 | None; split stage; included in full-stage envelope, measured split pending |
| X17-challenger-101-continue30 | core | X17-challenger-101-continue30-gpu | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X17-challenger-102-continue30 | core | X17-challenger-102-continue30-gpu | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X17-challenger-103-continue30 | core | X17-challenger-103-continue30-gpu | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X18-challenger-101-continue30 | core | X18-challenger-101-continue30-gpu | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X18-challenger-102-continue30 | core | X18-challenger-102-continue30-gpu | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X18-challenger-103-continue30 | core | X18-challenger-103-continue30-gpu | afterok | 0 | 1 | 8 | yes | 2 | [0, 0]; CPU/handoff; no GPU compute |
| X17-challenger-101-continue30-measure40-stage | core | X17-challenger-101-continue30, FV-x-tier2.b-101 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X17-challenger-102-continue30-measure40-stage | core | X17-challenger-102-continue30, FV-x-tier2.b-102 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X17-challenger-103-continue30-measure40-stage | core | X17-challenger-103-continue30, FV-x-tier2.b-103 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X18-challenger-101-continue30-measure40-stage | core | X18-challenger-101-continue30, FV-x-tier2.b-p2-101 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X18-challenger-102-continue30-measure40-stage | core | X18-challenger-102-continue30, FV-x-tier2.b-p2-102 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X18-challenger-103-continue30-measure40-stage | core | X18-challenger-103-continue30, FV-x-tier2.b-p2-103 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X17-challenger-101-continue30-measure40-gpu | core | X17-challenger-101-continue30-measure40-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X17-challenger-102-continue30-measure40-gpu | core | X17-challenger-102-continue30-measure40-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X17-challenger-103-continue30-measure40-gpu | core | X17-challenger-103-continue30-measure40-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X18-challenger-101-continue30-measure40-gpu | core | X18-challenger-101-continue30-measure40-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X18-challenger-102-continue30-measure40-gpu | core | X18-challenger-102-continue30-measure40-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X18-challenger-103-continue30-measure40-gpu | core | X18-challenger-103-continue30-measure40-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X17-challenger-101-continue30-measure40 | core | X17-challenger-101-continue30-measure40-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X17-challenger-102-continue30-measure40 | core | X17-challenger-102-continue30-measure40-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X17-challenger-103-continue30-measure40 | core | X17-challenger-103-continue30-measure40-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X18-challenger-101-continue30-measure40 | core | X18-challenger-101-continue30-measure40-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X18-challenger-102-continue30-measure40 | core | X18-challenger-102-continue30-measure40-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X18-challenger-103-continue30-measure40 | core | X18-challenger-103-continue30-measure40-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X17-challenger-101-continue30-measure40-score-stage | core | X17-challenger-101-continue30-measure40, FV-x-tier2.b-101 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X17-challenger-102-continue30-measure40-score-stage | core | X17-challenger-102-continue30-measure40, FV-x-tier2.b-102 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X17-challenger-103-continue30-measure40-score-stage | core | X17-challenger-103-continue30-measure40, FV-x-tier2.b-103 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X18-challenger-101-continue30-measure40-score-stage | core | X18-challenger-101-continue30-measure40, FV-x-tier2.b-p2-101 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X18-challenger-102-continue30-measure40-score-stage | core | X18-challenger-102-continue30-measure40, FV-x-tier2.b-p2-102 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X18-challenger-103-continue30-measure40-score-stage | core | X18-challenger-103-continue30-measure40, FV-x-tier2.b-p2-103 | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X17-challenger-101-continue30-measure40-score-gpu | core | X17-challenger-101-continue30-measure40-score-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X17-challenger-102-continue30-measure40-score-gpu | core | X17-challenger-102-continue30-measure40-score-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X17-challenger-103-continue30-measure40-score-gpu | core | X17-challenger-103-continue30-measure40-score-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X18-challenger-101-continue30-measure40-score-gpu | core | X18-challenger-101-continue30-measure40-score-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X18-challenger-102-continue30-measure40-score-gpu | core | X18-challenger-102-continue30-measure40-score-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X18-challenger-103-continue30-measure40-score-gpu | core | X18-challenger-103-continue30-measure40-score-stage | afterok | 1 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X17-challenger-101-continue30-measure40-score | core | X17-challenger-101-continue30-measure40-score-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X17-challenger-102-continue30-measure40-score | core | X17-challenger-102-continue30-measure40-score-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X17-challenger-103-continue30-measure40-score | core | X17-challenger-103-continue30-measure40-score-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X18-challenger-101-continue30-measure40-score | core | X18-challenger-101-continue30-measure40-score-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X18-challenger-102-continue30-measure40-score | core | X18-challenger-102-continue30-measure40-score-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X18-challenger-103-continue30-measure40-score | core | X18-challenger-103-continue30-measure40-score-gpu | afterok | 0 | 1 | 8 | yes | CPU-staged plan: len(commands); see lane-commands.json | [0, 0]; CPU/handoff; no GPU compute |
| X17-complete-101-stage | core | X17-control-101-continue10-score, X17-challenger-101-continue30-measure40-score, FV-x-tier2.b-101 | afterok | 0 | 1 | 8 | yes | 39 with both arms READY | [0, 0]; CPU/handoff; no GPU compute |
| X18-complete-101-stage | core | X18-control-101-continue10-score, X18-challenger-101-continue30-measure40-score, FV-x-tier2.b-p2-101 | afterok | 0 | 1 | 8 | yes | 39 with both arms READY | [0, 0]; CPU/handoff; no GPU compute |
| X17-complete-102-stage | core | X17-control-102-continue10-score, X17-challenger-102-continue30-measure40-score, FV-x-tier2.b-102 | afterok | 0 | 1 | 8 | yes | 39 with both arms READY | [0, 0]; CPU/handoff; no GPU compute |
| X18-complete-102-stage | core | X18-control-102-continue10-score, X18-challenger-102-continue30-measure40-score, FV-x-tier2.b-p2-102 | afterok | 0 | 1 | 8 | yes | 39 with both arms READY | [0, 0]; CPU/handoff; no GPU compute |
| X17-complete-103-stage | core | X17-control-103-continue10-score, X17-challenger-103-continue30-measure40-score, FV-x-tier2.b-103 | afterok | 0 | 1 | 8 | yes | 39 with both arms READY | [0, 0]; CPU/handoff; no GPU compute |
| X18-complete-103-stage | core | X18-control-103-continue10-score, X18-challenger-103-continue30-measure40-score, FV-x-tier2.b-p2-103 | afterok | 0 | 1 | 8 | yes | 39 with both arms READY | [0, 0]; CPU/handoff; no GPU compute |
| X17-complete-101-gpu | core | X17-complete-101-stage | afterok | 1 | 1 | 8 | yes | 39 with both arms READY / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X18-complete-101-gpu | core | X18-complete-101-stage | afterok | 1 | 1 | 8 | yes | 39 with both arms READY / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X17-complete-102-gpu | core | X17-complete-102-stage | afterok | 1 | 1 | 8 | yes | 39 with both arms READY / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X18-complete-102-gpu | core | X18-complete-102-stage | afterok | 1 | 1 | 8 | yes | 39 with both arms READY / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X17-complete-103-gpu | core | X17-complete-103-stage | afterok | 1 | 1 | 8 | yes | 39 with both arms READY / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X18-complete-103-gpu | core | X18-complete-103-stage | afterok | 1 | 1 | 8 | yes | 39 with both arms READY / K=3 | None; core measured scoring/preparation/qualification allocation pending |
| X17-complete-101 | core | X17-complete-101-gpu | afterok | 0 | 1 | 8 | yes | 39 with both arms READY | [0, 0]; CPU/handoff; no GPU compute |
| X18-complete-101 | core | X18-complete-101-gpu | afterok | 0 | 1 | 8 | yes | 39 with both arms READY | [0, 0]; CPU/handoff; no GPU compute |
| X17-complete-102 | core | X17-complete-102-gpu | afterok | 0 | 1 | 8 | yes | 39 with both arms READY | [0, 0]; CPU/handoff; no GPU compute |
| X18-complete-102 | core | X18-complete-102-gpu | afterok | 0 | 1 | 8 | yes | 39 with both arms READY | [0, 0]; CPU/handoff; no GPU compute |
| X17-complete-103 | core | X17-complete-103-gpu | afterok | 0 | 1 | 8 | yes | 39 with both arms READY | [0, 0]; CPU/handoff; no GPU compute |
| X18-complete-103 | core | X18-complete-103-gpu | afterok | 0 | 1 | 8 | yes | 39 with both arms READY | [0, 0]; CPU/handoff; no GPU compute |
| X23 | core | X12, X15-complete-101, X15-complete-102, X15-complete-103, X16-complete-101, X16-complete-102, X16-complete-103, X17-complete-101, X17-complete-102, X17-complete-103, X18-complete-101, X18-complete-102, X18-complete-103, X19-complete-101, X19-complete-102, X19-complete-103, X20-complete-101, X20-complete-102, X20-complete-103 | afterany | 0 | 1 | 4 | yes | 3 | [0, 0]; CPU/handoff; no GPU compute |
| M3-init | kcr | none | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-bind-inputs | kcr | M3-init | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-hellaswag | kcr | M3-init | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-bind-base | kcr | M3-init | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-math-resolve | kcr | M3-init | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-assign | kcr | M3-init | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-verify-superset | kcr | M3-bind-inputs | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-finqa-targets | kcr | M3-bind-inputs | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-contract | kcr | M3-bind-inputs, M3-hellaswag | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-finqa-replay | kcr | M3-bind-inputs, M3-hellaswag | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-math-targets | kcr | M3-math-resolve | afterok | 2 | 1 | 8 | yes | 1 | [10.0, 30.0]; K/CR shared generation; compatible committed core targets remove new math generation charge |
| M3-broad-prepare | kcr | M3-contract, M3-finqa-replay | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-pools-shared | kcr | M3-bind-inputs, M3-verify-superset, M3-hellaswag | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-math-ready | kcr | M3-math-resolve, M3-math-targets | afterany | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-broad | kcr | M3-broad-prepare | afterok | 1 | 1 | 8 | yes | 3 / K=3 | [2.0, 25.0]; K/CR shared generation; compatible committed core targets remove new math generation charge |
| M3-schedules | kcr | M3-pools-shared, M3-finqa-targets, M3-bind-inputs | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-schedules-math | kcr | M3-pools-shared, M3-math-ready | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-broad-finish | kcr | M3-broad | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-preflight-C0-inputs | kcr | M3-bind-base, M3-schedules | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-preflight-K-inputs | kcr | M3-bind-base, M3-schedules | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-base-101-ready | kcr | M3-assign, M3-bind-base, M3-schedules, M3-hellaswag | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-base-102-ready | kcr | M3-assign, M3-bind-base, M3-schedules, M3-hellaswag | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-base-103-ready | kcr | M3-assign, M3-bind-base, M3-schedules, M3-hellaswag | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-pools | kcr | M3-broad-finish, M3-finqa-replay | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-preflight-C0 | kcr | M3-preflight-C0-inputs | afterok | 8 | 1 | 32 | no | 1 | [4.8, 9.16]; K/CR one qualification per arm; includes diagnostic/setup reserve |
| M3-preflight-K | kcr | M3-preflight-K-inputs | afterok | 8 | 1 | 32 | no | 1 | [4.94, 9.79]; K/CR one qualification per arm; includes diagnostic/setup reserve |
| M3-base-101 | kcr | M3-base-101-ready | afterok | 1 | 1 | 8 | yes | 6 / K=3 | None; K/CR shared base/validation total 2.10–4.20; per-job measurement pending |
| M3-base-102 | kcr | M3-base-102-ready | afterok | 1 | 1 | 8 | yes | 6 / K=3 | None; K/CR shared base/validation total 2.10–4.20; per-job measurement pending |
| M3-base-103 | kcr | M3-base-103-ready | afterok | 1 | 1 | 8 | yes | 6 / K=3 | None; K/CR shared base/validation total 2.10–4.20; per-job measurement pending |
| M3-schedules-coverage | kcr | M3-pools, M3-schedules, M3-finqa-targets | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-schedules-coverage-math | kcr | M3-pools, M3-schedules-math, M3-math-ready | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-C0-101-finqa-prepare | kcr | M3-base-101, M3-preflight-C0 | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-K-101-finqa-prepare | kcr | M3-base-101, M3-preflight-K | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-C0-102-finqa-prepare | kcr | M3-base-102, M3-preflight-C0 | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-K-102-finqa-prepare | kcr | M3-base-102, M3-preflight-K | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-C0-103-finqa-prepare | kcr | M3-base-103, M3-preflight-C0 | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-K-103-finqa-prepare | kcr | M3-base-103, M3-preflight-K | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-preflight-G-inputs | kcr | M3-bind-base, M3-schedules-coverage | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-preflight-CR-inputs | kcr | M3-bind-base, M3-schedules-coverage | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-C0-101-finqa-train | kcr | M3-C0-101-finqa-prepare | afterok | 8 | 1 | 32 | no | 1 | [15.49, 25.56]; K/CR measured-smoke calibrated stage; setup/scoring/qualification are separate envelope components |
| M3-K-101-finqa-train | kcr | M3-K-101-finqa-prepare | afterok | 8 | 1 | 32 | no | 1 | [16.06, 28.09]; K/CR measured-smoke calibrated stage; setup/scoring/qualification are separate envelope components |
| M3-C0-102-finqa-train | kcr | M3-C0-102-finqa-prepare | afterok | 8 | 1 | 32 | no | 1 | [15.49, 25.56]; K/CR measured-smoke calibrated stage; setup/scoring/qualification are separate envelope components |
| M3-K-102-finqa-train | kcr | M3-K-102-finqa-prepare | afterok | 8 | 1 | 32 | no | 1 | [16.06, 28.09]; K/CR measured-smoke calibrated stage; setup/scoring/qualification are separate envelope components |
| M3-C0-103-finqa-train | kcr | M3-C0-103-finqa-prepare | afterok | 8 | 1 | 32 | no | 1 | [15.49, 25.56]; K/CR measured-smoke calibrated stage; setup/scoring/qualification are separate envelope components |
| M3-K-103-finqa-train | kcr | M3-K-103-finqa-prepare | afterok | 8 | 1 | 32 | no | 1 | [16.06, 28.09]; K/CR measured-smoke calibrated stage; setup/scoring/qualification are separate envelope components |
| M3-preflight-G | kcr | M3-preflight-G-inputs | afterok | 8 | 1 | 32 | no | 1 | [6.69, 12.24]; K/CR one qualification per arm; includes diagnostic/setup reserve |
| M3-preflight-CR | kcr | M3-preflight-CR-inputs | afterok | 8 | 1 | 32 | no | 1 | [6.69, 12.24]; K/CR one qualification per arm; includes diagnostic/setup reserve |
| M3-C0-101-finqa-export | kcr | M3-C0-101-finqa-train | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-K-101-finqa-export | kcr | M3-K-101-finqa-train | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-C0-102-finqa-export | kcr | M3-C0-102-finqa-train | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-K-102-finqa-export | kcr | M3-K-102-finqa-train | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-C0-103-finqa-export | kcr | M3-C0-103-finqa-train | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-K-103-finqa-export | kcr | M3-K-103-finqa-train | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-G-101-finqa-prepare | kcr | M3-base-101, M3-preflight-G, M3-schedules-coverage | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-CR-101-finqa-prepare | kcr | M3-base-101, M3-preflight-CR, M3-schedules-coverage | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-G-102-finqa-prepare | kcr | M3-base-102, M3-preflight-G, M3-schedules-coverage | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-CR-102-finqa-prepare | kcr | M3-base-102, M3-preflight-CR, M3-schedules-coverage | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-G-103-finqa-prepare | kcr | M3-base-103, M3-preflight-G, M3-schedules-coverage | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-CR-103-finqa-prepare | kcr | M3-base-103, M3-preflight-CR, M3-schedules-coverage | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-C0-101-finqa-score-ready | kcr | M3-C0-101-finqa-export | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-K-101-finqa-score-ready | kcr | M3-K-101-finqa-export | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-C0-102-finqa-score-ready | kcr | M3-C0-102-finqa-export | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-K-102-finqa-score-ready | kcr | M3-K-102-finqa-export | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-C0-103-finqa-score-ready | kcr | M3-C0-103-finqa-export | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-K-103-finqa-score-ready | kcr | M3-K-103-finqa-export | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-fv-101-ready | kcr | M3-assign, M3-bind-base, M3-hellaswag, M3-C0-101-finqa-export | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-kv-101-ready | kcr | M3-assign, M3-bind-base, M3-C0-101-finqa-export | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-fv-102-ready | kcr | M3-assign, M3-bind-base, M3-hellaswag, M3-C0-101-finqa-export | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-kv-102-ready | kcr | M3-assign, M3-bind-base, M3-C0-101-finqa-export | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-fv-103-ready | kcr | M3-assign, M3-bind-base, M3-hellaswag, M3-C0-101-finqa-export | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-kv-103-ready | kcr | M3-assign, M3-bind-base, M3-C0-101-finqa-export | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-C0-101-finqa-score | kcr | M3-C0-101-finqa-score-ready | afterok | 1 | 1 | 8 | yes | 15 / K=3 | [0.8, 1.6]; K/CR 16 call-equivalent allowance per arm/seed/stage |
| M3-K-101-finqa-score | kcr | M3-K-101-finqa-score-ready | afterok | 1 | 1 | 8 | yes | 15 / K=3 | [0.8, 1.6]; K/CR 16 call-equivalent allowance per arm/seed/stage |
| M3-G-101-finqa-train | kcr | M3-G-101-finqa-prepare | afterok | 8 | 1 | 32 | no | 1 | [19.37, 31.95]; K/CR measured-smoke calibrated stage; setup/scoring/qualification are separate envelope components |
| M3-CR-101-finqa-train | kcr | M3-CR-101-finqa-prepare | afterok | 8 | 1 | 32 | no | 1 | [19.37, 31.95]; K/CR measured-smoke calibrated stage; setup/scoring/qualification are separate envelope components |
| M3-C0-102-finqa-score | kcr | M3-C0-102-finqa-score-ready | afterok | 1 | 1 | 8 | yes | 15 / K=3 | [0.8, 1.6]; K/CR 16 call-equivalent allowance per arm/seed/stage |
| M3-K-102-finqa-score | kcr | M3-K-102-finqa-score-ready | afterok | 1 | 1 | 8 | yes | 15 / K=3 | [0.8, 1.6]; K/CR 16 call-equivalent allowance per arm/seed/stage |
| M3-G-102-finqa-train | kcr | M3-G-102-finqa-prepare | afterok | 8 | 1 | 32 | no | 1 | [19.37, 31.95]; K/CR measured-smoke calibrated stage; setup/scoring/qualification are separate envelope components |
| M3-CR-102-finqa-train | kcr | M3-CR-102-finqa-prepare | afterok | 8 | 1 | 32 | no | 1 | [19.37, 31.95]; K/CR measured-smoke calibrated stage; setup/scoring/qualification are separate envelope components |
| M3-C0-103-finqa-score | kcr | M3-C0-103-finqa-score-ready | afterok | 1 | 1 | 8 | yes | 15 / K=3 | [0.8, 1.6]; K/CR 16 call-equivalent allowance per arm/seed/stage |
| M3-K-103-finqa-score | kcr | M3-K-103-finqa-score-ready | afterok | 1 | 1 | 8 | yes | 15 / K=3 | [0.8, 1.6]; K/CR 16 call-equivalent allowance per arm/seed/stage |
| M3-G-103-finqa-train | kcr | M3-G-103-finqa-prepare | afterok | 8 | 1 | 32 | no | 1 | [19.37, 31.95]; K/CR measured-smoke calibrated stage; setup/scoring/qualification are separate envelope components |
| M3-CR-103-finqa-train | kcr | M3-CR-103-finqa-prepare | afterok | 8 | 1 | 32 | no | 1 | [19.37, 31.95]; K/CR measured-smoke calibrated stage; setup/scoring/qualification are separate envelope components |
| M3-fv-101 | kcr | M3-fv-101-ready | afterok | 1 | 1 | 8 | yes | 1 / K=3 | None; K/CR shared base/validation total 2.10–4.20; per-job measurement pending |
| M3-kv-101 | kcr | M3-kv-101-ready | afterok | 1 | 1 | 8 | yes | 1 / K=3 | None; K/CR shared base/validation total 2.10–4.20; per-job measurement pending |
| M3-fv-102 | kcr | M3-fv-102-ready | afterok | 1 | 1 | 8 | yes | 1 / K=3 | None; K/CR shared base/validation total 2.10–4.20; per-job measurement pending |
| M3-kv-102 | kcr | M3-kv-102-ready | afterok | 1 | 1 | 8 | yes | 1 / K=3 | None; K/CR shared base/validation total 2.10–4.20; per-job measurement pending |
| M3-fv-103 | kcr | M3-fv-103-ready | afterok | 1 | 1 | 8 | yes | 1 / K=3 | None; K/CR shared base/validation total 2.10–4.20; per-job measurement pending |
| M3-kv-103 | kcr | M3-kv-103-ready | afterok | 1 | 1 | 8 | yes | 1 / K=3 | None; K/CR shared base/validation total 2.10–4.20; per-job measurement pending |
| M3-C0-101-contract_nli-prepare | kcr | M3-C0-101-finqa-score, M3-preflight-C0 | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-K-101-contract_nli-prepare | kcr | M3-K-101-finqa-score, M3-preflight-K | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-G-101-finqa-export | kcr | M3-G-101-finqa-train | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-CR-101-finqa-export | kcr | M3-CR-101-finqa-train | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-C0-102-contract_nli-prepare | kcr | M3-C0-102-finqa-score, M3-preflight-C0 | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-K-102-contract_nli-prepare | kcr | M3-K-102-finqa-score, M3-preflight-K | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-G-102-finqa-export | kcr | M3-G-102-finqa-train | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-CR-102-finqa-export | kcr | M3-CR-102-finqa-train | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-C0-103-contract_nli-prepare | kcr | M3-C0-103-finqa-score, M3-preflight-C0 | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-K-103-contract_nli-prepare | kcr | M3-K-103-finqa-score, M3-preflight-K | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-G-103-finqa-export | kcr | M3-G-103-finqa-train | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-CR-103-finqa-export | kcr | M3-CR-103-finqa-train | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-C0-101-contract_nli-train | kcr | M3-C0-101-contract_nli-prepare | afterok | 8 | 1 | 32 | no | 1 | [15.49, 25.56]; K/CR measured-smoke calibrated stage; setup/scoring/qualification are separate envelope components |
| M3-K-101-contract_nli-train | kcr | M3-K-101-contract_nli-prepare | afterok | 8 | 1 | 32 | no | 1 | [16.06, 28.09]; K/CR measured-smoke calibrated stage; setup/scoring/qualification are separate envelope components |
| M3-C0-102-contract_nli-train | kcr | M3-C0-102-contract_nli-prepare | afterok | 8 | 1 | 32 | no | 1 | [15.49, 25.56]; K/CR measured-smoke calibrated stage; setup/scoring/qualification are separate envelope components |
| M3-K-102-contract_nli-train | kcr | M3-K-102-contract_nli-prepare | afterok | 8 | 1 | 32 | no | 1 | [16.06, 28.09]; K/CR measured-smoke calibrated stage; setup/scoring/qualification are separate envelope components |
| M3-C0-103-contract_nli-train | kcr | M3-C0-103-contract_nli-prepare | afterok | 8 | 1 | 32 | no | 1 | [15.49, 25.56]; K/CR measured-smoke calibrated stage; setup/scoring/qualification are separate envelope components |
| M3-K-103-contract_nli-train | kcr | M3-K-103-contract_nli-prepare | afterok | 8 | 1 | 32 | no | 1 | [16.06, 28.09]; K/CR measured-smoke calibrated stage; setup/scoring/qualification are separate envelope components |
| M3-G-101-finqa-score-ready | kcr | M3-G-101-finqa-export | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-CR-101-finqa-score-ready | kcr | M3-CR-101-finqa-export | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-G-102-finqa-score-ready | kcr | M3-G-102-finqa-export | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-CR-102-finqa-score-ready | kcr | M3-CR-102-finqa-export | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-G-103-finqa-score-ready | kcr | M3-G-103-finqa-export | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-CR-103-finqa-score-ready | kcr | M3-CR-103-finqa-export | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-C0-101-contract_nli-export | kcr | M3-C0-101-contract_nli-train | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-K-101-contract_nli-export | kcr | M3-K-101-contract_nli-train | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-G-101-finqa-score | kcr | M3-G-101-finqa-score-ready | afterok | 1 | 1 | 8 | yes | 15 / K=3 | [0.8, 1.6]; K/CR 16 call-equivalent allowance per arm/seed/stage |
| M3-CR-101-finqa-score | kcr | M3-CR-101-finqa-score-ready | afterok | 1 | 1 | 8 | yes | 15 / K=3 | [0.8, 1.6]; K/CR 16 call-equivalent allowance per arm/seed/stage |
| M3-C0-102-contract_nli-export | kcr | M3-C0-102-contract_nli-train | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-K-102-contract_nli-export | kcr | M3-K-102-contract_nli-train | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-G-102-finqa-score | kcr | M3-G-102-finqa-score-ready | afterok | 1 | 1 | 8 | yes | 15 / K=3 | [0.8, 1.6]; K/CR 16 call-equivalent allowance per arm/seed/stage |
| M3-CR-102-finqa-score | kcr | M3-CR-102-finqa-score-ready | afterok | 1 | 1 | 8 | yes | 15 / K=3 | [0.8, 1.6]; K/CR 16 call-equivalent allowance per arm/seed/stage |
| M3-C0-103-contract_nli-export | kcr | M3-C0-103-contract_nli-train | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-K-103-contract_nli-export | kcr | M3-K-103-contract_nli-train | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-G-103-finqa-score | kcr | M3-G-103-finqa-score-ready | afterok | 1 | 1 | 8 | yes | 15 / K=3 | [0.8, 1.6]; K/CR 16 call-equivalent allowance per arm/seed/stage |
| M3-CR-103-finqa-score | kcr | M3-CR-103-finqa-score-ready | afterok | 1 | 1 | 8 | yes | 15 / K=3 | [0.8, 1.6]; K/CR 16 call-equivalent allowance per arm/seed/stage |
| M3-G-101-contract_nli-prepare | kcr | M3-G-101-finqa-score, M3-preflight-G, M3-schedules-coverage | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-CR-101-contract_nli-prepare | kcr | M3-CR-101-finqa-score, M3-preflight-CR, M3-schedules-coverage | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-G-102-contract_nli-prepare | kcr | M3-G-102-finqa-score, M3-preflight-G, M3-schedules-coverage | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-CR-102-contract_nli-prepare | kcr | M3-CR-102-finqa-score, M3-preflight-CR, M3-schedules-coverage | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-G-103-contract_nli-prepare | kcr | M3-G-103-finqa-score, M3-preflight-G, M3-schedules-coverage | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-CR-103-contract_nli-prepare | kcr | M3-CR-103-finqa-score, M3-preflight-CR, M3-schedules-coverage | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-C0-101-contract_nli-score-ready | kcr | M3-C0-101-contract_nli-export | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-K-101-contract_nli-score-ready | kcr | M3-K-101-contract_nli-export | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-C0-102-contract_nli-score-ready | kcr | M3-C0-102-contract_nli-export | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-K-102-contract_nli-score-ready | kcr | M3-K-102-contract_nli-export | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-C0-103-contract_nli-score-ready | kcr | M3-C0-103-contract_nli-export | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-K-103-contract_nli-score-ready | kcr | M3-K-103-contract_nli-export | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-C0-101-contract_nli-score | kcr | M3-C0-101-contract_nli-score-ready | afterok | 1 | 1 | 8 | yes | 16 / K=3 | [0.8, 1.6]; K/CR 16 call-equivalent allowance per arm/seed/stage |
| M3-K-101-contract_nli-score | kcr | M3-K-101-contract_nli-score-ready | afterok | 1 | 1 | 8 | yes | 16 / K=3 | [0.8, 1.6]; K/CR 16 call-equivalent allowance per arm/seed/stage |
| M3-G-101-contract_nli-train | kcr | M3-G-101-contract_nli-prepare | afterok | 8 | 1 | 32 | no | 1 | [23.24, 38.34]; K/CR measured-smoke calibrated stage; setup/scoring/qualification are separate envelope components |
| M3-CR-101-contract_nli-train | kcr | M3-CR-101-contract_nli-prepare | afterok | 8 | 1 | 32 | no | 1 | [23.24, 38.34]; K/CR measured-smoke calibrated stage; setup/scoring/qualification are separate envelope components |
| M3-C0-102-contract_nli-score | kcr | M3-C0-102-contract_nli-score-ready | afterok | 1 | 1 | 8 | yes | 16 / K=3 | [0.8, 1.6]; K/CR 16 call-equivalent allowance per arm/seed/stage |
| M3-K-102-contract_nli-score | kcr | M3-K-102-contract_nli-score-ready | afterok | 1 | 1 | 8 | yes | 16 / K=3 | [0.8, 1.6]; K/CR 16 call-equivalent allowance per arm/seed/stage |
| M3-G-102-contract_nli-train | kcr | M3-G-102-contract_nli-prepare | afterok | 8 | 1 | 32 | no | 1 | [23.24, 38.34]; K/CR measured-smoke calibrated stage; setup/scoring/qualification are separate envelope components |
| M3-CR-102-contract_nli-train | kcr | M3-CR-102-contract_nli-prepare | afterok | 8 | 1 | 32 | no | 1 | [23.24, 38.34]; K/CR measured-smoke calibrated stage; setup/scoring/qualification are separate envelope components |
| M3-C0-103-contract_nli-score | kcr | M3-C0-103-contract_nli-score-ready | afterok | 1 | 1 | 8 | yes | 16 / K=3 | [0.8, 1.6]; K/CR 16 call-equivalent allowance per arm/seed/stage |
| M3-K-103-contract_nli-score | kcr | M3-K-103-contract_nli-score-ready | afterok | 1 | 1 | 8 | yes | 16 / K=3 | [0.8, 1.6]; K/CR 16 call-equivalent allowance per arm/seed/stage |
| M3-G-103-contract_nli-train | kcr | M3-G-103-contract_nli-prepare | afterok | 8 | 1 | 32 | no | 1 | [23.24, 38.34]; K/CR measured-smoke calibrated stage; setup/scoring/qualification are separate envelope components |
| M3-CR-103-contract_nli-train | kcr | M3-CR-103-contract_nli-prepare | afterok | 8 | 1 | 32 | no | 1 | [23.24, 38.34]; K/CR measured-smoke calibrated stage; setup/scoring/qualification are separate envelope components |
| M3-C0-101-math_hard-prepare | kcr | M3-C0-101-contract_nli-score, M3-preflight-C0, M3-schedules-math | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-K-101-math_hard-prepare | kcr | M3-K-101-contract_nli-score, M3-preflight-K, M3-schedules-math | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-G-101-contract_nli-export | kcr | M3-G-101-contract_nli-train | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-CR-101-contract_nli-export | kcr | M3-CR-101-contract_nli-train | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-C0-102-math_hard-prepare | kcr | M3-C0-102-contract_nli-score, M3-preflight-C0, M3-schedules-math | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-K-102-math_hard-prepare | kcr | M3-K-102-contract_nli-score, M3-preflight-K, M3-schedules-math | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-G-102-contract_nli-export | kcr | M3-G-102-contract_nli-train | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-CR-102-contract_nli-export | kcr | M3-CR-102-contract_nli-train | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-C0-103-math_hard-prepare | kcr | M3-C0-103-contract_nli-score, M3-preflight-C0, M3-schedules-math | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-K-103-math_hard-prepare | kcr | M3-K-103-contract_nli-score, M3-preflight-K, M3-schedules-math | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-G-103-contract_nli-export | kcr | M3-G-103-contract_nli-train | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-CR-103-contract_nli-export | kcr | M3-CR-103-contract_nli-train | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-C0-101-math_hard-train | kcr | M3-C0-101-math_hard-prepare | afterok | 8 | 1 | 32 | no | 1 | [15.49, 25.56]; K/CR measured-smoke calibrated stage; setup/scoring/qualification are separate envelope components |
| M3-K-101-math_hard-train | kcr | M3-K-101-math_hard-prepare | afterok | 8 | 1 | 32 | no | 1 | [16.06, 28.09]; K/CR measured-smoke calibrated stage; setup/scoring/qualification are separate envelope components |
| M3-C0-102-math_hard-train | kcr | M3-C0-102-math_hard-prepare | afterok | 8 | 1 | 32 | no | 1 | [15.49, 25.56]; K/CR measured-smoke calibrated stage; setup/scoring/qualification are separate envelope components |
| M3-K-102-math_hard-train | kcr | M3-K-102-math_hard-prepare | afterok | 8 | 1 | 32 | no | 1 | [16.06, 28.09]; K/CR measured-smoke calibrated stage; setup/scoring/qualification are separate envelope components |
| M3-C0-103-math_hard-train | kcr | M3-C0-103-math_hard-prepare | afterok | 8 | 1 | 32 | no | 1 | [15.49, 25.56]; K/CR measured-smoke calibrated stage; setup/scoring/qualification are separate envelope components |
| M3-K-103-math_hard-train | kcr | M3-K-103-math_hard-prepare | afterok | 8 | 1 | 32 | no | 1 | [16.06, 28.09]; K/CR measured-smoke calibrated stage; setup/scoring/qualification are separate envelope components |
| M3-G-101-contract_nli-score-ready | kcr | M3-G-101-contract_nli-export | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-CR-101-contract_nli-score-ready | kcr | M3-CR-101-contract_nli-export | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-G-102-contract_nli-score-ready | kcr | M3-G-102-contract_nli-export | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-CR-102-contract_nli-score-ready | kcr | M3-CR-102-contract_nli-export | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-G-103-contract_nli-score-ready | kcr | M3-G-103-contract_nli-export | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-CR-103-contract_nli-score-ready | kcr | M3-CR-103-contract_nli-export | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-C0-101-math_hard-export | kcr | M3-C0-101-math_hard-train | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-K-101-math_hard-export | kcr | M3-K-101-math_hard-train | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-G-101-contract_nli-score | kcr | M3-G-101-contract_nli-score-ready | afterok | 1 | 1 | 8 | yes | 16 / K=3 | [0.8, 1.6]; K/CR 16 call-equivalent allowance per arm/seed/stage |
| M3-CR-101-contract_nli-score | kcr | M3-CR-101-contract_nli-score-ready | afterok | 1 | 1 | 8 | yes | 16 / K=3 | [0.8, 1.6]; K/CR 16 call-equivalent allowance per arm/seed/stage |
| M3-C0-102-math_hard-export | kcr | M3-C0-102-math_hard-train | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-K-102-math_hard-export | kcr | M3-K-102-math_hard-train | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-G-102-contract_nli-score | kcr | M3-G-102-contract_nli-score-ready | afterok | 1 | 1 | 8 | yes | 16 / K=3 | [0.8, 1.6]; K/CR 16 call-equivalent allowance per arm/seed/stage |
| M3-CR-102-contract_nli-score | kcr | M3-CR-102-contract_nli-score-ready | afterok | 1 | 1 | 8 | yes | 16 / K=3 | [0.8, 1.6]; K/CR 16 call-equivalent allowance per arm/seed/stage |
| M3-C0-103-math_hard-export | kcr | M3-C0-103-math_hard-train | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-K-103-math_hard-export | kcr | M3-K-103-math_hard-train | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-G-103-contract_nli-score | kcr | M3-G-103-contract_nli-score-ready | afterok | 1 | 1 | 8 | yes | 16 / K=3 | [0.8, 1.6]; K/CR 16 call-equivalent allowance per arm/seed/stage |
| M3-CR-103-contract_nli-score | kcr | M3-CR-103-contract_nli-score-ready | afterok | 1 | 1 | 8 | yes | 16 / K=3 | [0.8, 1.6]; K/CR 16 call-equivalent allowance per arm/seed/stage |
| M3-G-101-math_hard-prepare | kcr | M3-G-101-contract_nli-score, M3-preflight-G, M3-schedules-coverage-math | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-CR-101-math_hard-prepare | kcr | M3-CR-101-contract_nli-score, M3-preflight-CR, M3-schedules-coverage-math | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-G-102-math_hard-prepare | kcr | M3-G-102-contract_nli-score, M3-preflight-G, M3-schedules-coverage-math | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-CR-102-math_hard-prepare | kcr | M3-CR-102-contract_nli-score, M3-preflight-CR, M3-schedules-coverage-math | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-G-103-math_hard-prepare | kcr | M3-G-103-contract_nli-score, M3-preflight-G, M3-schedules-coverage-math | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-CR-103-math_hard-prepare | kcr | M3-CR-103-contract_nli-score, M3-preflight-CR, M3-schedules-coverage-math | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-C0-101-math_hard-score-ready | kcr | M3-C0-101-math_hard-export | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-K-101-math_hard-score-ready | kcr | M3-K-101-math_hard-export | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-C0-102-math_hard-score-ready | kcr | M3-C0-102-math_hard-export | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-K-102-math_hard-score-ready | kcr | M3-K-102-math_hard-export | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-C0-103-math_hard-score-ready | kcr | M3-C0-103-math_hard-export | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-K-103-math_hard-score-ready | kcr | M3-K-103-math_hard-export | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-C0-101-math_hard-score | kcr | M3-C0-101-math_hard-score-ready | afterok | 1 | 1 | 8 | yes | 17 / K=3 | [0.8, 1.6]; K/CR 16 call-equivalent allowance per arm/seed/stage |
| M3-K-101-math_hard-score | kcr | M3-K-101-math_hard-score-ready | afterok | 1 | 1 | 8 | yes | 17 / K=3 | [0.8, 1.6]; K/CR 16 call-equivalent allowance per arm/seed/stage |
| M3-G-101-math_hard-train | kcr | M3-G-101-math_hard-prepare | afterok | 8 | 1 | 32 | no | 1 | [27.11, 44.73]; K/CR measured-smoke calibrated stage; setup/scoring/qualification are separate envelope components |
| M3-CR-101-math_hard-train | kcr | M3-CR-101-math_hard-prepare | afterok | 8 | 1 | 32 | no | 1 | [27.11, 44.73]; K/CR measured-smoke calibrated stage; setup/scoring/qualification are separate envelope components |
| M3-C0-102-math_hard-score | kcr | M3-C0-102-math_hard-score-ready | afterok | 1 | 1 | 8 | yes | 17 / K=3 | [0.8, 1.6]; K/CR 16 call-equivalent allowance per arm/seed/stage |
| M3-K-102-math_hard-score | kcr | M3-K-102-math_hard-score-ready | afterok | 1 | 1 | 8 | yes | 17 / K=3 | [0.8, 1.6]; K/CR 16 call-equivalent allowance per arm/seed/stage |
| M3-G-102-math_hard-train | kcr | M3-G-102-math_hard-prepare | afterok | 8 | 1 | 32 | no | 1 | [27.11, 44.73]; K/CR measured-smoke calibrated stage; setup/scoring/qualification are separate envelope components |
| M3-CR-102-math_hard-train | kcr | M3-CR-102-math_hard-prepare | afterok | 8 | 1 | 32 | no | 1 | [27.11, 44.73]; K/CR measured-smoke calibrated stage; setup/scoring/qualification are separate envelope components |
| M3-C0-103-math_hard-score | kcr | M3-C0-103-math_hard-score-ready | afterok | 1 | 1 | 8 | yes | 17 / K=3 | [0.8, 1.6]; K/CR 16 call-equivalent allowance per arm/seed/stage |
| M3-K-103-math_hard-score | kcr | M3-K-103-math_hard-score-ready | afterok | 1 | 1 | 8 | yes | 17 / K=3 | [0.8, 1.6]; K/CR 16 call-equivalent allowance per arm/seed/stage |
| M3-G-103-math_hard-train | kcr | M3-G-103-math_hard-prepare | afterok | 8 | 1 | 32 | no | 1 | [27.11, 44.73]; K/CR measured-smoke calibrated stage; setup/scoring/qualification are separate envelope components |
| M3-CR-103-math_hard-train | kcr | M3-CR-103-math_hard-prepare | afterok | 8 | 1 | 32 | no | 1 | [27.11, 44.73]; K/CR measured-smoke calibrated stage; setup/scoring/qualification are separate envelope components |
| M3-G-101-math_hard-export | kcr | M3-G-101-math_hard-train | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-CR-101-math_hard-export | kcr | M3-CR-101-math_hard-train | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-G-102-math_hard-export | kcr | M3-G-102-math_hard-train | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-CR-102-math_hard-export | kcr | M3-CR-102-math_hard-train | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-G-103-math_hard-export | kcr | M3-G-103-math_hard-train | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-CR-103-math_hard-export | kcr | M3-CR-103-math_hard-train | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-G-101-math_hard-score-ready | kcr | M3-G-101-math_hard-export | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-CR-101-math_hard-score-ready | kcr | M3-CR-101-math_hard-export | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-G-102-math_hard-score-ready | kcr | M3-G-102-math_hard-export | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-CR-102-math_hard-score-ready | kcr | M3-CR-102-math_hard-export | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-G-103-math_hard-score-ready | kcr | M3-G-103-math_hard-export | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-CR-103-math_hard-score-ready | kcr | M3-CR-103-math_hard-export | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-rescore-101-ready | kcr | M3-assign, M3-bind-base, M3-hellaswag, M3-C0-101-finqa-export, M3-C0-101-contract_nli-export, M3-C0-101-math_hard-export, M3-K-101-finqa-export, M3-K-101-contract_nli-export, M3-K-101-math_hard-export, M3-G-101-finqa-export, M3-G-101-contract_nli-export, M3-G-101-math_hard-export, M3-CR-101-finqa-export, M3-CR-101-contract_nli-export, M3-CR-101-math_hard-export | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-rescore-102-ready | kcr | M3-assign, M3-bind-base, M3-hellaswag, M3-C0-102-finqa-export, M3-C0-102-contract_nli-export, M3-C0-102-math_hard-export, M3-K-102-finqa-export, M3-K-102-contract_nli-export, M3-K-102-math_hard-export, M3-G-102-finqa-export, M3-G-102-contract_nli-export, M3-G-102-math_hard-export, M3-CR-102-finqa-export, M3-CR-102-contract_nli-export, M3-CR-102-math_hard-export | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-rescore-103-ready | kcr | M3-assign, M3-bind-base, M3-hellaswag, M3-C0-103-finqa-export, M3-C0-103-contract_nli-export, M3-C0-103-math_hard-export, M3-K-103-finqa-export, M3-K-103-contract_nli-export, M3-K-103-math_hard-export, M3-G-103-finqa-export, M3-G-103-contract_nli-export, M3-G-103-math_hard-export, M3-CR-103-finqa-export, M3-CR-103-contract_nli-export, M3-CR-103-math_hard-export | afterok | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| M3-G-101-math_hard-score | kcr | M3-G-101-math_hard-score-ready | afterok | 1 | 1 | 8 | yes | 17 / K=3 | [0.8, 1.6]; K/CR 16 call-equivalent allowance per arm/seed/stage |
| M3-CR-101-math_hard-score | kcr | M3-CR-101-math_hard-score-ready | afterok | 1 | 1 | 8 | yes | 17 / K=3 | [0.8, 1.6]; K/CR 16 call-equivalent allowance per arm/seed/stage |
| M3-G-102-math_hard-score | kcr | M3-G-102-math_hard-score-ready | afterok | 1 | 1 | 8 | yes | 17 / K=3 | [0.8, 1.6]; K/CR 16 call-equivalent allowance per arm/seed/stage |
| M3-CR-102-math_hard-score | kcr | M3-CR-102-math_hard-score-ready | afterok | 1 | 1 | 8 | yes | 17 / K=3 | [0.8, 1.6]; K/CR 16 call-equivalent allowance per arm/seed/stage |
| M3-G-103-math_hard-score | kcr | M3-G-103-math_hard-score-ready | afterok | 1 | 1 | 8 | yes | 17 / K=3 | [0.8, 1.6]; K/CR 16 call-equivalent allowance per arm/seed/stage |
| M3-CR-103-math_hard-score | kcr | M3-CR-103-math_hard-score-ready | afterok | 1 | 1 | 8 | yes | 17 / K=3 | [0.8, 1.6]; K/CR 16 call-equivalent allowance per arm/seed/stage |
| M3-rescore-101 | kcr | M3-rescore-101-ready | afterok | 1 | 1 | 8 | yes | 198 / K=3 | None; K/CR complete pinned generated-panel seed lane; measured cost pending |
| M3-rescore-102 | kcr | M3-rescore-102-ready | afterok | 1 | 1 | 8 | yes | 198 / K=3 | None; K/CR complete pinned generated-panel seed lane; measured cost pending |
| M3-rescore-103 | kcr | M3-rescore-103-ready | afterok | 1 | 1 | 8 | yes | 198 / K=3 | None; K/CR complete pinned generated-panel seed lane; measured cost pending |
| M3-read | kcr | M3-pools, M3-C0-101-math_hard-score, M3-K-101-math_hard-score, M3-G-101-math_hard-score, M3-CR-101-math_hard-score, M3-C0-102-math_hard-score, M3-K-102-math_hard-score, M3-G-102-math_hard-score, M3-CR-102-math_hard-score, M3-C0-103-math_hard-score, M3-K-103-math_hard-score, M3-G-103-math_hard-score, M3-CR-103-math_hard-score, M3-fv-101, M3-kv-101, M3-fv-102, M3-kv-102, M3-fv-103, M3-kv-103, M3-rescore-101, M3-rescore-102, M3-rescore-103 | afterany | 0 | 1 | 8 | yes | 1 | [0, 0]; K/CR CPU/handoff |
| SP-verify-pools | sp | M-pools, M3-pools, R-broad, R-nominate | afterany | 0 | 1 | 8 | yes | 1 | [0, 0]; SP(b) .25–1 GPU-h/entry; startup separate; unresolved coverage has no finite upper total |
| SP-prepare-core-s101 | sp | M-final-score-101, M-final-score-101-gpu, M-final-score-101-stage, M10-101-incoming, M10-101-incoming-gpu, M10-101-incoming-stage, M10-101-score, M10-101-score-gpu, M10-101-score-stage, M4-101, M4-101-gpu, M4-101-stage, M6-101-incoming, M6-101-incoming-gpu, M6-101-incoming-stage, M6-101-score, M6-101-score-gpu, M6-101-score-stage, M7-101-incoming, M7-101-incoming-gpu, M7-101-incoming-stage, M7-101-score, M7-101-score-gpu, M7-101-score-stage, M8-101-incoming, M8-101-incoming-gpu, M8-101-incoming-stage, M8-101-score, M8-101-score-gpu, M8-101-score-stage, M8-compare-101, M8-compare-101-gpu, M8-compare-101-stage, M9-101-incoming, M9-101-incoming-gpu, M9-101-incoming-stage, M9-101-score, M9-101-score-gpu, M9-101-score-stage, M9-compare-101, M9-compare-101-gpu, M9-compare-101-stage, SP-verify-pools | afterany | 0 | 1 | 8 | yes | 36 | [0, 0]; SP(b) .25–1 GPU-h/entry; startup separate; unresolved coverage has no finite upper total |
| SP-panel-core-s101 | sp | SP-prepare-core-s101 | afterany | 1 | 1 | 8 | yes | 18 / K=3 | [4.5, 18]; SP(b) .25–1 GPU-h/entry; startup separate; unresolved coverage has no finite upper total |
| SP-prepare-core-s102 | sp | M-final-score-102, M-final-score-102-gpu, M-final-score-102-stage, M10-102-incoming, M10-102-incoming-gpu, M10-102-incoming-stage, M10-102-score, M10-102-score-gpu, M10-102-score-stage, M4-102, M4-102-gpu, M4-102-stage, M6-102-incoming, M6-102-incoming-gpu, M6-102-incoming-stage, M6-102-score, M6-102-score-gpu, M6-102-score-stage, M7-102-incoming, M7-102-incoming-gpu, M7-102-incoming-stage, M7-102-score, M7-102-score-gpu, M7-102-score-stage, M8-102-incoming, M8-102-incoming-gpu, M8-102-incoming-stage, M8-102-score, M8-102-score-gpu, M8-102-score-stage, M8-compare-102, M8-compare-102-gpu, M8-compare-102-stage, M9-102-incoming, M9-102-incoming-gpu, M9-102-incoming-stage, M9-102-score, M9-102-score-gpu, M9-102-score-stage, M9-compare-102, M9-compare-102-gpu, M9-compare-102-stage, SP-verify-pools | afterany | 0 | 1 | 8 | yes | 36 | [0, 0]; SP(b) .25–1 GPU-h/entry; startup separate; unresolved coverage has no finite upper total |
| SP-panel-core-s102 | sp | SP-prepare-core-s102 | afterany | 1 | 1 | 8 | yes | 18 / K=3 | [4.5, 18]; SP(b) .25–1 GPU-h/entry; startup separate; unresolved coverage has no finite upper total |
| SP-prepare-core-s103 | sp | M-final-score-103, M-final-score-103-gpu, M-final-score-103-stage, M10-103-incoming, M10-103-incoming-gpu, M10-103-incoming-stage, M10-103-score, M10-103-score-gpu, M10-103-score-stage, M4-103, M4-103-gpu, M4-103-stage, M6-103-incoming, M6-103-incoming-gpu, M6-103-incoming-stage, M6-103-score, M6-103-score-gpu, M6-103-score-stage, M7-103-incoming, M7-103-incoming-gpu, M7-103-incoming-stage, M7-103-score, M7-103-score-gpu, M7-103-score-stage, M8-103-incoming, M8-103-incoming-gpu, M8-103-incoming-stage, M8-103-score, M8-103-score-gpu, M8-103-score-stage, M8-compare-103, M8-compare-103-gpu, M8-compare-103-stage, M9-103-incoming, M9-103-incoming-gpu, M9-103-incoming-stage, M9-103-score, M9-103-score-gpu, M9-103-score-stage, M9-compare-103, M9-compare-103-gpu, M9-compare-103-stage, SP-verify-pools | afterany | 0 | 1 | 8 | yes | 36 | [0, 0]; SP(b) .25–1 GPU-h/entry; startup separate; unresolved coverage has no finite upper total |
| SP-panel-core-s103 | sp | SP-prepare-core-s103 | afterany | 1 | 1 | 8 | yes | 18 / K=3 | [4.5, 18]; SP(b) .25–1 GPU-h/entry; startup separate; unresolved coverage has no finite upper total |
| SP-prepare-m3-s101 | sp | M3-C0-101-contract_nli-score, M3-C0-101-finqa-score, M3-C0-101-math_hard-score, M3-CR-101-contract_nli-score, M3-CR-101-finqa-score, M3-CR-101-math_hard-score, M3-G-101-contract_nli-score, M3-G-101-finqa-score, M3-G-101-math_hard-score, M3-K-101-contract_nli-score, M3-K-101-finqa-score, M3-K-101-math_hard-score, M3-base-101, M3-rescore-101, SP-verify-pools | afterany | 0 | 1 | 8 | yes | 26 | [0, 0]; SP(b) .25–1 GPU-h/entry; startup separate; unresolved coverage has no finite upper total |
| SP-panel-m3-s101 | sp | SP-prepare-m3-s101 | afterany | 1 | 1 | 8 | yes | 13 / K=3 | [3.25, 13]; SP(b) .25–1 GPU-h/entry; startup separate; unresolved coverage has no finite upper total |
| SP-prepare-m3-s102 | sp | M3-C0-102-contract_nli-score, M3-C0-102-finqa-score, M3-C0-102-math_hard-score, M3-CR-102-contract_nli-score, M3-CR-102-finqa-score, M3-CR-102-math_hard-score, M3-G-102-contract_nli-score, M3-G-102-finqa-score, M3-G-102-math_hard-score, M3-K-102-contract_nli-score, M3-K-102-finqa-score, M3-K-102-math_hard-score, M3-base-102, M3-rescore-102, SP-verify-pools | afterany | 0 | 1 | 8 | yes | 26 | [0, 0]; SP(b) .25–1 GPU-h/entry; startup separate; unresolved coverage has no finite upper total |
| SP-panel-m3-s102 | sp | SP-prepare-m3-s102 | afterany | 1 | 1 | 8 | yes | 13 / K=3 | [3.25, 13]; SP(b) .25–1 GPU-h/entry; startup separate; unresolved coverage has no finite upper total |
| SP-prepare-m3-s103 | sp | M3-C0-103-contract_nli-score, M3-C0-103-finqa-score, M3-C0-103-math_hard-score, M3-CR-103-contract_nli-score, M3-CR-103-finqa-score, M3-CR-103-math_hard-score, M3-G-103-contract_nli-score, M3-G-103-finqa-score, M3-G-103-math_hard-score, M3-K-103-contract_nli-score, M3-K-103-finqa-score, M3-K-103-math_hard-score, M3-base-103, M3-rescore-103, SP-verify-pools | afterany | 0 | 1 | 8 | yes | 26 | [0, 0]; SP(b) .25–1 GPU-h/entry; startup separate; unresolved coverage has no finite upper total |
| SP-panel-m3-s103 | sp | SP-prepare-m3-s103 | afterany | 1 | 1 | 8 | yes | 13 / K=3 | [3.25, 13]; SP(b) .25–1 GPU-h/entry; startup separate; unresolved coverage has no finite upper total |
| SP-prepare-reprobe-s101 | sp | R-both-101-score, R-both-101-score-gpu, R-both-101-score-stage, R-broad-101-score, R-broad-101-score-gpu, R-broad-101-score-stage, R-compare-broad-101, R-compare-broad-101-gpu, R-compare-broad-101-stage, R-compare-full-101, R-compare-full-101-gpu, R-compare-full-101-stage, R-earlier-101-score, R-earlier-101-score-gpu, R-earlier-101-score-stage, R-generic-101-score, R-generic-101-score-gpu, R-generic-101-score-stage, R-incoming-101, R-incoming-101-gpu, R-incoming-101-stage, SP-verify-pools | afterany | 0 | 1 | 8 | yes | 18 | [0, 0]; SP(b) .25–1 GPU-h/entry; startup separate; unresolved coverage has no finite upper total |
| SP-panel-reprobe-s101 | sp | SP-prepare-reprobe-s101 | afterany | 1 | 1 | 8 | yes | 9 / K=3 | [2.25, 9]; SP(b) .25–1 GPU-h/entry; startup separate; unresolved coverage has no finite upper total |
| SP-prepare-reprobe-s102 | sp | R-both-102-score, R-both-102-score-gpu, R-both-102-score-stage, R-broad-102-score, R-broad-102-score-gpu, R-broad-102-score-stage, R-compare-broad-102, R-compare-broad-102-gpu, R-compare-broad-102-stage, R-compare-full-102, R-compare-full-102-gpu, R-compare-full-102-stage, R-earlier-102-score, R-earlier-102-score-gpu, R-earlier-102-score-stage, R-generic-102-score, R-generic-102-score-gpu, R-generic-102-score-stage, R-incoming-102, R-incoming-102-gpu, R-incoming-102-stage, SP-verify-pools | afterany | 0 | 1 | 8 | yes | 18 | [0, 0]; SP(b) .25–1 GPU-h/entry; startup separate; unresolved coverage has no finite upper total |
| SP-panel-reprobe-s102 | sp | SP-prepare-reprobe-s102 | afterany | 1 | 1 | 8 | yes | 9 / K=3 | [2.25, 9]; SP(b) .25–1 GPU-h/entry; startup separate; unresolved coverage has no finite upper total |
| SP-prepare-reprobe-s103 | sp | R-both-103-score, R-both-103-score-gpu, R-both-103-score-stage, R-broad-103-score, R-broad-103-score-gpu, R-broad-103-score-stage, R-compare-broad-103, R-compare-broad-103-gpu, R-compare-broad-103-stage, R-compare-full-103, R-compare-full-103-gpu, R-compare-full-103-stage, R-earlier-103-score, R-earlier-103-score-gpu, R-earlier-103-score-stage, R-generic-103-score, R-generic-103-score-gpu, R-generic-103-score-stage, R-incoming-103, R-incoming-103-gpu, R-incoming-103-stage, SP-verify-pools | afterany | 0 | 1 | 8 | yes | 18 | [0, 0]; SP(b) .25–1 GPU-h/entry; startup separate; unresolved coverage has no finite upper total |
| SP-panel-reprobe-s103 | sp | SP-prepare-reprobe-s103 | afterany | 1 | 1 | 8 | yes | 9 / K=3 | [2.25, 9]; SP(b) .25–1 GPU-h/entry; startup separate; unresolved coverage has no finite upper total |
| SP-prepare-return1.A101-s101 | sp | RA101, RA101-gpu, RA101-stage, SP-verify-pools | afterany | 0 | 1 | 8 | yes | 14 | [0, 0]; SP(b) .25–1 GPU-h/entry; startup separate; unresolved coverage has no finite upper total |
| SP-panel-return1.A101-s101 | sp | SP-prepare-return1.A101-s101 | afterany | 1 | 1 | 8 | yes | 7 / K=3 | [1.75, 7]; SP(b) .25–1 GPU-h/entry; startup separate; unresolved coverage has no finite upper total |
| SP-prepare-return1.A102-s102 | sp | RA102, RA102-gpu, RA102-stage, SP-verify-pools | afterany | 0 | 1 | 8 | yes | 12 | [0, 0]; SP(b) .25–1 GPU-h/entry; startup separate; unresolved coverage has no finite upper total |
| SP-panel-return1.A102-s102 | sp | SP-prepare-return1.A102-s102 | afterany | 1 | 1 | 8 | yes | 6 / K=3 | [1.5, 6]; SP(b) .25–1 GPU-h/entry; startup separate; unresolved coverage has no finite upper total |
| SP-prepare-return1.A103-s103 | sp | RA103, RA103-gpu, RA103-stage, SP-verify-pools | afterany | 0 | 1 | 8 | yes | 20 | [0, 0]; SP(b) .25–1 GPU-h/entry; startup separate; unresolved coverage has no finite upper total |
| SP-panel-return1.A103-s103 | sp | SP-prepare-return1.A103-s103 | afterany | 1 | 1 | 8 | yes | 10 / K=3 | [2.5, 10]; SP(b) .25–1 GPU-h/entry; startup separate; unresolved coverage has no finite upper total |
| SP-prepare-x-tier1-s101 | sp | SP-verify-pools, X-complete-101, X-complete-101-gpu, X-complete-101-stage, X-incoming-101, X-incoming-101-gpu, X-incoming-101-stage, X5-control-101-train-score, X5-control-101-train-score-gpu, X5-control-101-train-score-stage, X6-p-101-train-score, X6-p-101-train-score-gpu, X6-p-101-train-score-stage, X8-101-continue-score, X8-101-continue-score-gpu, X8-101-continue-score-stage, X9-plasticity-101-train-score, X9-plasticity-101-train-score-gpu, X9-plasticity-101-train-score-stage | afterany | 0 | 1 | 8 | yes | 12 | [0, 0]; SP(b) .25–1 GPU-h/entry; startup separate; unresolved coverage has no finite upper total |
| SP-panel-x-tier1-s101 | sp | SP-prepare-x-tier1-s101 | afterany | 1 | 1 | 8 | yes | 6 / K=3 | [1.5, 6]; SP(b) .25–1 GPU-h/entry; startup separate; unresolved coverage has no finite upper total |
| SP-prepare-x-tier1-s102 | sp | SP-verify-pools, X-complete-102, X-complete-102-gpu, X-complete-102-stage, X-incoming-102, X-incoming-102-gpu, X-incoming-102-stage, X5-control-102-train-score, X5-control-102-train-score-gpu, X5-control-102-train-score-stage, X6-p-102-train-score, X6-p-102-train-score-gpu, X6-p-102-train-score-stage, X8-102-continue-score, X8-102-continue-score-gpu, X8-102-continue-score-stage, X9-plasticity-102-train-score, X9-plasticity-102-train-score-gpu, X9-plasticity-102-train-score-stage | afterany | 0 | 1 | 8 | yes | 12 | [0, 0]; SP(b) .25–1 GPU-h/entry; startup separate; unresolved coverage has no finite upper total |
| SP-panel-x-tier1-s102 | sp | SP-prepare-x-tier1-s102 | afterany | 1 | 1 | 8 | yes | 6 / K=3 | [1.5, 6]; SP(b) .25–1 GPU-h/entry; startup separate; unresolved coverage has no finite upper total |
| SP-prepare-x-tier1-s103 | sp | SP-verify-pools, X-complete-103, X-complete-103-gpu, X-complete-103-stage, X-incoming-103, X-incoming-103-gpu, X-incoming-103-stage, X5-control-103-train-score, X5-control-103-train-score-gpu, X5-control-103-train-score-stage, X6-p-103-train-score, X6-p-103-train-score-gpu, X6-p-103-train-score-stage, X8-103-continue-score, X8-103-continue-score-gpu, X8-103-continue-score-stage, X9-plasticity-103-train-score, X9-plasticity-103-train-score-gpu, X9-plasticity-103-train-score-stage | afterany | 0 | 1 | 8 | yes | 12 | [0, 0]; SP(b) .25–1 GPU-h/entry; startup separate; unresolved coverage has no finite upper total |
| SP-panel-x-tier1-s103 | sp | SP-prepare-x-tier1-s103 | afterany | 1 | 1 | 8 | yes | 6 / K=3 | [1.5, 6]; SP(b) .25–1 GPU-h/entry; startup separate; unresolved coverage has no finite upper total |
| SP-prepare-x-tier2.b-p2-s101 | sp | SP-verify-pools, X18-101-incoming, X18-101-incoming-gpu, X18-101-incoming-stage, X18-challenger-101-continue30-measure40-score, X18-challenger-101-continue30-measure40-score-gpu, X18-challenger-101-continue30-measure40-score-stage, X18-complete-101, X18-complete-101-gpu, X18-complete-101-stage, X18-control-101-continue10-score, X18-control-101-continue10-score-gpu, X18-control-101-continue10-score-stage | afterany | 0 | 1 | 8 | yes | 6 | [0, 0]; SP(b) .25–1 GPU-h/entry; startup separate; unresolved coverage has no finite upper total |
| SP-panel-x-tier2.b-p2-s101 | sp | SP-prepare-x-tier2.b-p2-s101 | afterany | 1 | 1 | 8 | yes | 3 / K=3 | [0.75, 3]; SP(b) .25–1 GPU-h/entry; startup separate; unresolved coverage has no finite upper total |
| SP-prepare-x-tier2.b-p2-s102 | sp | SP-verify-pools, X18-102-incoming, X18-102-incoming-gpu, X18-102-incoming-stage, X18-challenger-102-continue30-measure40-score, X18-challenger-102-continue30-measure40-score-gpu, X18-challenger-102-continue30-measure40-score-stage, X18-complete-102, X18-complete-102-gpu, X18-complete-102-stage, X18-control-102-continue10-score, X18-control-102-continue10-score-gpu, X18-control-102-continue10-score-stage | afterany | 0 | 1 | 8 | yes | 6 | [0, 0]; SP(b) .25–1 GPU-h/entry; startup separate; unresolved coverage has no finite upper total |
| SP-panel-x-tier2.b-p2-s102 | sp | SP-prepare-x-tier2.b-p2-s102 | afterany | 1 | 1 | 8 | yes | 3 / K=3 | [0.75, 3]; SP(b) .25–1 GPU-h/entry; startup separate; unresolved coverage has no finite upper total |
| SP-prepare-x-tier2.b-p2-s103 | sp | SP-verify-pools, X18-103-incoming, X18-103-incoming-gpu, X18-103-incoming-stage, X18-challenger-103-continue30-measure40-score, X18-challenger-103-continue30-measure40-score-gpu, X18-challenger-103-continue30-measure40-score-stage, X18-complete-103, X18-complete-103-gpu, X18-complete-103-stage, X18-control-103-continue10-score, X18-control-103-continue10-score-gpu, X18-control-103-continue10-score-stage | afterany | 0 | 1 | 8 | yes | 6 | [0, 0]; SP(b) .25–1 GPU-h/entry; startup separate; unresolved coverage has no finite upper total |
| SP-panel-x-tier2.b-p2-s103 | sp | SP-prepare-x-tier2.b-p2-s103 | afterany | 1 | 1 | 8 | yes | 3 / K=3 | [0.75, 3]; SP(b) .25–1 GPU-h/entry; startup separate; unresolved coverage has no finite upper total |
| SP-prepare-x-tier2.b-s101 | sp | SP-verify-pools, X17-101-incoming, X17-101-incoming-gpu, X17-101-incoming-stage, X17-challenger-101-continue30-measure40-score, X17-challenger-101-continue30-measure40-score-gpu, X17-challenger-101-continue30-measure40-score-stage, X17-complete-101, X17-complete-101-gpu, X17-complete-101-stage, X17-control-101-continue10-score, X17-control-101-continue10-score-gpu, X17-control-101-continue10-score-stage | afterany | 0 | 1 | 8 | yes | 6 | [0, 0]; SP(b) .25–1 GPU-h/entry; startup separate; unresolved coverage has no finite upper total |
| SP-panel-x-tier2.b-s101 | sp | SP-prepare-x-tier2.b-s101 | afterany | 1 | 1 | 8 | yes | 3 / K=3 | [0.75, 3]; SP(b) .25–1 GPU-h/entry; startup separate; unresolved coverage has no finite upper total |
| SP-prepare-x-tier2.b-s102 | sp | SP-verify-pools, X17-102-incoming, X17-102-incoming-gpu, X17-102-incoming-stage, X17-challenger-102-continue30-measure40-score, X17-challenger-102-continue30-measure40-score-gpu, X17-challenger-102-continue30-measure40-score-stage, X17-complete-102, X17-complete-102-gpu, X17-complete-102-stage, X17-control-102-continue10-score, X17-control-102-continue10-score-gpu, X17-control-102-continue10-score-stage | afterany | 0 | 1 | 8 | yes | 6 | [0, 0]; SP(b) .25–1 GPU-h/entry; startup separate; unresolved coverage has no finite upper total |
| SP-panel-x-tier2.b-s102 | sp | SP-prepare-x-tier2.b-s102 | afterany | 1 | 1 | 8 | yes | 3 / K=3 | [0.75, 3]; SP(b) .25–1 GPU-h/entry; startup separate; unresolved coverage has no finite upper total |
| SP-prepare-x-tier2.b-s103 | sp | SP-verify-pools, X17-103-incoming, X17-103-incoming-gpu, X17-103-incoming-stage, X17-challenger-103-continue30-measure40-score, X17-challenger-103-continue30-measure40-score-gpu, X17-challenger-103-continue30-measure40-score-stage, X17-complete-103, X17-complete-103-gpu, X17-complete-103-stage, X17-control-103-continue10-score, X17-control-103-continue10-score-gpu, X17-control-103-continue10-score-stage | afterany | 0 | 1 | 8 | yes | 6 | [0, 0]; SP(b) .25–1 GPU-h/entry; startup separate; unresolved coverage has no finite upper total |
| SP-panel-x-tier2.b-s103 | sp | SP-prepare-x-tier2.b-s103 | afterany | 1 | 1 | 8 | yes | 3 / K=3 | [0.75, 3]; SP(b) .25–1 GPU-h/entry; startup separate; unresolved coverage has no finite upper total |
| SP-prepare-x-tier2.incremental-p2-s101 | sp | SP-verify-pools, X20-101-incoming, X20-101-incoming-gpu, X20-101-incoming-stage, X20-challenger-101-continue10-score, X20-challenger-101-continue10-score-gpu, X20-challenger-101-continue10-score-stage, X20-complete-101, X20-complete-101-gpu, X20-complete-101-stage, X20-control-101-continue10-score, X20-control-101-continue10-score-gpu, X20-control-101-continue10-score-stage | afterany | 0 | 1 | 8 | yes | 6 | [0, 0]; SP(b) .25–1 GPU-h/entry; startup separate; unresolved coverage has no finite upper total |
| SP-panel-x-tier2.incremental-p2-s101 | sp | SP-prepare-x-tier2.incremental-p2-s101 | afterany | 1 | 1 | 8 | yes | 3 / K=3 | [0.75, 3]; SP(b) .25–1 GPU-h/entry; startup separate; unresolved coverage has no finite upper total |
| SP-prepare-x-tier2.incremental-p2-s102 | sp | SP-verify-pools, X20-102-incoming, X20-102-incoming-gpu, X20-102-incoming-stage, X20-challenger-102-continue10-score, X20-challenger-102-continue10-score-gpu, X20-challenger-102-continue10-score-stage, X20-complete-102, X20-complete-102-gpu, X20-complete-102-stage, X20-control-102-continue10-score, X20-control-102-continue10-score-gpu, X20-control-102-continue10-score-stage | afterany | 0 | 1 | 8 | yes | 6 | [0, 0]; SP(b) .25–1 GPU-h/entry; startup separate; unresolved coverage has no finite upper total |
| SP-panel-x-tier2.incremental-p2-s102 | sp | SP-prepare-x-tier2.incremental-p2-s102 | afterany | 1 | 1 | 8 | yes | 3 / K=3 | [0.75, 3]; SP(b) .25–1 GPU-h/entry; startup separate; unresolved coverage has no finite upper total |
| SP-prepare-x-tier2.incremental-p2-s103 | sp | SP-verify-pools, X20-103-incoming, X20-103-incoming-gpu, X20-103-incoming-stage, X20-challenger-103-continue10-score, X20-challenger-103-continue10-score-gpu, X20-challenger-103-continue10-score-stage, X20-complete-103, X20-complete-103-gpu, X20-complete-103-stage, X20-control-103-continue10-score, X20-control-103-continue10-score-gpu, X20-control-103-continue10-score-stage | afterany | 0 | 1 | 8 | yes | 6 | [0, 0]; SP(b) .25–1 GPU-h/entry; startup separate; unresolved coverage has no finite upper total |
| SP-panel-x-tier2.incremental-p2-s103 | sp | SP-prepare-x-tier2.incremental-p2-s103 | afterany | 1 | 1 | 8 | yes | 3 / K=3 | [0.75, 3]; SP(b) .25–1 GPU-h/entry; startup separate; unresolved coverage has no finite upper total |
| SP-prepare-x-tier2.incremental-s101 | sp | SP-verify-pools, X19-101-incoming, X19-101-incoming-gpu, X19-101-incoming-stage, X19-challenger-101-continue10-score, X19-challenger-101-continue10-score-gpu, X19-challenger-101-continue10-score-stage, X19-complete-101, X19-complete-101-gpu, X19-complete-101-stage, X19-control-101-continue10-score, X19-control-101-continue10-score-gpu, X19-control-101-continue10-score-stage | afterany | 0 | 1 | 8 | yes | 6 | [0, 0]; SP(b) .25–1 GPU-h/entry; startup separate; unresolved coverage has no finite upper total |
| SP-panel-x-tier2.incremental-s101 | sp | SP-prepare-x-tier2.incremental-s101 | afterany | 1 | 1 | 8 | yes | 3 / K=3 | [0.75, 3]; SP(b) .25–1 GPU-h/entry; startup separate; unresolved coverage has no finite upper total |
| SP-prepare-x-tier2.incremental-s102 | sp | SP-verify-pools, X19-102-incoming, X19-102-incoming-gpu, X19-102-incoming-stage, X19-challenger-102-continue10-score, X19-challenger-102-continue10-score-gpu, X19-challenger-102-continue10-score-stage, X19-complete-102, X19-complete-102-gpu, X19-complete-102-stage, X19-control-102-continue10-score, X19-control-102-continue10-score-gpu, X19-control-102-continue10-score-stage | afterany | 0 | 1 | 8 | yes | 6 | [0, 0]; SP(b) .25–1 GPU-h/entry; startup separate; unresolved coverage has no finite upper total |
| SP-panel-x-tier2.incremental-s102 | sp | SP-prepare-x-tier2.incremental-s102 | afterany | 1 | 1 | 8 | yes | 3 / K=3 | [0.75, 3]; SP(b) .25–1 GPU-h/entry; startup separate; unresolved coverage has no finite upper total |
| SP-prepare-x-tier2.incremental-s103 | sp | SP-verify-pools, X19-103-incoming, X19-103-incoming-gpu, X19-103-incoming-stage, X19-challenger-103-continue10-score, X19-challenger-103-continue10-score-gpu, X19-challenger-103-continue10-score-stage, X19-complete-103, X19-complete-103-gpu, X19-complete-103-stage, X19-control-103-continue10-score, X19-control-103-continue10-score-gpu, X19-control-103-continue10-score-stage | afterany | 0 | 1 | 8 | yes | 6 | [0, 0]; SP(b) .25–1 GPU-h/entry; startup separate; unresolved coverage has no finite upper total |
| SP-panel-x-tier2.incremental-s103 | sp | SP-prepare-x-tier2.incremental-s103 | afterany | 1 | 1 | 8 | yes | 3 / K=3 | [0.75, 3]; SP(b) .25–1 GPU-h/entry; startup separate; unresolved coverage has no finite upper total |
| SP-prepare-x-tier2.stack-drift-s101 | sp | SP-verify-pools, X15-101-incoming, X15-101-incoming-gpu, X15-101-incoming-stage, X15-challenger-101-continue10-score, X15-challenger-101-continue10-score-gpu, X15-challenger-101-continue10-score-stage, X15-complete-101, X15-complete-101-gpu, X15-complete-101-stage, X15-control-101-continue10-score, X15-control-101-continue10-score-gpu, X15-control-101-continue10-score-stage | afterany | 0 | 1 | 8 | yes | 6 | [0, 0]; SP(b) .25–1 GPU-h/entry; startup separate; unresolved coverage has no finite upper total |
| SP-panel-x-tier2.stack-drift-s101 | sp | SP-prepare-x-tier2.stack-drift-s101 | afterany | 1 | 1 | 8 | yes | 3 / K=3 | [0.75, 3]; SP(b) .25–1 GPU-h/entry; startup separate; unresolved coverage has no finite upper total |
| SP-prepare-x-tier2.stack-drift-s102 | sp | SP-verify-pools, X15-102-incoming, X15-102-incoming-gpu, X15-102-incoming-stage, X15-challenger-102-continue10-score, X15-challenger-102-continue10-score-gpu, X15-challenger-102-continue10-score-stage, X15-complete-102, X15-complete-102-gpu, X15-complete-102-stage, X15-control-102-continue10-score, X15-control-102-continue10-score-gpu, X15-control-102-continue10-score-stage | afterany | 0 | 1 | 8 | yes | 6 | [0, 0]; SP(b) .25–1 GPU-h/entry; startup separate; unresolved coverage has no finite upper total |
| SP-panel-x-tier2.stack-drift-s102 | sp | SP-prepare-x-tier2.stack-drift-s102 | afterany | 1 | 1 | 8 | yes | 3 / K=3 | [0.75, 3]; SP(b) .25–1 GPU-h/entry; startup separate; unresolved coverage has no finite upper total |
| SP-prepare-x-tier2.stack-drift-s103 | sp | SP-verify-pools, X15-103-incoming, X15-103-incoming-gpu, X15-103-incoming-stage, X15-challenger-103-continue10-score, X15-challenger-103-continue10-score-gpu, X15-challenger-103-continue10-score-stage, X15-complete-103, X15-complete-103-gpu, X15-complete-103-stage, X15-control-103-continue10-score, X15-control-103-continue10-score-gpu, X15-control-103-continue10-score-stage | afterany | 0 | 1 | 8 | yes | 6 | [0, 0]; SP(b) .25–1 GPU-h/entry; startup separate; unresolved coverage has no finite upper total |
| SP-panel-x-tier2.stack-drift-s103 | sp | SP-prepare-x-tier2.stack-drift-s103 | afterany | 1 | 1 | 8 | yes | 3 / K=3 | [0.75, 3]; SP(b) .25–1 GPU-h/entry; startup separate; unresolved coverage has no finite upper total |
| SP-prepare-x-tier2.stack-plasticity-s101 | sp | SP-verify-pools, X16-101-incoming, X16-101-incoming-gpu, X16-101-incoming-stage, X16-challenger-101-train-score, X16-challenger-101-train-score-gpu, X16-challenger-101-train-score-stage, X16-complete-101, X16-complete-101-gpu, X16-complete-101-stage, X16-control-101-train-score, X16-control-101-train-score-gpu, X16-control-101-train-score-stage | afterany | 0 | 1 | 8 | yes | 6 | [0, 0]; SP(b) .25–1 GPU-h/entry; startup separate; unresolved coverage has no finite upper total |
| SP-panel-x-tier2.stack-plasticity-s101 | sp | SP-prepare-x-tier2.stack-plasticity-s101 | afterany | 1 | 1 | 8 | yes | 3 / K=3 | [0.75, 3]; SP(b) .25–1 GPU-h/entry; startup separate; unresolved coverage has no finite upper total |
| SP-prepare-x-tier2.stack-plasticity-s102 | sp | SP-verify-pools, X16-102-incoming, X16-102-incoming-gpu, X16-102-incoming-stage, X16-challenger-102-train-score, X16-challenger-102-train-score-gpu, X16-challenger-102-train-score-stage, X16-complete-102, X16-complete-102-gpu, X16-complete-102-stage, X16-control-102-train-score, X16-control-102-train-score-gpu, X16-control-102-train-score-stage | afterany | 0 | 1 | 8 | yes | 6 | [0, 0]; SP(b) .25–1 GPU-h/entry; startup separate; unresolved coverage has no finite upper total |
| SP-panel-x-tier2.stack-plasticity-s102 | sp | SP-prepare-x-tier2.stack-plasticity-s102 | afterany | 1 | 1 | 8 | yes | 3 / K=3 | [0.75, 3]; SP(b) .25–1 GPU-h/entry; startup separate; unresolved coverage has no finite upper total |
| SP-prepare-x-tier2.stack-plasticity-s103 | sp | SP-verify-pools, X16-103-incoming, X16-103-incoming-gpu, X16-103-incoming-stage, X16-challenger-103-train-score, X16-challenger-103-train-score-gpu, X16-challenger-103-train-score-stage, X16-complete-103, X16-complete-103-gpu, X16-complete-103-stage, X16-control-103-train-score, X16-control-103-train-score-gpu, X16-control-103-train-score-stage | afterany | 0 | 1 | 8 | yes | 6 | [0, 0]; SP(b) .25–1 GPU-h/entry; startup separate; unresolved coverage has no finite upper total |
| SP-panel-x-tier2.stack-plasticity-s103 | sp | SP-prepare-x-tier2.stack-plasticity-s103 | afterany | 1 | 1 | 8 | yes | 3 / K=3 | [0.75, 3]; SP(b) .25–1 GPU-h/entry; startup separate; unresolved coverage has no finite upper total |
| SP-length | sp | M-final-score-101, M-final-score-101-gpu, M-final-score-101-stage, M-final-score-102, M-final-score-102-gpu, M-final-score-102-stage, M-final-score-103, M-final-score-103-gpu, M-final-score-103-stage, M10-101-incoming, M10-101-incoming-gpu, M10-101-incoming-stage, M10-101-score, M10-101-score-gpu, M10-101-score-stage, M10-102-incoming, M10-102-incoming-gpu, M10-102-incoming-stage, M10-102-score, M10-102-score-gpu, M10-102-score-stage, M10-103-incoming, M10-103-incoming-gpu, M10-103-incoming-stage, M10-103-score, M10-103-score-gpu, M10-103-score-stage, M3-C0-101-contract_nli-score, M3-C0-101-finqa-score, M3-C0-101-math_hard-score, M3-C0-102-contract_nli-score, M3-C0-102-finqa-score, M3-C0-102-math_hard-score, M3-C0-103-contract_nli-score, M3-C0-103-finqa-score, M3-C0-103-math_hard-score, M3-CR-101-contract_nli-score, M3-CR-101-finqa-score, M3-CR-101-math_hard-score, M3-CR-102-contract_nli-score, M3-CR-102-finqa-score, M3-CR-102-math_hard-score, M3-CR-103-contract_nli-score, M3-CR-103-finqa-score, M3-CR-103-math_hard-score, M3-G-101-contract_nli-score, M3-G-101-finqa-score, M3-G-101-math_hard-score, M3-G-102-contract_nli-score, M3-G-102-finqa-score, M3-G-102-math_hard-score, M3-G-103-contract_nli-score, M3-G-103-finqa-score, M3-G-103-math_hard-score, M3-K-101-contract_nli-score, M3-K-101-finqa-score, M3-K-101-math_hard-score, M3-K-102-contract_nli-score, M3-K-102-finqa-score, M3-K-102-math_hard-score, M3-K-103-contract_nli-score, M3-K-103-finqa-score, M3-K-103-math_hard-score, M3-base-101, M3-base-102, M3-base-103, M3-rescore-101, M3-rescore-102, M3-rescore-103, M4-101, M4-101-gpu, M4-101-stage, M4-102, M4-102-gpu, M4-102-stage, M4-103, M4-103-gpu, M4-103-stage, M6-101-incoming, M6-101-incoming-gpu, M6-101-incoming-stage, M6-101-score, M6-101-score-gpu, M6-101-score-stage, M6-102-incoming, M6-102-incoming-gpu, M6-102-incoming-stage, M6-102-score, M6-102-score-gpu, M6-102-score-stage, M6-103-incoming, M6-103-incoming-gpu, M6-103-incoming-stage, M6-103-score, M6-103-score-gpu, M6-103-score-stage, M7-101-incoming, M7-101-incoming-gpu, M7-101-incoming-stage, M7-101-score, M7-101-score-gpu, M7-101-score-stage, M7-102-incoming, M7-102-incoming-gpu, M7-102-incoming-stage, M7-102-score, M7-102-score-gpu, M7-102-score-stage, M7-103-incoming, M7-103-incoming-gpu, M7-103-incoming-stage, M7-103-score, M7-103-score-gpu, M7-103-score-stage, M8-101-incoming, M8-101-incoming-gpu, M8-101-incoming-stage, M8-101-score, M8-101-score-gpu, M8-101-score-stage, M8-102-incoming, M8-102-incoming-gpu, M8-102-incoming-stage, M8-102-score, M8-102-score-gpu, M8-102-score-stage, M8-103-incoming, M8-103-incoming-gpu, M8-103-incoming-stage, M8-103-score, M8-103-score-gpu, M8-103-score-stage, M8-compare-101, M8-compare-101-gpu, M8-compare-101-stage, M8-compare-102, M8-compare-102-gpu, M8-compare-102-stage, M8-compare-103, M8-compare-103-gpu, M8-compare-103-stage, M9-101-incoming, M9-101-incoming-gpu, M9-101-incoming-stage, M9-101-score, M9-101-score-gpu, M9-101-score-stage, M9-102-incoming, M9-102-incoming-gpu, M9-102-incoming-stage, M9-102-score, M9-102-score-gpu, M9-102-score-stage, M9-103-incoming, M9-103-incoming-gpu, M9-103-incoming-stage, M9-103-score, M9-103-score-gpu, M9-103-score-stage, M9-compare-101, M9-compare-101-gpu, M9-compare-101-stage, M9-compare-102, M9-compare-102-gpu, M9-compare-102-stage, M9-compare-103, M9-compare-103-gpu, M9-compare-103-stage, R-both-101-score, R-both-101-score-gpu, R-both-101-score-stage, R-both-102-score, R-both-102-score-gpu, R-both-102-score-stage, R-both-103-score, R-both-103-score-gpu, R-both-103-score-stage, R-broad-101-score, R-broad-101-score-gpu, R-broad-101-score-stage, R-broad-102-score, R-broad-102-score-gpu, R-broad-102-score-stage, R-broad-103-score, R-broad-103-score-gpu, R-broad-103-score-stage, R-compare-broad-101, R-compare-broad-101-gpu, R-compare-broad-101-stage, R-compare-broad-102, R-compare-broad-102-gpu, R-compare-broad-102-stage, R-compare-broad-103, R-compare-broad-103-gpu, R-compare-broad-103-stage, R-compare-full-101, R-compare-full-101-gpu, R-compare-full-101-stage, R-compare-full-102, R-compare-full-102-gpu, R-compare-full-102-stage, R-compare-full-103, R-compare-full-103-gpu, R-compare-full-103-stage, R-earlier-101-score, R-earlier-101-score-gpu, R-earlier-101-score-stage, R-earlier-102-score, R-earlier-102-score-gpu, R-earlier-102-score-stage, R-earlier-103-score, R-earlier-103-score-gpu, R-earlier-103-score-stage, R-generic-101-score, R-generic-101-score-gpu, R-generic-101-score-stage, R-generic-102-score, R-generic-102-score-gpu, R-generic-102-score-stage, R-generic-103-score, R-generic-103-score-gpu, R-generic-103-score-stage, R-incoming-101, R-incoming-101-gpu, R-incoming-101-stage, R-incoming-102, R-incoming-102-gpu, R-incoming-102-stage, R-incoming-103, R-incoming-103-gpu, R-incoming-103-stage, RA101, RA101-gpu, RA101-stage, RA102, RA102-gpu, RA102-stage, RA103, RA103-gpu, RA103-stage, X-complete-101, X-complete-101-gpu, X-complete-101-stage, X-complete-102, X-complete-102-gpu, X-complete-102-stage, X-complete-103, X-complete-103-gpu, X-complete-103-stage, X-incoming-101, X-incoming-101-gpu, X-incoming-101-stage, X-incoming-102, X-incoming-102-gpu, X-incoming-102-stage, X-incoming-103, X-incoming-103-gpu, X-incoming-103-stage, X15-101-incoming, X15-101-incoming-gpu, X15-101-incoming-stage, X15-102-incoming, X15-102-incoming-gpu, X15-102-incoming-stage, X15-103-incoming, X15-103-incoming-gpu, X15-103-incoming-stage, X15-challenger-101-continue10-score, X15-challenger-101-continue10-score-gpu, X15-challenger-101-continue10-score-stage, X15-challenger-102-continue10-score, X15-challenger-102-continue10-score-gpu, X15-challenger-102-continue10-score-stage, X15-challenger-103-continue10-score, X15-challenger-103-continue10-score-gpu, X15-challenger-103-continue10-score-stage, X15-complete-101, X15-complete-101-gpu, X15-complete-101-stage, X15-complete-102, X15-complete-102-gpu, X15-complete-102-stage, X15-complete-103, X15-complete-103-gpu, X15-complete-103-stage, X15-control-101-continue10-score, X15-control-101-continue10-score-gpu, X15-control-101-continue10-score-stage, X15-control-102-continue10-score, X15-control-102-continue10-score-gpu, X15-control-102-continue10-score-stage, X15-control-103-continue10-score, X15-control-103-continue10-score-gpu, X15-control-103-continue10-score-stage, X16-101-incoming, X16-101-incoming-gpu, X16-101-incoming-stage, X16-102-incoming, X16-102-incoming-gpu, X16-102-incoming-stage, X16-103-incoming, X16-103-incoming-gpu, X16-103-incoming-stage, X16-challenger-101-train-score, X16-challenger-101-train-score-gpu, X16-challenger-101-train-score-stage, X16-challenger-102-train-score, X16-challenger-102-train-score-gpu, X16-challenger-102-train-score-stage, X16-challenger-103-train-score, X16-challenger-103-train-score-gpu, X16-challenger-103-train-score-stage, X16-complete-101, X16-complete-101-gpu, X16-complete-101-stage, X16-complete-102, X16-complete-102-gpu, X16-complete-102-stage, X16-complete-103, X16-complete-103-gpu, X16-complete-103-stage, X16-control-101-train-score, X16-control-101-train-score-gpu, X16-control-101-train-score-stage, X16-control-102-train-score, X16-control-102-train-score-gpu, X16-control-102-train-score-stage, X16-control-103-train-score, X16-control-103-train-score-gpu, X16-control-103-train-score-stage, X17-101-incoming, X17-101-incoming-gpu, X17-101-incoming-stage, X17-102-incoming, X17-102-incoming-gpu, X17-102-incoming-stage, X17-103-incoming, X17-103-incoming-gpu, X17-103-incoming-stage, X17-challenger-101-continue30-measure40-score, X17-challenger-101-continue30-measure40-score-gpu, X17-challenger-101-continue30-measure40-score-stage, X17-challenger-102-continue30-measure40-score, X17-challenger-102-continue30-measure40-score-gpu, X17-challenger-102-continue30-measure40-score-stage, X17-challenger-103-continue30-measure40-score, X17-challenger-103-continue30-measure40-score-gpu, X17-challenger-103-continue30-measure40-score-stage, X17-complete-101, X17-complete-101-gpu, X17-complete-101-stage, X17-complete-102, X17-complete-102-gpu, X17-complete-102-stage, X17-complete-103, X17-complete-103-gpu, X17-complete-103-stage, X17-control-101-continue10-score, X17-control-101-continue10-score-gpu, X17-control-101-continue10-score-stage, X17-control-102-continue10-score, X17-control-102-continue10-score-gpu, X17-control-102-continue10-score-stage, X17-control-103-continue10-score, X17-control-103-continue10-score-gpu, X17-control-103-continue10-score-stage, X18-101-incoming, X18-101-incoming-gpu, X18-101-incoming-stage, X18-102-incoming, X18-102-incoming-gpu, X18-102-incoming-stage, X18-103-incoming, X18-103-incoming-gpu, X18-103-incoming-stage, X18-challenger-101-continue30-measure40-score, X18-challenger-101-continue30-measure40-score-gpu, X18-challenger-101-continue30-measure40-score-stage, X18-challenger-102-continue30-measure40-score, X18-challenger-102-continue30-measure40-score-gpu, X18-challenger-102-continue30-measure40-score-stage, X18-challenger-103-continue30-measure40-score, X18-challenger-103-continue30-measure40-score-gpu, X18-challenger-103-continue30-measure40-score-stage, X18-complete-101, X18-complete-101-gpu, X18-complete-101-stage, X18-complete-102, X18-complete-102-gpu, X18-complete-102-stage, X18-complete-103, X18-complete-103-gpu, X18-complete-103-stage, X18-control-101-continue10-score, X18-control-101-continue10-score-gpu, X18-control-101-continue10-score-stage, X18-control-102-continue10-score, X18-control-102-continue10-score-gpu, X18-control-102-continue10-score-stage, X18-control-103-continue10-score, X18-control-103-continue10-score-gpu, X18-control-103-continue10-score-stage, X19-101-incoming, X19-101-incoming-gpu, X19-101-incoming-stage, X19-102-incoming, X19-102-incoming-gpu, X19-102-incoming-stage, X19-103-incoming, X19-103-incoming-gpu, X19-103-incoming-stage, X19-challenger-101-continue10-score, X19-challenger-101-continue10-score-gpu, X19-challenger-101-continue10-score-stage, X19-challenger-102-continue10-score, X19-challenger-102-continue10-score-gpu, X19-challenger-102-continue10-score-stage, X19-challenger-103-continue10-score, X19-challenger-103-continue10-score-gpu, X19-challenger-103-continue10-score-stage, X19-complete-101, X19-complete-101-gpu, X19-complete-101-stage, X19-complete-102, X19-complete-102-gpu, X19-complete-102-stage, X19-complete-103, X19-complete-103-gpu, X19-complete-103-stage, X19-control-101-continue10-score, X19-control-101-continue10-score-gpu, X19-control-101-continue10-score-stage, X19-control-102-continue10-score, X19-control-102-continue10-score-gpu, X19-control-102-continue10-score-stage, X19-control-103-continue10-score, X19-control-103-continue10-score-gpu, X19-control-103-continue10-score-stage, X20-101-incoming, X20-101-incoming-gpu, X20-101-incoming-stage, X20-102-incoming, X20-102-incoming-gpu, X20-102-incoming-stage, X20-103-incoming, X20-103-incoming-gpu, X20-103-incoming-stage, X20-challenger-101-continue10-score, X20-challenger-101-continue10-score-gpu, X20-challenger-101-continue10-score-stage, X20-challenger-102-continue10-score, X20-challenger-102-continue10-score-gpu, X20-challenger-102-continue10-score-stage, X20-challenger-103-continue10-score, X20-challenger-103-continue10-score-gpu, X20-challenger-103-continue10-score-stage, X20-complete-101, X20-complete-101-gpu, X20-complete-101-stage, X20-complete-102, X20-complete-102-gpu, X20-complete-102-stage, X20-complete-103, X20-complete-103-gpu, X20-complete-103-stage, X20-control-101-continue10-score, X20-control-101-continue10-score-gpu, X20-control-101-continue10-score-stage, X20-control-102-continue10-score, X20-control-102-continue10-score-gpu, X20-control-102-continue10-score-stage, X20-control-103-continue10-score, X20-control-103-continue10-score-gpu, X20-control-103-continue10-score-stage, X5-control-101-train-score, X5-control-101-train-score-gpu, X5-control-101-train-score-stage, X5-control-102-train-score, X5-control-102-train-score-gpu, X5-control-102-train-score-stage, X5-control-103-train-score, X5-control-103-train-score-gpu, X5-control-103-train-score-stage, X6-p-101-train-score, X6-p-101-train-score-gpu, X6-p-101-train-score-stage, X6-p-102-train-score, X6-p-102-train-score-gpu, X6-p-102-train-score-stage, X6-p-103-train-score, X6-p-103-train-score-gpu, X6-p-103-train-score-stage, X8-101-continue-score, X8-101-continue-score-gpu, X8-101-continue-score-stage, X8-102-continue-score, X8-102-continue-score-gpu, X8-102-continue-score-stage, X8-103-continue-score, X8-103-continue-score-gpu, X8-103-continue-score-stage, X9-plasticity-101-train-score, X9-plasticity-101-train-score-gpu, X9-plasticity-101-train-score-stage, X9-plasticity-102-train-score, X9-plasticity-102-train-score-gpu, X9-plasticity-102-train-score-stage, X9-plasticity-103-train-score, X9-plasticity-103-train-score-gpu, X9-plasticity-103-train-score-stage | afterany | 0 | 1 | 8 | yes | 1 | [0, 0]; SP(b) .25–1 GPU-h/entry; startup separate; unresolved coverage has no finite upper total |
| SP-read | sp | SP-length, SP-panel-core-s101, SP-panel-core-s102, SP-panel-core-s103, SP-panel-m3-s101, SP-panel-m3-s102, SP-panel-m3-s103, SP-panel-reprobe-s101, SP-panel-reprobe-s102, SP-panel-reprobe-s103, SP-panel-return1.A101-s101, SP-panel-return1.A102-s102, SP-panel-return1.A103-s103, SP-panel-x-tier1-s101, SP-panel-x-tier1-s102, SP-panel-x-tier1-s103, SP-panel-x-tier2.b-p2-s101, SP-panel-x-tier2.b-p2-s102, SP-panel-x-tier2.b-p2-s103, SP-panel-x-tier2.b-s101, SP-panel-x-tier2.b-s102, SP-panel-x-tier2.b-s103, SP-panel-x-tier2.incremental-p2-s101, SP-panel-x-tier2.incremental-p2-s102, SP-panel-x-tier2.incremental-p2-s103, SP-panel-x-tier2.incremental-s101, SP-panel-x-tier2.incremental-s102, SP-panel-x-tier2.incremental-s103, SP-panel-x-tier2.stack-drift-s101, SP-panel-x-tier2.stack-drift-s102, SP-panel-x-tier2.stack-drift-s103, SP-panel-x-tier2.stack-plasticity-s101, SP-panel-x-tier2.stack-plasticity-s102, SP-panel-x-tier2.stack-plasticity-s103, SP-verify-pools | afterany | 0 | 1 | 8 | yes | 1 | [0, 0]; SP(b) .25–1 GPU-h/entry; startup separate; unresolved coverage has no finite upper total |
| PD-101-prep | pd | none | afterok | 0 | 1 | 8 | yes | CPU | [0, 0]; PD split phase; all seeds 63–98 GPU-h; measured phase cost pending |
| PD-101-pre | pd | PD-101-prep | afterok | 1 | 1 | 8 | yes | 3 / K=2 | None; PD split phase; all seeds 63–98 GPU-h; measured phase cost pending |
| PD-101-train | pd | PD-101-pre | afterok | 8 | 1 | 32 | no | phase worker | None; PD split phase; all seeds 63–98 GPU-h; measured phase cost pending |
| PD-101-finalize | pd | PD-101-train | afterok | 0 | 1 | 16 | yes | CPU | [0, 0]; PD split phase; all seeds 63–98 GPU-h; measured phase cost pending |
| PD-101-post | pd | PD-101-finalize | afterok | 1 | 1 | 8 | yes | 18 / K=2 | None; PD split phase; all seeds 63–98 GPU-h; measured phase cost pending |
| PD-101-collect | pd | PD-101-post | afterany | 0 | 1 | 4 | yes | CPU | [0, 0]; PD split phase; all seeds 63–98 GPU-h; measured phase cost pending |
| PD-102-prep | pd | none | afterok | 0 | 1 | 8 | yes | CPU | [0, 0]; PD split phase; all seeds 63–98 GPU-h; measured phase cost pending |
| PD-102-pre | pd | PD-102-prep | afterok | 1 | 1 | 8 | yes | 3 / K=2 | None; PD split phase; all seeds 63–98 GPU-h; measured phase cost pending |
| PD-102-train | pd | PD-102-pre | afterok | 8 | 1 | 32 | no | phase worker | None; PD split phase; all seeds 63–98 GPU-h; measured phase cost pending |
| PD-102-finalize | pd | PD-102-train | afterok | 0 | 1 | 16 | yes | CPU | [0, 0]; PD split phase; all seeds 63–98 GPU-h; measured phase cost pending |
| PD-102-post | pd | PD-102-finalize | afterok | 1 | 1 | 8 | yes | 18 / K=2 | None; PD split phase; all seeds 63–98 GPU-h; measured phase cost pending |
| PD-102-collect | pd | PD-102-post | afterany | 0 | 1 | 4 | yes | CPU | [0, 0]; PD split phase; all seeds 63–98 GPU-h; measured phase cost pending |
| PD-103-prep | pd | none | afterok | 0 | 1 | 8 | yes | CPU | [0, 0]; PD split phase; all seeds 63–98 GPU-h; measured phase cost pending |
| PD-103-pre | pd | PD-103-prep | afterok | 1 | 1 | 8 | yes | 3 / K=2 | None; PD split phase; all seeds 63–98 GPU-h; measured phase cost pending |
| PD-103-train | pd | PD-103-pre | afterok | 8 | 1 | 32 | no | phase worker | None; PD split phase; all seeds 63–98 GPU-h; measured phase cost pending |
| PD-103-finalize | pd | PD-103-train | afterok | 0 | 1 | 16 | yes | CPU | [0, 0]; PD split phase; all seeds 63–98 GPU-h; measured phase cost pending |
| PD-103-post | pd | PD-103-finalize | afterok | 1 | 1 | 8 | yes | 18 / K=2 | None; PD split phase; all seeds 63–98 GPU-h; measured phase cost pending |
| PD-103-collect | pd | PD-103-post | afterany | 0 | 1 | 4 | yes | CPU | [0, 0]; PD split phase; all seeds 63–98 GPU-h; measured phase cost pending |
| PD-read | pd | PD-101-collect, PD-102-collect, PD-103-collect | afterany | 0 | 1 | 4 | yes | CPU | [0, 0]; PD split phase; all seeds 63–98 GPU-h; measured phase cost pending |
| A4-bind | 4a | none | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 0} | [0, 0]; 0 GPU-h; CPU work only |
| A4-context-m1 | 4a | A4-bind | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 0} | [0, 0]; 0 GPU-h; CPU work only |
| A4-arrival-prepare-m1 | 4a | A4-context-m1 | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 0} | [0, 0]; 0 GPU-h; CPU work only |
| A4-arrival-m1 | 4a | A4-arrival-prepare-m1 | afterok | 1 | 1 | 8 | yes | {'dispatch': 1, 'score': 11, 'nll': 0, 'trainer': 0, 'export': 0} / K=3 | None; 1 × allocation wall hours, including load, failures and idle; vLLM K=3; HF NLL K=1 |
| A4-ledger-m1 | 4a | A4-arrival-m1 | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 0} | [0, 0]; 0 GPU-h; CPU work only |
| A4-direct-s101-m1-prepare | 4a | A4-ledger-m1 | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 0} | [0, 0]; 0 GPU-h; CPU work only |
| A4-direct-s101-m1-train | 4a | A4-direct-s101-m1-prepare | afterok | 8 | 1 | 32 | no | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 1, 'export': 0} | None; 8 × measured training wall hours; limit 1.5 × bound W stage GPU-h |
| A4-direct-s101-m1-export | 4a | A4-direct-s101-m1-train | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 1} | [0, 0]; 0 GPU-h; CPU work only |
| A4-direct-s101-m1-score-prepare | 4a | A4-direct-s101-m1-export | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 0} | [0, 0]; 0 GPU-h; CPU work only |
| A4-direct-s101-m1-score | 4a | A4-direct-s101-m1-score-prepare | afterok | 1 | 1 | 8 | yes | {'dispatch': 1, 'score': 3, 'nll': 2, 'trainer': 0, 'export': 0} / K=3 | None; 1 × allocation wall hours, including load, failures and idle; vLLM K=3; HF NLL K=1 |
| A4-direct-s101-m1-finish | 4a | A4-direct-s101-m1-score | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 0} | [0, 0]; 0 GPU-h; CPU work only |
| A4-direct-s102-m1-prepare | 4a | A4-ledger-m1 | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 0} | [0, 0]; 0 GPU-h; CPU work only |
| A4-direct-s102-m1-train | 4a | A4-direct-s102-m1-prepare | afterok | 8 | 1 | 32 | no | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 1, 'export': 0} | None; 8 × measured training wall hours; limit 1.5 × bound W stage GPU-h |
| A4-direct-s102-m1-export | 4a | A4-direct-s102-m1-train | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 1} | [0, 0]; 0 GPU-h; CPU work only |
| A4-direct-s102-m1-score-prepare | 4a | A4-direct-s102-m1-export | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 0} | [0, 0]; 0 GPU-h; CPU work only |
| A4-direct-s102-m1-score | 4a | A4-direct-s102-m1-score-prepare | afterok | 1 | 1 | 8 | yes | {'dispatch': 1, 'score': 3, 'nll': 2, 'trainer': 0, 'export': 0} / K=3 | None; 1 × allocation wall hours, including load, failures and idle; vLLM K=3; HF NLL K=1 |
| A4-direct-s102-m1-finish | 4a | A4-direct-s102-m1-score | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 0} | [0, 0]; 0 GPU-h; CPU work only |
| A4-direct-s103-m1-prepare | 4a | A4-ledger-m1 | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 0} | [0, 0]; 0 GPU-h; CPU work only |
| A4-direct-s103-m1-train | 4a | A4-direct-s103-m1-prepare | afterok | 8 | 1 | 32 | no | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 1, 'export': 0} | None; 8 × measured training wall hours; limit 1.5 × bound W stage GPU-h |
| A4-direct-s103-m1-export | 4a | A4-direct-s103-m1-train | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 1} | [0, 0]; 0 GPU-h; CPU work only |
| A4-direct-s103-m1-score-prepare | 4a | A4-direct-s103-m1-export | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 0} | [0, 0]; 0 GPU-h; CPU work only |
| A4-direct-s103-m1-score | 4a | A4-direct-s103-m1-score-prepare | afterok | 1 | 1 | 8 | yes | {'dispatch': 1, 'score': 3, 'nll': 2, 'trainer': 0, 'export': 0} / K=3 | None; 1 × allocation wall hours, including load, failures and idle; vLLM K=3; HF NLL K=1 |
| A4-direct-s103-m1-finish | 4a | A4-direct-s103-m1-score | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 0} | [0, 0]; 0 GPU-h; CPU work only |
| A4-month-m1 | 4a | A4-ledger-m1, A4-direct-s101-m1-finish, A4-direct-s102-m1-finish, A4-direct-s103-m1-finish | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 0} | [0, 0]; 0 GPU-h; CPU work only |
| A4-context-m2 | 4a | A4-month-m1 | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 0} | [0, 0]; 0 GPU-h; CPU work only |
| A4-arrival-prepare-m2 | 4a | A4-context-m2 | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 0} | [0, 0]; 0 GPU-h; CPU work only |
| A4-arrival-m2 | 4a | A4-arrival-prepare-m2 | afterok | 1 | 1 | 8 | yes | {'dispatch': 1, 'score': 11, 'nll': 0, 'trainer': 0, 'export': 0} / K=3 | None; 1 × allocation wall hours, including load, failures and idle; vLLM K=3; HF NLL K=1 |
| A4-ledger-m2 | 4a | A4-arrival-m2 | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 0} | [0, 0]; 0 GPU-h; CPU work only |
| A4-direct-s101-m2-prepare | 4a | A4-ledger-m2 | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 0} | [0, 0]; 0 GPU-h; CPU work only |
| A4-direct-s101-m2-train | 4a | A4-direct-s101-m2-prepare | afterok | 8 | 1 | 32 | no | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 1, 'export': 0} | None; 8 × measured training wall hours; limit 1.5 × bound W stage GPU-h |
| A4-direct-s101-m2-export | 4a | A4-direct-s101-m2-train | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 1} | [0, 0]; 0 GPU-h; CPU work only |
| A4-direct-s101-m2-score-prepare | 4a | A4-direct-s101-m2-export | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 0} | [0, 0]; 0 GPU-h; CPU work only |
| A4-direct-s101-m2-score | 4a | A4-direct-s101-m2-score-prepare | afterok | 1 | 1 | 8 | yes | {'dispatch': 1, 'score': 4, 'nll': 2, 'trainer': 0, 'export': 0} / K=3 | None; 1 × allocation wall hours, including load, failures and idle; vLLM K=3; HF NLL K=1 |
| A4-direct-s101-m2-finish | 4a | A4-direct-s101-m2-score | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 0} | [0, 0]; 0 GPU-h; CPU work only |
| A4-direct-s102-m2-prepare | 4a | A4-ledger-m2 | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 0} | [0, 0]; 0 GPU-h; CPU work only |
| A4-direct-s102-m2-train | 4a | A4-direct-s102-m2-prepare | afterok | 8 | 1 | 32 | no | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 1, 'export': 0} | None; 8 × measured training wall hours; limit 1.5 × bound W stage GPU-h |
| A4-direct-s102-m2-export | 4a | A4-direct-s102-m2-train | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 1} | [0, 0]; 0 GPU-h; CPU work only |
| A4-direct-s102-m2-score-prepare | 4a | A4-direct-s102-m2-export | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 0} | [0, 0]; 0 GPU-h; CPU work only |
| A4-direct-s102-m2-score | 4a | A4-direct-s102-m2-score-prepare | afterok | 1 | 1 | 8 | yes | {'dispatch': 1, 'score': 4, 'nll': 2, 'trainer': 0, 'export': 0} / K=3 | None; 1 × allocation wall hours, including load, failures and idle; vLLM K=3; HF NLL K=1 |
| A4-direct-s102-m2-finish | 4a | A4-direct-s102-m2-score | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 0} | [0, 0]; 0 GPU-h; CPU work only |
| A4-direct-s103-m2-prepare | 4a | A4-ledger-m2 | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 0} | [0, 0]; 0 GPU-h; CPU work only |
| A4-direct-s103-m2-train | 4a | A4-direct-s103-m2-prepare | afterok | 8 | 1 | 32 | no | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 1, 'export': 0} | None; 8 × measured training wall hours; limit 1.5 × bound W stage GPU-h |
| A4-direct-s103-m2-export | 4a | A4-direct-s103-m2-train | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 1} | [0, 0]; 0 GPU-h; CPU work only |
| A4-direct-s103-m2-score-prepare | 4a | A4-direct-s103-m2-export | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 0} | [0, 0]; 0 GPU-h; CPU work only |
| A4-direct-s103-m2-score | 4a | A4-direct-s103-m2-score-prepare | afterok | 1 | 1 | 8 | yes | {'dispatch': 1, 'score': 4, 'nll': 2, 'trainer': 0, 'export': 0} / K=3 | None; 1 × allocation wall hours, including load, failures and idle; vLLM K=3; HF NLL K=1 |
| A4-direct-s103-m2-finish | 4a | A4-direct-s103-m2-score | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 0} | [0, 0]; 0 GPU-h; CPU work only |
| A4-consolidation-s101-m2-prepare | 4a | A4-ledger-m2 | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 0} | [0, 0]; 0 GPU-h; CPU work only |
| A4-consolidation-s101-m2-train | 4a | A4-consolidation-s101-m2-prepare | afterok | 8 | 1 | 32 | no | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 1, 'export': 0} | None; 8 × measured training wall hours; limit 1.5 × bound W stage GPU-h |
| A4-consolidation-s101-m2-export | 4a | A4-consolidation-s101-m2-train | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 1} | [0, 0]; 0 GPU-h; CPU work only |
| A4-consolidation-s101-m2-score-prepare | 4a | A4-consolidation-s101-m2-export | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 0} | [0, 0]; 0 GPU-h; CPU work only |
| A4-consolidation-s101-m2-score | 4a | A4-consolidation-s101-m2-score-prepare | afterok | 1 | 1 | 8 | yes | {'dispatch': 1, 'score': 6, 'nll': 2, 'trainer': 0, 'export': 0} / K=3 | None; 1 × allocation wall hours, including load, failures and idle; vLLM K=3; HF NLL K=1 |
| A4-consolidation-s101-m2-finish | 4a | A4-consolidation-s101-m2-score | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 0} | [0, 0]; 0 GPU-h; CPU work only |
| A4-consolidation-s102-m2-prepare | 4a | A4-ledger-m2 | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 0} | [0, 0]; 0 GPU-h; CPU work only |
| A4-consolidation-s102-m2-train | 4a | A4-consolidation-s102-m2-prepare | afterok | 8 | 1 | 32 | no | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 1, 'export': 0} | None; 8 × measured training wall hours; limit 1.5 × bound W stage GPU-h |
| A4-consolidation-s102-m2-export | 4a | A4-consolidation-s102-m2-train | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 1} | [0, 0]; 0 GPU-h; CPU work only |
| A4-consolidation-s102-m2-score-prepare | 4a | A4-consolidation-s102-m2-export | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 0} | [0, 0]; 0 GPU-h; CPU work only |
| A4-consolidation-s102-m2-score | 4a | A4-consolidation-s102-m2-score-prepare | afterok | 1 | 1 | 8 | yes | {'dispatch': 1, 'score': 6, 'nll': 2, 'trainer': 0, 'export': 0} / K=3 | None; 1 × allocation wall hours, including load, failures and idle; vLLM K=3; HF NLL K=1 |
| A4-consolidation-s102-m2-finish | 4a | A4-consolidation-s102-m2-score | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 0} | [0, 0]; 0 GPU-h; CPU work only |
| A4-consolidation-s103-m2-prepare | 4a | A4-ledger-m2 | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 0} | [0, 0]; 0 GPU-h; CPU work only |
| A4-consolidation-s103-m2-train | 4a | A4-consolidation-s103-m2-prepare | afterok | 8 | 1 | 32 | no | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 1, 'export': 0} | None; 8 × measured training wall hours; limit 1.5 × bound W stage GPU-h |
| A4-consolidation-s103-m2-export | 4a | A4-consolidation-s103-m2-train | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 1} | [0, 0]; 0 GPU-h; CPU work only |
| A4-consolidation-s103-m2-score-prepare | 4a | A4-consolidation-s103-m2-export | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 0} | [0, 0]; 0 GPU-h; CPU work only |
| A4-consolidation-s103-m2-score | 4a | A4-consolidation-s103-m2-score-prepare | afterok | 1 | 1 | 8 | yes | {'dispatch': 1, 'score': 6, 'nll': 2, 'trainer': 0, 'export': 0} / K=3 | None; 1 × allocation wall hours, including load, failures and idle; vLLM K=3; HF NLL K=1 |
| A4-consolidation-s103-m2-finish | 4a | A4-consolidation-s103-m2-score | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 0} | [0, 0]; 0 GPU-h; CPU work only |
| A4-month-m2 | 4a | A4-ledger-m2, A4-direct-s101-m2-finish, A4-direct-s102-m2-finish, A4-direct-s103-m2-finish, A4-consolidation-s101-m2-finish, A4-consolidation-s102-m2-finish, A4-consolidation-s103-m2-finish | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 0} | [0, 0]; 0 GPU-h; CPU work only |
| A4-context-m3 | 4a | A4-month-m2 | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 0} | [0, 0]; 0 GPU-h; CPU work only |
| A4-arrival-prepare-m3 | 4a | A4-context-m3 | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 0} | [0, 0]; 0 GPU-h; CPU work only |
| A4-arrival-m3 | 4a | A4-arrival-prepare-m3 | afterok | 1 | 1 | 8 | yes | {'dispatch': 1, 'score': 11, 'nll': 0, 'trainer': 0, 'export': 0} / K=3 | None; 1 × allocation wall hours, including load, failures and idle; vLLM K=3; HF NLL K=1 |
| A4-ledger-m3 | 4a | A4-arrival-m3 | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 0} | [0, 0]; 0 GPU-h; CPU work only |
| A4-direct-s101-m3-prepare | 4a | A4-ledger-m3 | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 0} | [0, 0]; 0 GPU-h; CPU work only |
| A4-direct-s101-m3-train | 4a | A4-direct-s101-m3-prepare | afterok | 8 | 1 | 32 | no | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 1, 'export': 0} | None; 8 × measured training wall hours; limit 1.5 × bound W stage GPU-h |
| A4-direct-s101-m3-export | 4a | A4-direct-s101-m3-train | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 1} | [0, 0]; 0 GPU-h; CPU work only |
| A4-direct-s101-m3-score-prepare | 4a | A4-direct-s101-m3-export | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 0} | [0, 0]; 0 GPU-h; CPU work only |
| A4-direct-s101-m3-score | 4a | A4-direct-s101-m3-score-prepare | afterok | 1 | 1 | 8 | yes | {'dispatch': 1, 'score': 5, 'nll': 2, 'trainer': 0, 'export': 0} / K=3 | None; 1 × allocation wall hours, including load, failures and idle; vLLM K=3; HF NLL K=1 |
| A4-direct-s101-m3-finish | 4a | A4-direct-s101-m3-score | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 0} | [0, 0]; 0 GPU-h; CPU work only |
| A4-direct-s102-m3-prepare | 4a | A4-ledger-m3 | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 0} | [0, 0]; 0 GPU-h; CPU work only |
| A4-direct-s102-m3-train | 4a | A4-direct-s102-m3-prepare | afterok | 8 | 1 | 32 | no | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 1, 'export': 0} | None; 8 × measured training wall hours; limit 1.5 × bound W stage GPU-h |
| A4-direct-s102-m3-export | 4a | A4-direct-s102-m3-train | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 1} | [0, 0]; 0 GPU-h; CPU work only |
| A4-direct-s102-m3-score-prepare | 4a | A4-direct-s102-m3-export | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 0} | [0, 0]; 0 GPU-h; CPU work only |
| A4-direct-s102-m3-score | 4a | A4-direct-s102-m3-score-prepare | afterok | 1 | 1 | 8 | yes | {'dispatch': 1, 'score': 5, 'nll': 2, 'trainer': 0, 'export': 0} / K=3 | None; 1 × allocation wall hours, including load, failures and idle; vLLM K=3; HF NLL K=1 |
| A4-direct-s102-m3-finish | 4a | A4-direct-s102-m3-score | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 0} | [0, 0]; 0 GPU-h; CPU work only |
| A4-direct-s103-m3-prepare | 4a | A4-ledger-m3 | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 0} | [0, 0]; 0 GPU-h; CPU work only |
| A4-direct-s103-m3-train | 4a | A4-direct-s103-m3-prepare | afterok | 8 | 1 | 32 | no | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 1, 'export': 0} | None; 8 × measured training wall hours; limit 1.5 × bound W stage GPU-h |
| A4-direct-s103-m3-export | 4a | A4-direct-s103-m3-train | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 1} | [0, 0]; 0 GPU-h; CPU work only |
| A4-direct-s103-m3-score-prepare | 4a | A4-direct-s103-m3-export | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 0} | [0, 0]; 0 GPU-h; CPU work only |
| A4-direct-s103-m3-score | 4a | A4-direct-s103-m3-score-prepare | afterok | 1 | 1 | 8 | yes | {'dispatch': 1, 'score': 5, 'nll': 2, 'trainer': 0, 'export': 0} / K=3 | None; 1 × allocation wall hours, including load, failures and idle; vLLM K=3; HF NLL K=1 |
| A4-direct-s103-m3-finish | 4a | A4-direct-s103-m3-score | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 0} | [0, 0]; 0 GPU-h; CPU work only |
| A4-month-m3 | 4a | A4-ledger-m3, A4-direct-s101-m3-finish, A4-direct-s102-m3-finish, A4-direct-s103-m3-finish | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 0} | [0, 0]; 0 GPU-h; CPU work only |
| A4-context-m4 | 4a | A4-month-m3 | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 0} | [0, 0]; 0 GPU-h; CPU work only |
| A4-arrival-prepare-m4 | 4a | A4-context-m4 | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 0} | [0, 0]; 0 GPU-h; CPU work only |
| A4-arrival-m4 | 4a | A4-arrival-prepare-m4 | afterok | 1 | 1 | 8 | yes | {'dispatch': 1, 'score': 11, 'nll': 0, 'trainer': 0, 'export': 0} / K=3 | None; 1 × allocation wall hours, including load, failures and idle; vLLM K=3; HF NLL K=1 |
| A4-ledger-m4 | 4a | A4-arrival-m4 | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 0} | [0, 0]; 0 GPU-h; CPU work only |
| A4-direct-s101-m4-prepare | 4a | A4-ledger-m4 | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 0} | [0, 0]; 0 GPU-h; CPU work only |
| A4-direct-s101-m4-train | 4a | A4-direct-s101-m4-prepare | afterok | 8 | 1 | 32 | no | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 1, 'export': 0} | None; 8 × measured training wall hours; limit 1.5 × bound W stage GPU-h |
| A4-direct-s101-m4-export | 4a | A4-direct-s101-m4-train | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 1} | [0, 0]; 0 GPU-h; CPU work only |
| A4-direct-s101-m4-score-prepare | 4a | A4-direct-s101-m4-export | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 0} | [0, 0]; 0 GPU-h; CPU work only |
| A4-direct-s101-m4-score | 4a | A4-direct-s101-m4-score-prepare | afterok | 1 | 1 | 8 | yes | {'dispatch': 1, 'score': 6, 'nll': 2, 'trainer': 0, 'export': 0} / K=3 | None; 1 × allocation wall hours, including load, failures and idle; vLLM K=3; HF NLL K=1 |
| A4-direct-s101-m4-finish | 4a | A4-direct-s101-m4-score | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 0} | [0, 0]; 0 GPU-h; CPU work only |
| A4-direct-s102-m4-prepare | 4a | A4-ledger-m4 | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 0} | [0, 0]; 0 GPU-h; CPU work only |
| A4-direct-s102-m4-train | 4a | A4-direct-s102-m4-prepare | afterok | 8 | 1 | 32 | no | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 1, 'export': 0} | None; 8 × measured training wall hours; limit 1.5 × bound W stage GPU-h |
| A4-direct-s102-m4-export | 4a | A4-direct-s102-m4-train | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 1} | [0, 0]; 0 GPU-h; CPU work only |
| A4-direct-s102-m4-score-prepare | 4a | A4-direct-s102-m4-export | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 0} | [0, 0]; 0 GPU-h; CPU work only |
| A4-direct-s102-m4-score | 4a | A4-direct-s102-m4-score-prepare | afterok | 1 | 1 | 8 | yes | {'dispatch': 1, 'score': 6, 'nll': 2, 'trainer': 0, 'export': 0} / K=3 | None; 1 × allocation wall hours, including load, failures and idle; vLLM K=3; HF NLL K=1 |
| A4-direct-s102-m4-finish | 4a | A4-direct-s102-m4-score | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 0} | [0, 0]; 0 GPU-h; CPU work only |
| A4-direct-s103-m4-prepare | 4a | A4-ledger-m4 | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 0} | [0, 0]; 0 GPU-h; CPU work only |
| A4-direct-s103-m4-train | 4a | A4-direct-s103-m4-prepare | afterok | 8 | 1 | 32 | no | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 1, 'export': 0} | None; 8 × measured training wall hours; limit 1.5 × bound W stage GPU-h |
| A4-direct-s103-m4-export | 4a | A4-direct-s103-m4-train | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 1} | [0, 0]; 0 GPU-h; CPU work only |
| A4-direct-s103-m4-score-prepare | 4a | A4-direct-s103-m4-export | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 0} | [0, 0]; 0 GPU-h; CPU work only |
| A4-direct-s103-m4-score | 4a | A4-direct-s103-m4-score-prepare | afterok | 1 | 1 | 8 | yes | {'dispatch': 1, 'score': 6, 'nll': 2, 'trainer': 0, 'export': 0} / K=3 | None; 1 × allocation wall hours, including load, failures and idle; vLLM K=3; HF NLL K=1 |
| A4-direct-s103-m4-finish | 4a | A4-direct-s103-m4-score | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 0} | [0, 0]; 0 GPU-h; CPU work only |
| A4-consolidation-s101-m4-prepare | 4a | A4-ledger-m4 | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 0} | [0, 0]; 0 GPU-h; CPU work only |
| A4-consolidation-s101-m4-train | 4a | A4-consolidation-s101-m4-prepare | afterok | 8 | 1 | 32 | no | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 1, 'export': 0} | None; 8 × measured training wall hours; limit 1.5 × bound W stage GPU-h |
| A4-consolidation-s101-m4-export | 4a | A4-consolidation-s101-m4-train | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 1} | [0, 0]; 0 GPU-h; CPU work only |
| A4-consolidation-s101-m4-score-prepare | 4a | A4-consolidation-s101-m4-export | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 0} | [0, 0]; 0 GPU-h; CPU work only |
| A4-consolidation-s101-m4-score | 4a | A4-consolidation-s101-m4-score-prepare | afterok | 1 | 1 | 8 | yes | {'dispatch': 1, 'score': 10, 'nll': 2, 'trainer': 0, 'export': 0} / K=3 | None; 1 × allocation wall hours, including load, failures and idle; vLLM K=3; HF NLL K=1 |
| A4-consolidation-s101-m4-finish | 4a | A4-consolidation-s101-m4-score | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 0} | [0, 0]; 0 GPU-h; CPU work only |
| A4-consolidation-s102-m4-prepare | 4a | A4-ledger-m4 | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 0} | [0, 0]; 0 GPU-h; CPU work only |
| A4-consolidation-s102-m4-train | 4a | A4-consolidation-s102-m4-prepare | afterok | 8 | 1 | 32 | no | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 1, 'export': 0} | None; 8 × measured training wall hours; limit 1.5 × bound W stage GPU-h |
| A4-consolidation-s102-m4-export | 4a | A4-consolidation-s102-m4-train | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 1} | [0, 0]; 0 GPU-h; CPU work only |
| A4-consolidation-s102-m4-score-prepare | 4a | A4-consolidation-s102-m4-export | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 0} | [0, 0]; 0 GPU-h; CPU work only |
| A4-consolidation-s102-m4-score | 4a | A4-consolidation-s102-m4-score-prepare | afterok | 1 | 1 | 8 | yes | {'dispatch': 1, 'score': 10, 'nll': 2, 'trainer': 0, 'export': 0} / K=3 | None; 1 × allocation wall hours, including load, failures and idle; vLLM K=3; HF NLL K=1 |
| A4-consolidation-s102-m4-finish | 4a | A4-consolidation-s102-m4-score | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 0} | [0, 0]; 0 GPU-h; CPU work only |
| A4-consolidation-s103-m4-prepare | 4a | A4-ledger-m4 | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 0} | [0, 0]; 0 GPU-h; CPU work only |
| A4-consolidation-s103-m4-train | 4a | A4-consolidation-s103-m4-prepare | afterok | 8 | 1 | 32 | no | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 1, 'export': 0} | None; 8 × measured training wall hours; limit 1.5 × bound W stage GPU-h |
| A4-consolidation-s103-m4-export | 4a | A4-consolidation-s103-m4-train | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 1} | [0, 0]; 0 GPU-h; CPU work only |
| A4-consolidation-s103-m4-score-prepare | 4a | A4-consolidation-s103-m4-export | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 0} | [0, 0]; 0 GPU-h; CPU work only |
| A4-consolidation-s103-m4-score | 4a | A4-consolidation-s103-m4-score-prepare | afterok | 1 | 1 | 8 | yes | {'dispatch': 1, 'score': 10, 'nll': 2, 'trainer': 0, 'export': 0} / K=3 | None; 1 × allocation wall hours, including load, failures and idle; vLLM K=3; HF NLL K=1 |
| A4-consolidation-s103-m4-finish | 4a | A4-consolidation-s103-m4-score | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 0} | [0, 0]; 0 GPU-h; CPU work only |
| A4-month-m4 | 4a | A4-ledger-m4, A4-direct-s101-m4-finish, A4-direct-s102-m4-finish, A4-direct-s103-m4-finish, A4-consolidation-s101-m4-finish, A4-consolidation-s102-m4-finish, A4-consolidation-s103-m4-finish | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 0} | [0, 0]; 0 GPU-h; CPU work only |
| A4-context-m5 | 4a | A4-month-m4 | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 0} | [0, 0]; 0 GPU-h; CPU work only |
| A4-arrival-prepare-m5 | 4a | A4-context-m5 | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 0} | [0, 0]; 0 GPU-h; CPU work only |
| A4-arrival-m5 | 4a | A4-arrival-prepare-m5 | afterok | 1 | 1 | 8 | yes | {'dispatch': 1, 'score': 11, 'nll': 0, 'trainer': 0, 'export': 0} / K=3 | None; 1 × allocation wall hours, including load, failures and idle; vLLM K=3; HF NLL K=1 |
| A4-ledger-m5 | 4a | A4-arrival-m5 | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 0} | [0, 0]; 0 GPU-h; CPU work only |
| A4-direct-s101-m5-prepare | 4a | A4-ledger-m5 | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 0} | [0, 0]; 0 GPU-h; CPU work only |
| A4-direct-s101-m5-train | 4a | A4-direct-s101-m5-prepare | afterok | 8 | 1 | 32 | no | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 1, 'export': 0} | None; 8 × measured training wall hours; limit 1.5 × bound W stage GPU-h |
| A4-direct-s101-m5-export | 4a | A4-direct-s101-m5-train | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 1} | [0, 0]; 0 GPU-h; CPU work only |
| A4-direct-s101-m5-score-prepare | 4a | A4-direct-s101-m5-export | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 0} | [0, 0]; 0 GPU-h; CPU work only |
| A4-direct-s101-m5-score | 4a | A4-direct-s101-m5-score-prepare | afterok | 1 | 1 | 8 | yes | {'dispatch': 1, 'score': 7, 'nll': 2, 'trainer': 0, 'export': 0} / K=3 | None; 1 × allocation wall hours, including load, failures and idle; vLLM K=3; HF NLL K=1 |
| A4-direct-s101-m5-finish | 4a | A4-direct-s101-m5-score | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 0} | [0, 0]; 0 GPU-h; CPU work only |
| A4-direct-s102-m5-prepare | 4a | A4-ledger-m5 | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 0} | [0, 0]; 0 GPU-h; CPU work only |
| A4-direct-s102-m5-train | 4a | A4-direct-s102-m5-prepare | afterok | 8 | 1 | 32 | no | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 1, 'export': 0} | None; 8 × measured training wall hours; limit 1.5 × bound W stage GPU-h |
| A4-direct-s102-m5-export | 4a | A4-direct-s102-m5-train | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 1} | [0, 0]; 0 GPU-h; CPU work only |
| A4-direct-s102-m5-score-prepare | 4a | A4-direct-s102-m5-export | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 0} | [0, 0]; 0 GPU-h; CPU work only |
| A4-direct-s102-m5-score | 4a | A4-direct-s102-m5-score-prepare | afterok | 1 | 1 | 8 | yes | {'dispatch': 1, 'score': 7, 'nll': 2, 'trainer': 0, 'export': 0} / K=3 | None; 1 × allocation wall hours, including load, failures and idle; vLLM K=3; HF NLL K=1 |
| A4-direct-s102-m5-finish | 4a | A4-direct-s102-m5-score | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 0} | [0, 0]; 0 GPU-h; CPU work only |
| A4-direct-s103-m5-prepare | 4a | A4-ledger-m5 | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 0} | [0, 0]; 0 GPU-h; CPU work only |
| A4-direct-s103-m5-train | 4a | A4-direct-s103-m5-prepare | afterok | 8 | 1 | 32 | no | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 1, 'export': 0} | None; 8 × measured training wall hours; limit 1.5 × bound W stage GPU-h |
| A4-direct-s103-m5-export | 4a | A4-direct-s103-m5-train | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 1} | [0, 0]; 0 GPU-h; CPU work only |
| A4-direct-s103-m5-score-prepare | 4a | A4-direct-s103-m5-export | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 0} | [0, 0]; 0 GPU-h; CPU work only |
| A4-direct-s103-m5-score | 4a | A4-direct-s103-m5-score-prepare | afterok | 1 | 1 | 8 | yes | {'dispatch': 1, 'score': 7, 'nll': 2, 'trainer': 0, 'export': 0} / K=3 | None; 1 × allocation wall hours, including load, failures and idle; vLLM K=3; HF NLL K=1 |
| A4-direct-s103-m5-finish | 4a | A4-direct-s103-m5-score | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 0} | [0, 0]; 0 GPU-h; CPU work only |
| A4-month-m5 | 4a | A4-ledger-m5, A4-direct-s101-m5-finish, A4-direct-s102-m5-finish, A4-direct-s103-m5-finish | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 0} | [0, 0]; 0 GPU-h; CPU work only |
| A4-context-m6 | 4a | A4-month-m5 | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 0} | [0, 0]; 0 GPU-h; CPU work only |
| A4-arrival-prepare-m6 | 4a | A4-context-m6 | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 0} | [0, 0]; 0 GPU-h; CPU work only |
| A4-arrival-m6 | 4a | A4-arrival-prepare-m6 | afterok | 1 | 1 | 8 | yes | {'dispatch': 1, 'score': 11, 'nll': 0, 'trainer': 0, 'export': 0} / K=3 | None; 1 × allocation wall hours, including load, failures and idle; vLLM K=3; HF NLL K=1 |
| A4-ledger-m6 | 4a | A4-arrival-m6 | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 0} | [0, 0]; 0 GPU-h; CPU work only |
| A4-direct-s101-m6-prepare | 4a | A4-ledger-m6 | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 0} | [0, 0]; 0 GPU-h; CPU work only |
| A4-direct-s101-m6-train | 4a | A4-direct-s101-m6-prepare | afterok | 8 | 1 | 32 | no | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 1, 'export': 0} | None; 8 × measured training wall hours; limit 1.5 × bound W stage GPU-h |
| A4-direct-s101-m6-export | 4a | A4-direct-s101-m6-train | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 1} | [0, 0]; 0 GPU-h; CPU work only |
| A4-direct-s101-m6-score-prepare | 4a | A4-direct-s101-m6-export | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 0} | [0, 0]; 0 GPU-h; CPU work only |
| A4-direct-s101-m6-score | 4a | A4-direct-s101-m6-score-prepare | afterok | 1 | 1 | 8 | yes | {'dispatch': 1, 'score': 8, 'nll': 2, 'trainer': 0, 'export': 0} / K=3 | None; 1 × allocation wall hours, including load, failures and idle; vLLM K=3; HF NLL K=1 |
| A4-direct-s101-m6-finish | 4a | A4-direct-s101-m6-score | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 0} | [0, 0]; 0 GPU-h; CPU work only |
| A4-direct-s102-m6-prepare | 4a | A4-ledger-m6 | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 0} | [0, 0]; 0 GPU-h; CPU work only |
| A4-direct-s102-m6-train | 4a | A4-direct-s102-m6-prepare | afterok | 8 | 1 | 32 | no | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 1, 'export': 0} | None; 8 × measured training wall hours; limit 1.5 × bound W stage GPU-h |
| A4-direct-s102-m6-export | 4a | A4-direct-s102-m6-train | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 1} | [0, 0]; 0 GPU-h; CPU work only |
| A4-direct-s102-m6-score-prepare | 4a | A4-direct-s102-m6-export | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 0} | [0, 0]; 0 GPU-h; CPU work only |
| A4-direct-s102-m6-score | 4a | A4-direct-s102-m6-score-prepare | afterok | 1 | 1 | 8 | yes | {'dispatch': 1, 'score': 8, 'nll': 2, 'trainer': 0, 'export': 0} / K=3 | None; 1 × allocation wall hours, including load, failures and idle; vLLM K=3; HF NLL K=1 |
| A4-direct-s102-m6-finish | 4a | A4-direct-s102-m6-score | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 0} | [0, 0]; 0 GPU-h; CPU work only |
| A4-direct-s103-m6-prepare | 4a | A4-ledger-m6 | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 0} | [0, 0]; 0 GPU-h; CPU work only |
| A4-direct-s103-m6-train | 4a | A4-direct-s103-m6-prepare | afterok | 8 | 1 | 32 | no | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 1, 'export': 0} | None; 8 × measured training wall hours; limit 1.5 × bound W stage GPU-h |
| A4-direct-s103-m6-export | 4a | A4-direct-s103-m6-train | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 1} | [0, 0]; 0 GPU-h; CPU work only |
| A4-direct-s103-m6-score-prepare | 4a | A4-direct-s103-m6-export | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 0} | [0, 0]; 0 GPU-h; CPU work only |
| A4-direct-s103-m6-score | 4a | A4-direct-s103-m6-score-prepare | afterok | 1 | 1 | 8 | yes | {'dispatch': 1, 'score': 8, 'nll': 2, 'trainer': 0, 'export': 0} / K=3 | None; 1 × allocation wall hours, including load, failures and idle; vLLM K=3; HF NLL K=1 |
| A4-direct-s103-m6-finish | 4a | A4-direct-s103-m6-score | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 0} | [0, 0]; 0 GPU-h; CPU work only |
| A4-consolidation-s101-m6-prepare | 4a | A4-ledger-m6 | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 0} | [0, 0]; 0 GPU-h; CPU work only |
| A4-consolidation-s101-m6-train | 4a | A4-consolidation-s101-m6-prepare | afterok | 8 | 1 | 32 | no | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 1, 'export': 0} | None; 8 × measured training wall hours; limit 1.5 × bound W stage GPU-h |
| A4-consolidation-s101-m6-export | 4a | A4-consolidation-s101-m6-train | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 1} | [0, 0]; 0 GPU-h; CPU work only |
| A4-consolidation-s101-m6-score-prepare | 4a | A4-consolidation-s101-m6-export | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 0} | [0, 0]; 0 GPU-h; CPU work only |
| A4-consolidation-s101-m6-score | 4a | A4-consolidation-s101-m6-score-prepare | afterok | 1 | 1 | 8 | yes | {'dispatch': 1, 'score': 14, 'nll': 2, 'trainer': 0, 'export': 0} / K=3 | None; 1 × allocation wall hours, including load, failures and idle; vLLM K=3; HF NLL K=1 |
| A4-consolidation-s101-m6-finish | 4a | A4-consolidation-s101-m6-score | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 0} | [0, 0]; 0 GPU-h; CPU work only |
| A4-consolidation-s102-m6-prepare | 4a | A4-ledger-m6 | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 0} | [0, 0]; 0 GPU-h; CPU work only |
| A4-consolidation-s102-m6-train | 4a | A4-consolidation-s102-m6-prepare | afterok | 8 | 1 | 32 | no | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 1, 'export': 0} | None; 8 × measured training wall hours; limit 1.5 × bound W stage GPU-h |
| A4-consolidation-s102-m6-export | 4a | A4-consolidation-s102-m6-train | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 1} | [0, 0]; 0 GPU-h; CPU work only |
| A4-consolidation-s102-m6-score-prepare | 4a | A4-consolidation-s102-m6-export | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 0} | [0, 0]; 0 GPU-h; CPU work only |
| A4-consolidation-s102-m6-score | 4a | A4-consolidation-s102-m6-score-prepare | afterok | 1 | 1 | 8 | yes | {'dispatch': 1, 'score': 14, 'nll': 2, 'trainer': 0, 'export': 0} / K=3 | None; 1 × allocation wall hours, including load, failures and idle; vLLM K=3; HF NLL K=1 |
| A4-consolidation-s102-m6-finish | 4a | A4-consolidation-s102-m6-score | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 0} | [0, 0]; 0 GPU-h; CPU work only |
| A4-consolidation-s103-m6-prepare | 4a | A4-ledger-m6 | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 0} | [0, 0]; 0 GPU-h; CPU work only |
| A4-consolidation-s103-m6-train | 4a | A4-consolidation-s103-m6-prepare | afterok | 8 | 1 | 32 | no | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 1, 'export': 0} | None; 8 × measured training wall hours; limit 1.5 × bound W stage GPU-h |
| A4-consolidation-s103-m6-export | 4a | A4-consolidation-s103-m6-train | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 1} | [0, 0]; 0 GPU-h; CPU work only |
| A4-consolidation-s103-m6-score-prepare | 4a | A4-consolidation-s103-m6-export | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 0} | [0, 0]; 0 GPU-h; CPU work only |
| A4-consolidation-s103-m6-score | 4a | A4-consolidation-s103-m6-score-prepare | afterok | 1 | 1 | 8 | yes | {'dispatch': 1, 'score': 14, 'nll': 2, 'trainer': 0, 'export': 0} / K=3 | None; 1 × allocation wall hours, including load, failures and idle; vLLM K=3; HF NLL K=1 |
| A4-consolidation-s103-m6-finish | 4a | A4-consolidation-s103-m6-score | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 0} | [0, 0]; 0 GPU-h; CPU work only |
| A4-month-m6 | 4a | A4-ledger-m6, A4-direct-s101-m6-finish, A4-direct-s102-m6-finish, A4-direct-s103-m6-finish, A4-consolidation-s101-m6-finish, A4-consolidation-s102-m6-finish, A4-consolidation-s103-m6-finish | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 0} | [0, 0]; 0 GPU-h; CPU work only |
| A4-comparison-prepare | 4a | A4-month-m6 | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 0} | [0, 0]; 0 GPU-h; CPU work only |
| A4-comparison | 4a | A4-comparison-prepare | afterok | 1 | 1 | 8 | yes | {'dispatch': 1, 'score': 418, 'nll': 54, 'trainer': 0, 'export': 0} / K=3 | None; 1 × allocation wall hours, including load, failures and idle; vLLM K=3; HF NLL K=1 |
| A4-read | 4a | A4-comparison | afterany | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 0} | [0, 0]; 0 GPU-h; CPU work only |
| A4-collect | 4a | A4-read | afterok | 0 | 1 | 8 | yes | {'dispatch': 1, 'score': 0, 'nll': 0, 'trainer': 0, 'export': 0} | [0, 0]; 0 GPU-h; CPU work only |
| REF-core-assign | reference | none | afterok | 0 | 1 | 8 | yes | CPU | [0, 0]; CPU reference assignment freeze |
| REF-core-101-prepare | reference | REF-core-assign, M-final-score-101 | afterok | 0 | 1 | 8 | yes | CPU | [0, 0]; CPU reference checkpoint verification |
| REF-core-101 | reference | REF-core-101-prepare | afterok | 1 | 1 | 8 | yes | 2 × decision checkpoints (both panels) / K=2 | None; both complete HF panel lanes on one allocated GPU; K=2 |
| REF-core-102-prepare | reference | REF-core-assign, M-final-score-102 | afterok | 0 | 1 | 8 | yes | CPU | [0, 0]; CPU reference checkpoint verification |
| REF-core-102 | reference | REF-core-102-prepare | afterok | 1 | 1 | 8 | yes | 2 × decision checkpoints (both panels) / K=2 | None; both complete HF panel lanes on one allocated GPU; K=2 |
| REF-core-103-prepare | reference | REF-core-assign, M-final-score-103 | afterok | 0 | 1 | 8 | yes | CPU | [0, 0]; CPU reference checkpoint verification |
| REF-core-103 | reference | REF-core-103-prepare | afterok | 1 | 1 | 8 | yes | 2 × decision checkpoints (both panels) / K=2 | None; both complete HF panel lanes on one allocated GPU; K=2 |
| REF-reprobe-assign | reference | none | afterok | 0 | 1 | 8 | yes | CPU | [0, 0]; CPU reference assignment freeze |
| REF-reprobe-101-prepare | reference | REF-reprobe-assign, R-compare-full-101 | afterok | 0 | 1 | 8 | yes | CPU | [0, 0]; CPU reference checkpoint verification |
| REF-reprobe-101 | reference | REF-reprobe-101-prepare | afterok | 1 | 1 | 8 | yes | 2 × decision checkpoints (both panels) / K=2 | None; both complete HF panel lanes on one allocated GPU; K=2 |
| REF-reprobe-102-prepare | reference | REF-reprobe-assign, R-compare-full-102 | afterok | 0 | 1 | 8 | yes | CPU | [0, 0]; CPU reference checkpoint verification |
| REF-reprobe-102 | reference | REF-reprobe-102-prepare | afterok | 1 | 1 | 8 | yes | 2 × decision checkpoints (both panels) / K=2 | None; both complete HF panel lanes on one allocated GPU; K=2 |
| REF-reprobe-103-prepare | reference | REF-reprobe-assign, R-compare-full-103 | afterok | 0 | 1 | 8 | yes | CPU | [0, 0]; CPU reference checkpoint verification |
| REF-reprobe-103 | reference | REF-reprobe-103-prepare | afterok | 1 | 1 | 8 | yes | 2 × decision checkpoints (both panels) / K=2 | None; both complete HF panel lanes on one allocated GPU; K=2 |
| REF-m3-assign | reference | none | afterok | 0 | 1 | 8 | yes | CPU | [0, 0]; CPU reference assignment freeze |
| REF-m3-101-prepare | reference | REF-m3-assign, M3-C0-101-finqa-export, M3-K-101-finqa-export, M3-G-101-finqa-export, M3-CR-101-finqa-export, M3-C0-101-contract_nli-export, M3-K-101-contract_nli-export, M3-G-101-contract_nli-export, M3-CR-101-contract_nli-export, M3-C0-101-math_hard-export, M3-K-101-math_hard-export, M3-G-101-math_hard-export, M3-CR-101-math_hard-export | afterok | 0 | 1 | 8 | yes | CPU | [0, 0]; CPU reference checkpoint verification |
| REF-m3-101 | reference | REF-m3-101-prepare | afterok | 1 | 1 | 8 | yes | 2 × decision checkpoints (both panels) / K=2 | None; both complete HF panel lanes on one allocated GPU; K=2 |
| REF-m3-102-prepare | reference | REF-m3-assign, M3-C0-102-finqa-export, M3-K-102-finqa-export, M3-G-102-finqa-export, M3-CR-102-finqa-export, M3-C0-102-contract_nli-export, M3-K-102-contract_nli-export, M3-G-102-contract_nli-export, M3-CR-102-contract_nli-export, M3-C0-102-math_hard-export, M3-K-102-math_hard-export, M3-G-102-math_hard-export, M3-CR-102-math_hard-export | afterok | 0 | 1 | 8 | yes | CPU | [0, 0]; CPU reference checkpoint verification |
| REF-m3-102 | reference | REF-m3-102-prepare | afterok | 1 | 1 | 8 | yes | 2 × decision checkpoints (both panels) / K=2 | None; both complete HF panel lanes on one allocated GPU; K=2 |
| REF-m3-103-prepare | reference | REF-m3-assign, M3-C0-103-finqa-export, M3-K-103-finqa-export, M3-G-103-finqa-export, M3-CR-103-finqa-export, M3-C0-103-contract_nli-export, M3-K-103-contract_nli-export, M3-G-103-contract_nli-export, M3-CR-103-contract_nli-export, M3-C0-103-math_hard-export, M3-K-103-math_hard-export, M3-G-103-math_hard-export, M3-CR-103-math_hard-export | afterok | 0 | 1 | 8 | yes | CPU | [0, 0]; CPU reference checkpoint verification |
| REF-m3-103 | reference | REF-m3-103-prepare | afterok | 1 | 1 | 8 | yes | 2 × decision checkpoints (both panels) / K=2 | None; both complete HF panel lanes on one allocated GPU; K=2 |
| REF-x-tier1-assign | reference | none | afterok | 0 | 1 | 8 | yes | CPU | [0, 0]; CPU reference assignment freeze |
| REF-x-tier1-101-prepare | reference | REF-x-tier1-assign, X-complete-101 | afterok | 0 | 1 | 8 | yes | CPU | [0, 0]; CPU reference checkpoint verification |
| REF-x-tier1-101 | reference | REF-x-tier1-101-prepare | afterok | 1 | 1 | 8 | yes | 2 × decision checkpoints (both panels) / K=2 | None; both complete HF panel lanes on one allocated GPU; K=2 |
| REF-x-tier1-102-prepare | reference | REF-x-tier1-assign, X-complete-102 | afterok | 0 | 1 | 8 | yes | CPU | [0, 0]; CPU reference checkpoint verification |
| REF-x-tier1-102 | reference | REF-x-tier1-102-prepare | afterok | 1 | 1 | 8 | yes | 2 × decision checkpoints (both panels) / K=2 | None; both complete HF panel lanes on one allocated GPU; K=2 |
| REF-x-tier1-103-prepare | reference | REF-x-tier1-assign, X-complete-103 | afterok | 0 | 1 | 8 | yes | CPU | [0, 0]; CPU reference checkpoint verification |
| REF-x-tier1-103 | reference | REF-x-tier1-103-prepare | afterok | 1 | 1 | 8 | yes | 2 × decision checkpoints (both panels) / K=2 | None; both complete HF panel lanes on one allocated GPU; K=2 |
| REF-x-tier2.b-assign | reference | none | afterok | 0 | 1 | 8 | yes | CPU | [0, 0]; CPU reference assignment freeze |
| REF-x-tier2.b-101-prepare | reference | REF-x-tier2.b-assign, X17-complete-101 | afterok | 0 | 1 | 8 | yes | CPU | [0, 0]; CPU reference checkpoint verification |
| REF-x-tier2.b-101 | reference | REF-x-tier2.b-101-prepare | afterok | 1 | 1 | 8 | yes | 2 × decision checkpoints (both panels) / K=2 | None; both complete HF panel lanes on one allocated GPU; K=2 |
| REF-x-tier2.b-102-prepare | reference | REF-x-tier2.b-assign, X17-complete-102 | afterok | 0 | 1 | 8 | yes | CPU | [0, 0]; CPU reference checkpoint verification |
| REF-x-tier2.b-102 | reference | REF-x-tier2.b-102-prepare | afterok | 1 | 1 | 8 | yes | 2 × decision checkpoints (both panels) / K=2 | None; both complete HF panel lanes on one allocated GPU; K=2 |
| REF-x-tier2.b-103-prepare | reference | REF-x-tier2.b-assign, X17-complete-103 | afterok | 0 | 1 | 8 | yes | CPU | [0, 0]; CPU reference checkpoint verification |
| REF-x-tier2.b-103 | reference | REF-x-tier2.b-103-prepare | afterok | 1 | 1 | 8 | yes | 2 × decision checkpoints (both panels) / K=2 | None; both complete HF panel lanes on one allocated GPU; K=2 |
| REF-x-tier2.b-p2-assign | reference | none | afterok | 0 | 1 | 8 | yes | CPU | [0, 0]; CPU reference assignment freeze |
| REF-x-tier2.b-p2-101-prepare | reference | REF-x-tier2.b-p2-assign, X18-complete-101 | afterok | 0 | 1 | 8 | yes | CPU | [0, 0]; CPU reference checkpoint verification |
| REF-x-tier2.b-p2-101 | reference | REF-x-tier2.b-p2-101-prepare | afterok | 1 | 1 | 8 | yes | 2 × decision checkpoints (both panels) / K=2 | None; both complete HF panel lanes on one allocated GPU; K=2 |
| REF-x-tier2.b-p2-102-prepare | reference | REF-x-tier2.b-p2-assign, X18-complete-102 | afterok | 0 | 1 | 8 | yes | CPU | [0, 0]; CPU reference checkpoint verification |
| REF-x-tier2.b-p2-102 | reference | REF-x-tier2.b-p2-102-prepare | afterok | 1 | 1 | 8 | yes | 2 × decision checkpoints (both panels) / K=2 | None; both complete HF panel lanes on one allocated GPU; K=2 |
| REF-x-tier2.b-p2-103-prepare | reference | REF-x-tier2.b-p2-assign, X18-complete-103 | afterok | 0 | 1 | 8 | yes | CPU | [0, 0]; CPU reference checkpoint verification |
| REF-x-tier2.b-p2-103 | reference | REF-x-tier2.b-p2-103-prepare | afterok | 1 | 1 | 8 | yes | 2 × decision checkpoints (both panels) / K=2 | None; both complete HF panel lanes on one allocated GPU; K=2 |
| REF-x-tier2.incremental-assign | reference | none | afterok | 0 | 1 | 8 | yes | CPU | [0, 0]; CPU reference assignment freeze |
| REF-x-tier2.incremental-101-prepare | reference | REF-x-tier2.incremental-assign, X19-complete-101 | afterok | 0 | 1 | 8 | yes | CPU | [0, 0]; CPU reference checkpoint verification |
| REF-x-tier2.incremental-101 | reference | REF-x-tier2.incremental-101-prepare | afterok | 1 | 1 | 8 | yes | 2 × decision checkpoints (both panels) / K=2 | None; both complete HF panel lanes on one allocated GPU; K=2 |
| REF-x-tier2.incremental-102-prepare | reference | REF-x-tier2.incremental-assign, X19-complete-102 | afterok | 0 | 1 | 8 | yes | CPU | [0, 0]; CPU reference checkpoint verification |
| REF-x-tier2.incremental-102 | reference | REF-x-tier2.incremental-102-prepare | afterok | 1 | 1 | 8 | yes | 2 × decision checkpoints (both panels) / K=2 | None; both complete HF panel lanes on one allocated GPU; K=2 |
| REF-x-tier2.incremental-103-prepare | reference | REF-x-tier2.incremental-assign, X19-complete-103 | afterok | 0 | 1 | 8 | yes | CPU | [0, 0]; CPU reference checkpoint verification |
| REF-x-tier2.incremental-103 | reference | REF-x-tier2.incremental-103-prepare | afterok | 1 | 1 | 8 | yes | 2 × decision checkpoints (both panels) / K=2 | None; both complete HF panel lanes on one allocated GPU; K=2 |
| REF-x-tier2.incremental-p2-assign | reference | none | afterok | 0 | 1 | 8 | yes | CPU | [0, 0]; CPU reference assignment freeze |
| REF-x-tier2.incremental-p2-101-prepare | reference | REF-x-tier2.incremental-p2-assign, X20-complete-101 | afterok | 0 | 1 | 8 | yes | CPU | [0, 0]; CPU reference checkpoint verification |
| REF-x-tier2.incremental-p2-101 | reference | REF-x-tier2.incremental-p2-101-prepare | afterok | 1 | 1 | 8 | yes | 2 × decision checkpoints (both panels) / K=2 | None; both complete HF panel lanes on one allocated GPU; K=2 |
| REF-x-tier2.incremental-p2-102-prepare | reference | REF-x-tier2.incremental-p2-assign, X20-complete-102 | afterok | 0 | 1 | 8 | yes | CPU | [0, 0]; CPU reference checkpoint verification |
| REF-x-tier2.incremental-p2-102 | reference | REF-x-tier2.incremental-p2-102-prepare | afterok | 1 | 1 | 8 | yes | 2 × decision checkpoints (both panels) / K=2 | None; both complete HF panel lanes on one allocated GPU; K=2 |
| REF-x-tier2.incremental-p2-103-prepare | reference | REF-x-tier2.incremental-p2-assign, X20-complete-103 | afterok | 0 | 1 | 8 | yes | CPU | [0, 0]; CPU reference checkpoint verification |
| REF-x-tier2.incremental-p2-103 | reference | REF-x-tier2.incremental-p2-103-prepare | afterok | 1 | 1 | 8 | yes | 2 × decision checkpoints (both panels) / K=2 | None; both complete HF panel lanes on one allocated GPU; K=2 |
| REF-x-tier2.stack-drift-assign | reference | none | afterok | 0 | 1 | 8 | yes | CPU | [0, 0]; CPU reference assignment freeze |
| REF-x-tier2.stack-drift-101-prepare | reference | REF-x-tier2.stack-drift-assign, X15-complete-101 | afterok | 0 | 1 | 8 | yes | CPU | [0, 0]; CPU reference checkpoint verification |
| REF-x-tier2.stack-drift-101 | reference | REF-x-tier2.stack-drift-101-prepare | afterok | 1 | 1 | 8 | yes | 2 × decision checkpoints (both panels) / K=2 | None; both complete HF panel lanes on one allocated GPU; K=2 |
| REF-x-tier2.stack-drift-102-prepare | reference | REF-x-tier2.stack-drift-assign, X15-complete-102 | afterok | 0 | 1 | 8 | yes | CPU | [0, 0]; CPU reference checkpoint verification |
| REF-x-tier2.stack-drift-102 | reference | REF-x-tier2.stack-drift-102-prepare | afterok | 1 | 1 | 8 | yes | 2 × decision checkpoints (both panels) / K=2 | None; both complete HF panel lanes on one allocated GPU; K=2 |
| REF-x-tier2.stack-drift-103-prepare | reference | REF-x-tier2.stack-drift-assign, X15-complete-103 | afterok | 0 | 1 | 8 | yes | CPU | [0, 0]; CPU reference checkpoint verification |
| REF-x-tier2.stack-drift-103 | reference | REF-x-tier2.stack-drift-103-prepare | afterok | 1 | 1 | 8 | yes | 2 × decision checkpoints (both panels) / K=2 | None; both complete HF panel lanes on one allocated GPU; K=2 |
| REF-x-tier2.stack-plasticity-assign | reference | none | afterok | 0 | 1 | 8 | yes | CPU | [0, 0]; CPU reference assignment freeze |
| REF-x-tier2.stack-plasticity-101-prepare | reference | REF-x-tier2.stack-plasticity-assign, X16-complete-101 | afterok | 0 | 1 | 8 | yes | CPU | [0, 0]; CPU reference checkpoint verification |
| REF-x-tier2.stack-plasticity-101 | reference | REF-x-tier2.stack-plasticity-101-prepare | afterok | 1 | 1 | 8 | yes | 2 × decision checkpoints (both panels) / K=2 | None; both complete HF panel lanes on one allocated GPU; K=2 |
| REF-x-tier2.stack-plasticity-102-prepare | reference | REF-x-tier2.stack-plasticity-assign, X16-complete-102 | afterok | 0 | 1 | 8 | yes | CPU | [0, 0]; CPU reference checkpoint verification |
| REF-x-tier2.stack-plasticity-102 | reference | REF-x-tier2.stack-plasticity-102-prepare | afterok | 1 | 1 | 8 | yes | 2 × decision checkpoints (both panels) / K=2 | None; both complete HF panel lanes on one allocated GPU; K=2 |
| REF-x-tier2.stack-plasticity-103-prepare | reference | REF-x-tier2.stack-plasticity-assign, X16-complete-103 | afterok | 0 | 1 | 8 | yes | CPU | [0, 0]; CPU reference checkpoint verification |
| REF-x-tier2.stack-plasticity-103 | reference | REF-x-tier2.stack-plasticity-103-prepare | afterok | 1 | 1 | 8 | yes | 2 × decision checkpoints (both panels) / K=2 | None; both complete HF panel lanes on one allocated GPU; K=2 |
| REPORT-text | report | REF-core-101, REF-core-102, REF-core-103, REF-reprobe-101, REF-reprobe-102, REF-reprobe-103, REF-m3-101, REF-m3-102, REF-m3-103, REF-x-tier1-101, REF-x-tier1-102, REF-x-tier1-103, REF-x-tier2.b-101, REF-x-tier2.b-102, REF-x-tier2.b-103, REF-x-tier2.b-p2-101, REF-x-tier2.b-p2-102, REF-x-tier2.b-p2-103, REF-x-tier2.incremental-101, REF-x-tier2.incremental-102, REF-x-tier2.incremental-103, REF-x-tier2.incremental-p2-101, REF-x-tier2.incremental-p2-102, REF-x-tier2.incremental-p2-103, REF-x-tier2.stack-drift-101, REF-x-tier2.stack-drift-102, REF-x-tier2.stack-drift-103, REF-x-tier2.stack-plasticity-101, REF-x-tier2.stack-plasticity-102, REF-x-tier2.stack-plasticity-103, RA-collect, S12, R-collect, M-core-collect, X12, X23, M3-read, SP-read, PD-read, A4-read, A4-collect | afterany | 0 | 1 | 8 | yes | CPU | [0, 0]; CPU reporter and reference guards |

Component envelopes (overlap per-job components; not additive):

- Core M: 109.6 floor includes 46.4 MATH +10 targets +30 scoring/export +23.2 legal; +23.2 per confirmation set; fresh 12-stage re-probe +92.8 proxy; independent 31-call rescore/overhead unpriced
- Screen S: 30–50 envelope; v2 scoring/transfer measurement pending
- X tier 1: 12 x 7.733333 = 92.8 training; probes/generation/scoring/X4/startup additional
- X tier 2: 36 x 7.733333 = 278.4 training; 295.2–316.8 historical envelope; v2 overheads remain measured separately
- K/CR: [786.297517, 1387.157525]
- PD: [62.211758, 97.382143]
- 4A: [400, 450]; planning proxy; separate approved ceiling 800 GPU-h; all allocations and failed attempts charged
- SP: [54.75, 216.0]; entry sum 53.75–215 plus 1 GPU-h startup =54.75–216; SP(a)=0

Approved tier allowance: 1,800 GPU-h; component limits K/CR <=1,400, PD <=264, SP <=216. The revised upper case is about 1,700.54 GPU-h before unpriced rescoring; a PD whole-seed restart adds about 20.74–32.46 GPU-h plus CPU/queue cost. These are warnings monitored in partner receipts, never launch/scientific gates; no cells are dropped. Core is separate. Queue/restore/transfer time and failed attempts must be charged separately.
