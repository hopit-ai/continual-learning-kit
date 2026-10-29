# Combined launch inventory

Message-3 joins only on Opus SEND by 2026-11-02; otherwise send core alone. No launch authority.

Generated from every source task. Runtime locations stay in the local JSON; this table contains portable IDs only.

| Job | Group | Prerequisites | Dependency | GPU-h estimate / basis |
|---|---|---|---|---|
| PF-node-0 | core | none | first wave | None; core measured scoring/preparation/qualification allocation pending |
| PF-ARC | core | INPUTS | first wave | [0, 0]; CPU/handoff; no GPU compute |
| PF-tier-0 | preflight | none | first wave | None; whole-node dependency/cgroup/UUID check; measure allocation time |
| INPUTS | core | none | afterok | [0, 0]; CPU/handoff; no GPU compute |
| M2 | core | INPUTS | afterok | [0, 0]; CPU/handoff; no GPU compute |
| S1 | core | INPUTS | afterok | [0, 0]; CPU/handoff; no GPU compute |
| M3b | core | INPUTS, PF-ARC | afterok | None; core measured scoring/preparation/qualification allocation pending |
| M3 | core | M2 | afterok | [0, 0]; CPU/handoff; no GPU compute |
| RA0 | core | M2 | afterok | [0, 0]; CPU/handoff; no GPU compute |
| S2 | core | S1, M2 | afterok | [0, 0]; CPU/handoff; no GPU compute |
| M-pools | core | M3, M3b | afterok | [0, 0]; CPU/handoff; no GPU compute |
| RA101 | core | RA0 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| RA102 | core | RA0 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| RA103 | core | RA0 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| M5 | core | M3 | afterok | [10.0, 10.0]; core 500-item bounded 27B target allowance; charge once |
| S3 | core | S2 | afterok | [0, 0]; CPU/handoff; no GPU compute |
| S5 | core | S2 | afterok | [0, 0]; CPU/handoff; no GPU compute |
| S4-D-finqa-alone-s101 | core | S2 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| S4-S-finqa-alone-s101 | core | S2 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| S4-F-finqa-alone-s101 | core | S2 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| S4-R-finqa-alone-s101 | core | S2 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| S4-S-finqa-alone-s102 | core | S2 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| S4-F-finqa-alone-s102 | core | S2 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| S4-R-finqa-alone-s102 | core | S2 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| S4-D-finqa-alone-s102 | core | S2 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| S4-S-finqa-alone-s103 | core | S2 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| S4-F-finqa-alone-s103 | core | S2 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| S4-R-finqa-alone-s103 | core | S2 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| S4-D-finqa-alone-s103 | core | S2 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| S4-D-R-finqa-alone-s101 | core | S2 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| S4-D-R-finqa-alone-s102 | core | S2 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| S4-D-R-finqa-alone-s103 | core | S2 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| FV-core-101 | core | M3 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| FV-core-102 | core | M3 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| FV-core-103 | core | M3 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| FV-reprobe-101 | core | M3 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| FV-reprobe-102 | core | M3 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| FV-reprobe-103 | core | M3 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| FV-x-tier1-101 | core | M3 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| FV-x-tier1-102 | core | M3 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| FV-x-tier1-103 | core | M3 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| FV-x-tier2.b-101 | core | M3 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| FV-x-tier2.b-102 | core | M3 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| FV-x-tier2.b-103 | core | M3 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| FV-x-tier2.b-p2-101 | core | M3 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| FV-x-tier2.b-p2-102 | core | M3 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| FV-x-tier2.b-p2-103 | core | M3 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| FV-x-tier2.incremental-101 | core | M3 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| FV-x-tier2.incremental-102 | core | M3 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| FV-x-tier2.incremental-103 | core | M3 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| FV-x-tier2.incremental-p2-101 | core | M3 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| FV-x-tier2.incremental-p2-102 | core | M3 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| FV-x-tier2.incremental-p2-103 | core | M3 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| FV-x-tier2.stack-drift-101 | core | M3 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| FV-x-tier2.stack-drift-102 | core | M3 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| FV-x-tier2.stack-drift-103 | core | M3 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| FV-x-tier2.stack-plasticity-101 | core | M3 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| FV-x-tier2.stack-plasticity-102 | core | M3 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| FV-x-tier2.stack-plasticity-103 | core | M3 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| R0 | core | M-pools | afterok | [0, 0]; CPU/handoff; no GPU compute |
| RA-collect | core | RA101, RA102, RA103 | afterany | [0, 0]; CPU/handoff; no GPU compute |
| M4-101 | core | M3, FV-core-101 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| M4-102 | core | M3, FV-core-102 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| M4-103 | core | M3, FV-core-103 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| S6 | core | S5 | afterok | [0, 0]; CPU/handoff; no GPU compute |
| R-incoming-101 | core | R0, FV-reprobe-101 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| R-incoming-102 | core | R0, FV-reprobe-102 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| R-incoming-103 | core | R0, FV-reprobe-103 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| M6-101-incoming | core | M4-101, M5, FV-core-101 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| M6-102-incoming | core | M4-102, M5, FV-core-102 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| M6-103-incoming | core | M4-103, M5, FV-core-103 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| M7-101-incoming | core | M-pools, M4-101, FV-core-101 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| M7-102-incoming | core | M-pools, M4-102, FV-core-102 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| M7-103-incoming | core | M-pools, M4-103, FV-core-103 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| S7-base | core | S6 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| S7-t1-D-s101-40 | core | S6 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| S7-t1-D-s102-40 | core | S6 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| S7-t1-D-s103-40 | core | S6 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| S7-t1-F-s101-40 | core | S6 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| S7-t1-F-s102-40 | core | S6 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| S7-t1-F-s103-40 | core | S6 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| R-generic-101 | core | R-incoming-101 | afterok | [7.733333333333333, 7.733333333333333]; core 58 min x 8 GPU proxy/full stage |
| R-broad-101 | core | R-incoming-101 | afterok | [7.733333333333333, 7.733333333333333]; core 58 min x 8 GPU proxy/full stage |
| R-earlier-101 | core | R-incoming-101 | afterok | [7.733333333333333, 7.733333333333333]; core 58 min x 8 GPU proxy/full stage |
| R-both-101 | core | R-incoming-101 | afterok | [7.733333333333333, 7.733333333333333]; core 58 min x 8 GPU proxy/full stage |
| R-generic-102 | core | R-incoming-102 | afterok | [7.733333333333333, 7.733333333333333]; core 58 min x 8 GPU proxy/full stage |
| R-broad-102 | core | R-incoming-102 | afterok | [7.733333333333333, 7.733333333333333]; core 58 min x 8 GPU proxy/full stage |
| R-earlier-102 | core | R-incoming-102 | afterok | [7.733333333333333, 7.733333333333333]; core 58 min x 8 GPU proxy/full stage |
| R-both-102 | core | R-incoming-102 | afterok | [7.733333333333333, 7.733333333333333]; core 58 min x 8 GPU proxy/full stage |
| R-generic-103 | core | R-incoming-103 | afterok | [7.733333333333333, 7.733333333333333]; core 58 min x 8 GPU proxy/full stage |
| R-broad-103 | core | R-incoming-103 | afterok | [7.733333333333333, 7.733333333333333]; core 58 min x 8 GPU proxy/full stage |
| R-earlier-103 | core | R-incoming-103 | afterok | [7.733333333333333, 7.733333333333333]; core 58 min x 8 GPU proxy/full stage |
| R-both-103 | core | R-incoming-103 | afterok | [7.733333333333333, 7.733333333333333]; core 58 min x 8 GPU proxy/full stage |
| M6-101-train | core | M6-101-incoming | afterok | [7.733333333333333, 7.733333333333333]; core 58 min x 8 GPU proxy/full stage |
| M6-102-train | core | M6-102-incoming | afterok | [7.733333333333333, 7.733333333333333]; core 58 min x 8 GPU proxy/full stage |
| M6-103-train | core | M6-103-incoming | afterok | [7.733333333333333, 7.733333333333333]; core 58 min x 8 GPU proxy/full stage |
| M7-101-train | core | M7-101-incoming | afterok | [7.733333333333333, 7.733333333333333]; core 58 min x 8 GPU proxy/full stage |
| M7-102-train | core | M7-102-incoming | afterok | [7.733333333333333, 7.733333333333333]; core 58 min x 8 GPU proxy/full stage |
| M7-103-train | core | M7-103-incoming | afterok | [7.733333333333333, 7.733333333333333]; core 58 min x 8 GPU proxy/full stage |
| S10-base-ifeval-train | core | S7-base | afterok | [0.75, 1.4]; core Screen S four-update probe; included in S envelope |
| S10-base-gsm8k-train | core | S7-base | afterok | [0.75, 1.4]; core Screen S four-update probe; included in S envelope |
| S10-base-arc-train | core | S7-base | afterok | [0.75, 1.4]; core Screen S four-update probe; included in S envelope |
| S10-t1-D-s101-40-ifeval-train | core | S7-t1-D-s101-40 | afterok | [0.75, 1.4]; core Screen S four-update probe; included in S envelope |
| S10-t1-D-s101-40-gsm8k-train | core | S7-t1-D-s101-40 | afterok | [0.75, 1.4]; core Screen S four-update probe; included in S envelope |
| S10-t1-D-s101-40-arc-train | core | S7-t1-D-s101-40 | afterok | [0.75, 1.4]; core Screen S four-update probe; included in S envelope |
| S10-t1-D-s101-40-finqa-train | core | S7-t1-D-s101-40 | afterok | [0.75, 1.4]; core Screen S four-update probe; included in S envelope |
| S10-t1-D-s102-40-ifeval-train | core | S7-t1-D-s102-40 | afterok | [0.75, 1.4]; core Screen S four-update probe; included in S envelope |
| S10-t1-D-s102-40-gsm8k-train | core | S7-t1-D-s102-40 | afterok | [0.75, 1.4]; core Screen S four-update probe; included in S envelope |
| S10-t1-D-s102-40-arc-train | core | S7-t1-D-s102-40 | afterok | [0.75, 1.4]; core Screen S four-update probe; included in S envelope |
| S10-t1-D-s102-40-finqa-train | core | S7-t1-D-s102-40 | afterok | [0.75, 1.4]; core Screen S four-update probe; included in S envelope |
| S10-t1-D-s103-40-ifeval-train | core | S7-t1-D-s103-40 | afterok | [0.75, 1.4]; core Screen S four-update probe; included in S envelope |
| S10-t1-D-s103-40-gsm8k-train | core | S7-t1-D-s103-40 | afterok | [0.75, 1.4]; core Screen S four-update probe; included in S envelope |
| S10-t1-D-s103-40-arc-train | core | S7-t1-D-s103-40 | afterok | [0.75, 1.4]; core Screen S four-update probe; included in S envelope |
| S10-t1-D-s103-40-finqa-train | core | S7-t1-D-s103-40 | afterok | [0.75, 1.4]; core Screen S four-update probe; included in S envelope |
| S10-t1-F-s101-40-ifeval-train | core | S7-t1-F-s101-40 | afterok | [0.75, 1.4]; core Screen S four-update probe; included in S envelope |
| S10-t1-F-s101-40-gsm8k-train | core | S7-t1-F-s101-40 | afterok | [0.75, 1.4]; core Screen S four-update probe; included in S envelope |
| S10-t1-F-s101-40-arc-train | core | S7-t1-F-s101-40 | afterok | [0.75, 1.4]; core Screen S four-update probe; included in S envelope |
| S10-t1-F-s101-40-finqa-train | core | S7-t1-F-s101-40 | afterok | [0.75, 1.4]; core Screen S four-update probe; included in S envelope |
| S10-t1-F-s102-40-ifeval-train | core | S7-t1-F-s102-40 | afterok | [0.75, 1.4]; core Screen S four-update probe; included in S envelope |
| S10-t1-F-s102-40-gsm8k-train | core | S7-t1-F-s102-40 | afterok | [0.75, 1.4]; core Screen S four-update probe; included in S envelope |
| S10-t1-F-s102-40-arc-train | core | S7-t1-F-s102-40 | afterok | [0.75, 1.4]; core Screen S four-update probe; included in S envelope |
| S10-t1-F-s102-40-finqa-train | core | S7-t1-F-s102-40 | afterok | [0.75, 1.4]; core Screen S four-update probe; included in S envelope |
| S10-t1-F-s103-40-ifeval-train | core | S7-t1-F-s103-40 | afterok | [0.75, 1.4]; core Screen S four-update probe; included in S envelope |
| S10-t1-F-s103-40-gsm8k-train | core | S7-t1-F-s103-40 | afterok | [0.75, 1.4]; core Screen S four-update probe; included in S envelope |
| S10-t1-F-s103-40-arc-train | core | S7-t1-F-s103-40 | afterok | [0.75, 1.4]; core Screen S four-update probe; included in S envelope |
| S10-t1-F-s103-40-finqa-train | core | S7-t1-F-s103-40 | afterok | [0.75, 1.4]; core Screen S four-update probe; included in S envelope |
| R-generic-101-score | core | R-generic-101, FV-reprobe-101 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| R-broad-101-score | core | R-broad-101, FV-reprobe-101 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| R-earlier-101-score | core | R-earlier-101, FV-reprobe-101 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| R-both-101-score | core | R-both-101, FV-reprobe-101 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| R-generic-102-score | core | R-generic-102, FV-reprobe-102 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| R-broad-102-score | core | R-broad-102, FV-reprobe-102 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| R-earlier-102-score | core | R-earlier-102, FV-reprobe-102 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| R-both-102-score | core | R-both-102, FV-reprobe-102 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| R-generic-103-score | core | R-generic-103, FV-reprobe-103 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| R-broad-103-score | core | R-broad-103, FV-reprobe-103 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| R-earlier-103-score | core | R-earlier-103, FV-reprobe-103 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| R-both-103-score | core | R-both-103, FV-reprobe-103 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| M6-101-score | core | M6-101-train, FV-core-101 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| M6-102-score | core | M6-102-train, FV-core-102 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| M6-103-score | core | M6-103-train, FV-core-103 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| M7-101-score | core | M7-101-train, FV-core-101 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| M7-102-score | core | M7-102-train, FV-core-102 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| M7-103-score | core | M7-103-train, FV-core-103 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| S8-S10-101 | core | S6, S7-base, S7-t1-D-s101-40, S7-t1-F-s101-40, S10-base-ifeval-train, S10-base-gsm8k-train, S10-base-arc-train, S10-t1-D-s101-40-ifeval-train, S10-t1-D-s101-40-gsm8k-train, S10-t1-D-s101-40-arc-train, S10-t1-D-s101-40-finqa-train, S10-t1-F-s101-40-ifeval-train, S10-t1-F-s101-40-gsm8k-train, S10-t1-F-s101-40-arc-train, S10-t1-F-s101-40-finqa-train | afterok | None; core measured scoring/preparation/qualification allocation pending |
| S8-S10-102 | core | S6, S7-base, S7-t1-D-s102-40, S7-t1-F-s102-40, S10-base-ifeval-train, S10-base-gsm8k-train, S10-base-arc-train, S10-t1-D-s102-40-ifeval-train, S10-t1-D-s102-40-gsm8k-train, S10-t1-D-s102-40-arc-train, S10-t1-D-s102-40-finqa-train, S10-t1-F-s102-40-ifeval-train, S10-t1-F-s102-40-gsm8k-train, S10-t1-F-s102-40-arc-train, S10-t1-F-s102-40-finqa-train | afterok | None; core measured scoring/preparation/qualification allocation pending |
| S8-S10-103 | core | S6, S7-base, S7-t1-D-s103-40, S7-t1-F-s103-40, S10-base-ifeval-train, S10-base-gsm8k-train, S10-base-arc-train, S10-t1-D-s103-40-ifeval-train, S10-t1-D-s103-40-gsm8k-train, S10-t1-D-s103-40-arc-train, S10-t1-D-s103-40-finqa-train, S10-t1-F-s103-40-ifeval-train, S10-t1-F-s103-40-gsm8k-train, S10-t1-F-s103-40-arc-train, S10-t1-F-s103-40-finqa-train | afterok | None; core measured scoring/preparation/qualification allocation pending |
| R-broad | core | R0, R-generic-101-score, R-broad-101-score, R-generic-102-score, R-broad-102-score, R-generic-103-score, R-broad-103-score | afterany | [0, 0]; CPU/handoff; no GPU compute |
| R-nominate | core | R0, R-generic-101-score, R-broad-101-score, R-earlier-101-score, R-both-101-score, R-generic-102-score, R-broad-102-score, R-earlier-102-score, R-both-102-score, R-generic-103-score, R-broad-103-score, R-earlier-103-score, R-both-103-score | afterany | [0, 0]; CPU/handoff; no GPU compute |
| S12 | core | S3, S4-D-finqa-alone-s101, S4-S-finqa-alone-s101, S4-F-finqa-alone-s101, S4-R-finqa-alone-s101, S4-S-finqa-alone-s102, S4-F-finqa-alone-s102, S4-R-finqa-alone-s102, S4-D-finqa-alone-s102, S4-S-finqa-alone-s103, S4-F-finqa-alone-s103, S4-R-finqa-alone-s103, S4-D-finqa-alone-s103, S4-D-R-finqa-alone-s101, S4-D-R-finqa-alone-s102, S4-D-R-finqa-alone-s103, S8-S10-101, S8-S10-102, S8-S10-103 | afterany | [0, 0]; CPU/handoff; no GPU compute |
| R-collect | core | R0, R-broad, R-nominate, R-generic-101-score, R-broad-101-score, R-earlier-101-score, R-both-101-score, R-generic-102-score, R-broad-102-score, R-earlier-102-score, R-both-102-score, R-generic-103-score, R-broad-103-score, R-earlier-103-score, R-both-103-score | afterany | [0, 0]; CPU/handoff; no GPU compute |
| M-nomination | core | M-pools, R-broad | afterok | [0, 0]; CPU/handoff; no GPU compute |
| M-final-nomination | core | M-pools, R-nominate | afterok | [0, 0]; CPU/handoff; no GPU compute |
| M8-101-incoming | core | M-nomination, M4-101, FV-core-101 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| M8-102-incoming | core | M-nomination, M4-102, FV-core-102 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| M8-103-incoming | core | M-nomination, M4-103, FV-core-103 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| M8-101-train | core | M8-101-incoming | afterok | [7.733333333333333, 7.733333333333333]; core 58 min x 8 GPU proxy/full stage |
| M8-102-train | core | M8-102-incoming | afterok | [7.733333333333333, 7.733333333333333]; core 58 min x 8 GPU proxy/full stage |
| M8-103-train | core | M8-103-incoming | afterok | [7.733333333333333, 7.733333333333333]; core 58 min x 8 GPU proxy/full stage |
| M8-101-score | core | M8-101-train, FV-core-101 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| M8-102-score | core | M8-102-train, FV-core-102 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| M8-103-score | core | M8-103-train, FV-core-103 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| M8-read | core | M7-101-score, M7-102-score, M7-103-score, M8-101-score, M8-102-score, M8-103-score | afterok | [0, 0]; CPU/handoff; no GPU compute |
| M9-101-incoming | core | M8-read, M-final-nomination, M4-101, FV-core-101 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| M9-102-incoming | core | M8-read, M-final-nomination, M4-102, FV-core-102 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| M9-103-incoming | core | M8-read, M-final-nomination, M4-103, FV-core-103 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| M9-101-train | core | M9-101-incoming | afterok | [7.733333333333333, 7.733333333333333]; core 58 min x 8 GPU proxy/full stage |
| M9-102-train | core | M9-102-incoming | afterok | [7.733333333333333, 7.733333333333333]; core 58 min x 8 GPU proxy/full stage |
| M9-103-train | core | M9-103-incoming | afterok | [7.733333333333333, 7.733333333333333]; core 58 min x 8 GPU proxy/full stage |
| M9-101-score | core | M9-101-train, FV-core-101 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| M9-102-score | core | M9-102-train, FV-core-102 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| M9-103-score | core | M9-103-train, FV-core-103 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| M9 | core | M8-read, M9-101-score, M9-102-score, M9-103-score | afterok | [0, 0]; CPU/handoff; no GPU compute |
| M10-101-incoming | core | M9, M5, FV-core-101 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| M10-102-incoming | core | M9, M5, FV-core-102 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| M10-103-incoming | core | M9, M5, FV-core-103 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| M10-101-train | core | M10-101-incoming | afterok | [7.733333333333333, 7.733333333333333]; core 58 min x 8 GPU proxy/full stage |
| M10-102-train | core | M10-102-incoming | afterok | [7.733333333333333, 7.733333333333333]; core 58 min x 8 GPU proxy/full stage |
| M10-103-train | core | M10-103-incoming | afterok | [7.733333333333333, 7.733333333333333]; core 58 min x 8 GPU proxy/full stage |
| M10-101-score | core | M10-101-train, FV-core-101 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| M10-102-score | core | M10-102-train, FV-core-102 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| M10-103-score | core | M10-103-train, FV-core-103 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| M10 | core | M10-101-score, M10-102-score, M10-103-score | afterok | [0, 0]; CPU/handoff; no GPU compute |
| M-core-collect | core | M3, M4-101, M4-102, M4-103, M6-101-score, M6-102-score, M6-103-score, M9, M10 | afterany | [0, 0]; CPU/handoff; no GPU compute |
| X1 | core | M10, S6 | afterok | [0, 0]; CPU/handoff; no GPU compute |
| X2 | core | X1 | afterok | [0, 0]; CPU/handoff; no GPU compute |
| X3-101 | core | X2 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X3-102 | core | X2 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X3-103 | core | X2 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X3 | core | X3-101, X3-102, X3-103 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X4-101 | core | X3 | afterok | None; core X4 all-three-seed feasibility/calibration envelope 5–7 GPU-h; per-job measurement pending |
| X4-102 | core | X3 | afterok | None; core X4 all-three-seed feasibility/calibration envelope 5–7 GPU-h; per-job measurement pending |
| X4-103 | core | X3 | afterok | None; core X4 all-three-seed feasibility/calibration envelope 5–7 GPU-h; per-job measurement pending |
| X9-probes-101-train | core | X3 | afterok | [1.5, 2.8]; two core probes x .75–1.4; included in X probe envelope |
| X9-probes-102-train | core | X3 | afterok | [1.5, 2.8]; two core probes x .75–1.4; included in X probe envelope |
| X9-probes-103-train | core | X3 | afterok | [1.5, 2.8]; two core probes x .75–1.4; included in X probe envelope |
| X4 | core | X4-101, X4-102, X4-103 | afterok | None; core X4 all-three-seed feasibility/calibration envelope 5–7 GPU-h; per-job measurement pending |
| X9-probes-101-score | core | X9-probes-101-train, FV-x-tier1-101 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X9-probes-102-score | core | X9-probes-102-train, FV-x-tier1-102 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X9-probes-103-score | core | X9-probes-103-train, FV-x-tier1-103 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X-incoming-101 | core | X4, FV-x-tier1-101 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X-incoming-102 | core | X4, FV-x-tier1-102 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X-incoming-103 | core | X4, FV-x-tier1-103 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X5-control-101-train | core | X-incoming-101 | afterok | [7.733333333333333, 7.733333333333333]; core 58 min x 8 GPU proxy/full stage |
| X6-p-101-train | core | X-incoming-101 | afterok | [7.733333333333333, 7.733333333333333]; core 58 min x 8 GPU proxy/full stage |
| X7-drift-101-train | core | X-incoming-101 | afterok | None; split stage; included in full-stage envelope, measured split pending |
| X9-plasticity-101-train | core | X-incoming-101, X9-probes-101-score | afterok | [7.733333333333333, 7.733333333333333]; core 58 min x 8 GPU proxy/full stage |
| X5-control-102-train | core | X-incoming-102 | afterok | [7.733333333333333, 7.733333333333333]; core 58 min x 8 GPU proxy/full stage |
| X6-p-102-train | core | X-incoming-102 | afterok | [7.733333333333333, 7.733333333333333]; core 58 min x 8 GPU proxy/full stage |
| X7-drift-102-train | core | X-incoming-102 | afterok | None; split stage; included in full-stage envelope, measured split pending |
| X9-plasticity-102-train | core | X-incoming-102, X9-probes-102-score | afterok | [7.733333333333333, 7.733333333333333]; core 58 min x 8 GPU proxy/full stage |
| X5-control-103-train | core | X-incoming-103 | afterok | [7.733333333333333, 7.733333333333333]; core 58 min x 8 GPU proxy/full stage |
| X6-p-103-train | core | X-incoming-103 | afterok | [7.733333333333333, 7.733333333333333]; core 58 min x 8 GPU proxy/full stage |
| X7-drift-103-train | core | X-incoming-103 | afterok | None; split stage; included in full-stage envelope, measured split pending |
| X9-plasticity-103-train | core | X-incoming-103, X9-probes-103-score | afterok | [7.733333333333333, 7.733333333333333]; core 58 min x 8 GPU proxy/full stage |
| X5-control-101-train-score | core | X5-control-101-train, FV-x-tier1-101 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X6-p-101-train-score | core | X6-p-101-train, FV-x-tier1-101 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X8-101-signal | core | X7-drift-101-train, FV-x-tier1-101 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X9-plasticity-101-train-score | core | X9-plasticity-101-train, FV-x-tier1-101 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X5-control-102-train-score | core | X5-control-102-train, FV-x-tier1-102 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X6-p-102-train-score | core | X6-p-102-train, FV-x-tier1-102 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X8-102-signal | core | X7-drift-102-train, FV-x-tier1-102 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X9-plasticity-102-train-score | core | X9-plasticity-102-train, FV-x-tier1-102 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X5-control-103-train-score | core | X5-control-103-train, FV-x-tier1-103 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X6-p-103-train-score | core | X6-p-103-train, FV-x-tier1-103 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X8-103-signal | core | X7-drift-103-train, FV-x-tier1-103 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X9-plasticity-103-train-score | core | X9-plasticity-103-train, FV-x-tier1-103 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X8-101-continue | core | X8-101-signal | afterok | None; split stage; included in full-stage envelope, measured split pending |
| X8-102-continue | core | X8-102-signal | afterok | None; split stage; included in full-stage envelope, measured split pending |
| X8-103-continue | core | X8-103-signal | afterok | None; split stage; included in full-stage envelope, measured split pending |
| X8-101-continue-score | core | X8-101-continue, FV-x-tier1-101 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X8-102-continue-score | core | X8-102-continue, FV-x-tier1-102 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X8-103-continue-score | core | X8-103-continue, FV-x-tier1-103 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X12 | core | X5-control-101-train-score, X6-p-101-train-score, X8-101-continue-score, X9-plasticity-101-train-score, X5-control-102-train-score, X6-p-102-train-score, X8-102-continue-score, X9-plasticity-102-train-score, X5-control-103-train-score, X6-p-103-train-score, X8-103-continue-score, X9-plasticity-103-train-score | afterany | [0, 0]; CPU/handoff; no GPU compute |
| X14 | core | X12 | afterok | [0, 0]; CPU/handoff; no GPU compute |
| X15-101-incoming | core | X14, FV-x-tier2.stack-drift-101 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X15-101-probe-score | core | X14, X9-probes-101-train, FV-x-tier2.stack-drift-101 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X15-102-incoming | core | X14, FV-x-tier2.stack-drift-102 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X15-102-probe-score | core | X14, X9-probes-102-train, FV-x-tier2.stack-drift-102 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X15-103-incoming | core | X14, FV-x-tier2.stack-drift-103 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X15-103-probe-score | core | X14, X9-probes-103-train, FV-x-tier2.stack-drift-103 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X16-101-incoming | core | X14, FV-x-tier2.stack-plasticity-101 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X16-101-probe-score | core | X14, X9-probes-101-train, FV-x-tier2.stack-plasticity-101 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X16-102-incoming | core | X14, FV-x-tier2.stack-plasticity-102 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X16-102-probe-score | core | X14, X9-probes-102-train, FV-x-tier2.stack-plasticity-102 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X16-103-incoming | core | X14, FV-x-tier2.stack-plasticity-103 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X16-103-probe-score | core | X14, X9-probes-103-train, FV-x-tier2.stack-plasticity-103 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X17-101-incoming | core | X14, FV-x-tier2.b-101 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X17-101-probe-score | core | X14, X9-probes-101-train, FV-x-tier2.b-101 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X17-102-incoming | core | X14, FV-x-tier2.b-102 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X17-102-probe-score | core | X14, X9-probes-102-train, FV-x-tier2.b-102 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X17-103-incoming | core | X14, FV-x-tier2.b-103 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X17-103-probe-score | core | X14, X9-probes-103-train, FV-x-tier2.b-103 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X18-101-incoming | core | X14, FV-x-tier2.b-p2-101 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X18-101-probe-score | core | X14, X9-probes-101-train, FV-x-tier2.b-p2-101 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X18-102-incoming | core | X14, FV-x-tier2.b-p2-102 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X18-102-probe-score | core | X14, X9-probes-102-train, FV-x-tier2.b-p2-102 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X18-103-incoming | core | X14, FV-x-tier2.b-p2-103 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X18-103-probe-score | core | X14, X9-probes-103-train, FV-x-tier2.b-p2-103 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X19-101-incoming | core | X14, FV-x-tier2.incremental-101 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X19-101-probe-score | core | X14, X9-probes-101-train, FV-x-tier2.incremental-101 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X19-102-incoming | core | X14, FV-x-tier2.incremental-102 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X19-102-probe-score | core | X14, X9-probes-102-train, FV-x-tier2.incremental-102 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X19-103-incoming | core | X14, FV-x-tier2.incremental-103 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X19-103-probe-score | core | X14, X9-probes-103-train, FV-x-tier2.incremental-103 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X20-101-incoming | core | X14, FV-x-tier2.incremental-p2-101 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X20-101-probe-score | core | X14, X9-probes-101-train, FV-x-tier2.incremental-p2-101 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X20-102-incoming | core | X14, FV-x-tier2.incremental-p2-102 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X20-102-probe-score | core | X14, X9-probes-102-train, FV-x-tier2.incremental-p2-102 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X20-103-incoming | core | X14, FV-x-tier2.incremental-p2-103 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X20-103-probe-score | core | X14, X9-probes-103-train, FV-x-tier2.incremental-p2-103 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X15-control-101-train | core | X15-101-incoming, X15-101-probe-score | afterok | None; split stage; included in full-stage envelope, measured split pending |
| X15-challenger-101-train | core | X15-101-incoming, X15-101-probe-score | afterok | None; split stage; included in full-stage envelope, measured split pending |
| X15-control-102-train | core | X15-102-incoming, X15-102-probe-score | afterok | None; split stage; included in full-stage envelope, measured split pending |
| X15-challenger-102-train | core | X15-102-incoming, X15-102-probe-score | afterok | None; split stage; included in full-stage envelope, measured split pending |
| X15-control-103-train | core | X15-103-incoming, X15-103-probe-score | afterok | None; split stage; included in full-stage envelope, measured split pending |
| X15-challenger-103-train | core | X15-103-incoming, X15-103-probe-score | afterok | None; split stage; included in full-stage envelope, measured split pending |
| X16-control-101-train | core | X16-101-incoming, X16-101-probe-score | afterok | [7.733333333333333, 7.733333333333333]; core 58 min x 8 GPU proxy/full stage |
| X16-challenger-101-train | core | X16-101-incoming, X16-101-probe-score | afterok | [7.733333333333333, 7.733333333333333]; core 58 min x 8 GPU proxy/full stage |
| X16-control-102-train | core | X16-102-incoming, X16-102-probe-score | afterok | [7.733333333333333, 7.733333333333333]; core 58 min x 8 GPU proxy/full stage |
| X16-challenger-102-train | core | X16-102-incoming, X16-102-probe-score | afterok | [7.733333333333333, 7.733333333333333]; core 58 min x 8 GPU proxy/full stage |
| X16-control-103-train | core | X16-103-incoming, X16-103-probe-score | afterok | [7.733333333333333, 7.733333333333333]; core 58 min x 8 GPU proxy/full stage |
| X16-challenger-103-train | core | X16-103-incoming, X16-103-probe-score | afterok | [7.733333333333333, 7.733333333333333]; core 58 min x 8 GPU proxy/full stage |
| X17-control-101-train | core | X17-101-incoming, X17-101-probe-score | afterok | None; split stage; included in full-stage envelope, measured split pending |
| X17-challenger-101-train | core | X17-101-incoming, X17-101-probe-score | afterok | None; split stage; included in full-stage envelope, measured split pending |
| X17-control-102-train | core | X17-102-incoming, X17-102-probe-score | afterok | None; split stage; included in full-stage envelope, measured split pending |
| X17-challenger-102-train | core | X17-102-incoming, X17-102-probe-score | afterok | None; split stage; included in full-stage envelope, measured split pending |
| X17-control-103-train | core | X17-103-incoming, X17-103-probe-score | afterok | None; split stage; included in full-stage envelope, measured split pending |
| X17-challenger-103-train | core | X17-103-incoming, X17-103-probe-score | afterok | None; split stage; included in full-stage envelope, measured split pending |
| X18-control-101-train | core | X18-101-incoming, X18-101-probe-score | afterok | None; split stage; included in full-stage envelope, measured split pending |
| X18-challenger-101-train | core | X18-101-incoming, X18-101-probe-score | afterok | None; split stage; included in full-stage envelope, measured split pending |
| X18-control-102-train | core | X18-102-incoming, X18-102-probe-score | afterok | None; split stage; included in full-stage envelope, measured split pending |
| X18-challenger-102-train | core | X18-102-incoming, X18-102-probe-score | afterok | None; split stage; included in full-stage envelope, measured split pending |
| X18-control-103-train | core | X18-103-incoming, X18-103-probe-score | afterok | None; split stage; included in full-stage envelope, measured split pending |
| X18-challenger-103-train | core | X18-103-incoming, X18-103-probe-score | afterok | None; split stage; included in full-stage envelope, measured split pending |
| X19-control-101-train | core | X19-101-incoming, X19-101-probe-score | afterok | None; split stage; included in full-stage envelope, measured split pending |
| X19-challenger-101-train | core | X19-101-incoming, X19-101-probe-score | afterok | None; split stage; included in full-stage envelope, measured split pending |
| X19-control-102-train | core | X19-102-incoming, X19-102-probe-score | afterok | None; split stage; included in full-stage envelope, measured split pending |
| X19-challenger-102-train | core | X19-102-incoming, X19-102-probe-score | afterok | None; split stage; included in full-stage envelope, measured split pending |
| X19-control-103-train | core | X19-103-incoming, X19-103-probe-score | afterok | None; split stage; included in full-stage envelope, measured split pending |
| X19-challenger-103-train | core | X19-103-incoming, X19-103-probe-score | afterok | None; split stage; included in full-stage envelope, measured split pending |
| X20-control-101-train | core | X20-101-incoming, X20-101-probe-score | afterok | None; split stage; included in full-stage envelope, measured split pending |
| X20-challenger-101-train | core | X20-101-incoming, X20-101-probe-score | afterok | None; split stage; included in full-stage envelope, measured split pending |
| X20-control-102-train | core | X20-102-incoming, X20-102-probe-score | afterok | None; split stage; included in full-stage envelope, measured split pending |
| X20-challenger-102-train | core | X20-102-incoming, X20-102-probe-score | afterok | None; split stage; included in full-stage envelope, measured split pending |
| X20-control-103-train | core | X20-103-incoming, X20-103-probe-score | afterok | None; split stage; included in full-stage envelope, measured split pending |
| X20-challenger-103-train | core | X20-103-incoming, X20-103-probe-score | afterok | None; split stage; included in full-stage envelope, measured split pending |
| X15-control-101-train-measure10 | core | X15-control-101-train, FV-x-tier2.stack-drift-101 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X15-challenger-101-train-measure10 | core | X15-challenger-101-train, FV-x-tier2.stack-drift-101 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X15-control-102-train-measure10 | core | X15-control-102-train, FV-x-tier2.stack-drift-102 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X15-challenger-102-train-measure10 | core | X15-challenger-102-train, FV-x-tier2.stack-drift-102 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X15-control-103-train-measure10 | core | X15-control-103-train, FV-x-tier2.stack-drift-103 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X15-challenger-103-train-measure10 | core | X15-challenger-103-train, FV-x-tier2.stack-drift-103 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X16-control-101-train-score | core | X16-control-101-train, FV-x-tier2.stack-plasticity-101 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X16-challenger-101-train-score | core | X16-challenger-101-train, FV-x-tier2.stack-plasticity-101 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X16-control-102-train-score | core | X16-control-102-train, FV-x-tier2.stack-plasticity-102 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X16-challenger-102-train-score | core | X16-challenger-102-train, FV-x-tier2.stack-plasticity-102 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X16-control-103-train-score | core | X16-control-103-train, FV-x-tier2.stack-plasticity-103 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X16-challenger-103-train-score | core | X16-challenger-103-train, FV-x-tier2.stack-plasticity-103 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X17-control-101-train-measure10 | core | X17-control-101-train, FV-x-tier2.b-101 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X17-challenger-101-train-measure10 | core | X17-challenger-101-train, FV-x-tier2.b-101 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X17-control-102-train-measure10 | core | X17-control-102-train, FV-x-tier2.b-102 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X17-challenger-102-train-measure10 | core | X17-challenger-102-train, FV-x-tier2.b-102 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X17-control-103-train-measure10 | core | X17-control-103-train, FV-x-tier2.b-103 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X17-challenger-103-train-measure10 | core | X17-challenger-103-train, FV-x-tier2.b-103 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X18-control-101-train-measure10 | core | X18-control-101-train, FV-x-tier2.b-p2-101 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X18-challenger-101-train-measure10 | core | X18-challenger-101-train, FV-x-tier2.b-p2-101 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X18-control-102-train-measure10 | core | X18-control-102-train, FV-x-tier2.b-p2-102 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X18-challenger-102-train-measure10 | core | X18-challenger-102-train, FV-x-tier2.b-p2-102 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X18-control-103-train-measure10 | core | X18-control-103-train, FV-x-tier2.b-p2-103 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X18-challenger-103-train-measure10 | core | X18-challenger-103-train, FV-x-tier2.b-p2-103 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X19-control-101-train-measure10 | core | X19-control-101-train, FV-x-tier2.incremental-101 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X19-challenger-101-continue10 | core | X19-challenger-101-train | afterok | None; split stage; included in full-stage envelope, measured split pending |
| X19-control-102-train-measure10 | core | X19-control-102-train, FV-x-tier2.incremental-102 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X19-challenger-102-continue10 | core | X19-challenger-102-train | afterok | None; split stage; included in full-stage envelope, measured split pending |
| X19-control-103-train-measure10 | core | X19-control-103-train, FV-x-tier2.incremental-103 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X19-challenger-103-continue10 | core | X19-challenger-103-train | afterok | None; split stage; included in full-stage envelope, measured split pending |
| X20-control-101-train-measure10 | core | X20-control-101-train, FV-x-tier2.incremental-p2-101 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X20-challenger-101-continue10 | core | X20-challenger-101-train | afterok | None; split stage; included in full-stage envelope, measured split pending |
| X20-control-102-train-measure10 | core | X20-control-102-train, FV-x-tier2.incremental-p2-102 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X20-challenger-102-continue10 | core | X20-challenger-102-train | afterok | None; split stage; included in full-stage envelope, measured split pending |
| X20-control-103-train-measure10 | core | X20-control-103-train, FV-x-tier2.incremental-p2-103 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X20-challenger-103-continue10 | core | X20-challenger-103-train | afterok | None; split stage; included in full-stage envelope, measured split pending |
| X15-control-101-continue10 | core | X15-control-101-train-measure10 | afterok | None; split stage; included in full-stage envelope, measured split pending |
| X15-challenger-101-continue10 | core | X15-challenger-101-train-measure10 | afterok | None; split stage; included in full-stage envelope, measured split pending |
| X15-control-102-continue10 | core | X15-control-102-train-measure10 | afterok | None; split stage; included in full-stage envelope, measured split pending |
| X15-challenger-102-continue10 | core | X15-challenger-102-train-measure10 | afterok | None; split stage; included in full-stage envelope, measured split pending |
| X15-control-103-continue10 | core | X15-control-103-train-measure10 | afterok | None; split stage; included in full-stage envelope, measured split pending |
| X15-challenger-103-continue10 | core | X15-challenger-103-train-measure10 | afterok | None; split stage; included in full-stage envelope, measured split pending |
| X17-control-101-continue10 | core | X17-control-101-train-measure10 | afterok | None; split stage; included in full-stage envelope, measured split pending |
| X17-challenger-101-continue10 | core | X17-challenger-101-train-measure10 | afterok | None; split stage; included in full-stage envelope, measured split pending |
| X17-control-102-continue10 | core | X17-control-102-train-measure10 | afterok | None; split stage; included in full-stage envelope, measured split pending |
| X17-challenger-102-continue10 | core | X17-challenger-102-train-measure10 | afterok | None; split stage; included in full-stage envelope, measured split pending |
| X17-control-103-continue10 | core | X17-control-103-train-measure10 | afterok | None; split stage; included in full-stage envelope, measured split pending |
| X17-challenger-103-continue10 | core | X17-challenger-103-train-measure10 | afterok | None; split stage; included in full-stage envelope, measured split pending |
| X18-control-101-continue10 | core | X18-control-101-train-measure10 | afterok | None; split stage; included in full-stage envelope, measured split pending |
| X18-challenger-101-continue10 | core | X18-challenger-101-train-measure10 | afterok | None; split stage; included in full-stage envelope, measured split pending |
| X18-control-102-continue10 | core | X18-control-102-train-measure10 | afterok | None; split stage; included in full-stage envelope, measured split pending |
| X18-challenger-102-continue10 | core | X18-challenger-102-train-measure10 | afterok | None; split stage; included in full-stage envelope, measured split pending |
| X18-control-103-continue10 | core | X18-control-103-train-measure10 | afterok | None; split stage; included in full-stage envelope, measured split pending |
| X18-challenger-103-continue10 | core | X18-challenger-103-train-measure10 | afterok | None; split stage; included in full-stage envelope, measured split pending |
| X19-control-101-continue10 | core | X19-control-101-train-measure10 | afterok | None; split stage; included in full-stage envelope, measured split pending |
| X19-challenger-101-continue10-score | core | X19-challenger-101-continue10, FV-x-tier2.incremental-101 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X19-control-102-continue10 | core | X19-control-102-train-measure10 | afterok | None; split stage; included in full-stage envelope, measured split pending |
| X19-challenger-102-continue10-score | core | X19-challenger-102-continue10, FV-x-tier2.incremental-102 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X19-control-103-continue10 | core | X19-control-103-train-measure10 | afterok | None; split stage; included in full-stage envelope, measured split pending |
| X19-challenger-103-continue10-score | core | X19-challenger-103-continue10, FV-x-tier2.incremental-103 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X20-control-101-continue10 | core | X20-control-101-train-measure10 | afterok | None; split stage; included in full-stage envelope, measured split pending |
| X20-challenger-101-continue10-score | core | X20-challenger-101-continue10, FV-x-tier2.incremental-p2-101 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X20-control-102-continue10 | core | X20-control-102-train-measure10 | afterok | None; split stage; included in full-stage envelope, measured split pending |
| X20-challenger-102-continue10-score | core | X20-challenger-102-continue10, FV-x-tier2.incremental-p2-102 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X20-control-103-continue10 | core | X20-control-103-train-measure10 | afterok | None; split stage; included in full-stage envelope, measured split pending |
| X20-challenger-103-continue10-score | core | X20-challenger-103-continue10, FV-x-tier2.incremental-p2-103 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X15-control-101-continue10-score | core | X15-control-101-continue10, FV-x-tier2.stack-drift-101 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X15-challenger-101-continue10-score | core | X15-challenger-101-continue10, FV-x-tier2.stack-drift-101 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X15-control-102-continue10-score | core | X15-control-102-continue10, FV-x-tier2.stack-drift-102 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X15-challenger-102-continue10-score | core | X15-challenger-102-continue10, FV-x-tier2.stack-drift-102 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X15-control-103-continue10-score | core | X15-control-103-continue10, FV-x-tier2.stack-drift-103 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X15-challenger-103-continue10-score | core | X15-challenger-103-continue10, FV-x-tier2.stack-drift-103 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X17-control-101-continue10-score | core | X17-control-101-continue10, FV-x-tier2.b-101 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X17-challenger-101-continue10-measure20 | core | X17-challenger-101-continue10, FV-x-tier2.b-101 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X17-control-102-continue10-score | core | X17-control-102-continue10, FV-x-tier2.b-102 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X17-challenger-102-continue10-measure20 | core | X17-challenger-102-continue10, FV-x-tier2.b-102 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X17-control-103-continue10-score | core | X17-control-103-continue10, FV-x-tier2.b-103 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X17-challenger-103-continue10-measure20 | core | X17-challenger-103-continue10, FV-x-tier2.b-103 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X18-control-101-continue10-score | core | X18-control-101-continue10, FV-x-tier2.b-p2-101 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X18-challenger-101-continue10-measure20 | core | X18-challenger-101-continue10, FV-x-tier2.b-p2-101 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X18-control-102-continue10-score | core | X18-control-102-continue10, FV-x-tier2.b-p2-102 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X18-challenger-102-continue10-measure20 | core | X18-challenger-102-continue10, FV-x-tier2.b-p2-102 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X18-control-103-continue10-score | core | X18-control-103-continue10, FV-x-tier2.b-p2-103 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X18-challenger-103-continue10-measure20 | core | X18-challenger-103-continue10, FV-x-tier2.b-p2-103 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X19-control-101-continue10-score | core | X19-control-101-continue10, FV-x-tier2.incremental-101 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X19-control-102-continue10-score | core | X19-control-102-continue10, FV-x-tier2.incremental-102 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X19-control-103-continue10-score | core | X19-control-103-continue10, FV-x-tier2.incremental-103 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X20-control-101-continue10-score | core | X20-control-101-continue10, FV-x-tier2.incremental-p2-101 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X20-control-102-continue10-score | core | X20-control-102-continue10, FV-x-tier2.incremental-p2-102 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X20-control-103-continue10-score | core | X20-control-103-continue10, FV-x-tier2.incremental-p2-103 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X17-challenger-101-continue20 | core | X17-challenger-101-continue10-measure20 | afterok | None; split stage; included in full-stage envelope, measured split pending |
| X17-challenger-102-continue20 | core | X17-challenger-102-continue10-measure20 | afterok | None; split stage; included in full-stage envelope, measured split pending |
| X17-challenger-103-continue20 | core | X17-challenger-103-continue10-measure20 | afterok | None; split stage; included in full-stage envelope, measured split pending |
| X18-challenger-101-continue20 | core | X18-challenger-101-continue10-measure20 | afterok | None; split stage; included in full-stage envelope, measured split pending |
| X18-challenger-102-continue20 | core | X18-challenger-102-continue10-measure20 | afterok | None; split stage; included in full-stage envelope, measured split pending |
| X18-challenger-103-continue20 | core | X18-challenger-103-continue10-measure20 | afterok | None; split stage; included in full-stage envelope, measured split pending |
| X17-challenger-101-continue20-measure30 | core | X17-challenger-101-continue20, FV-x-tier2.b-101 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X17-challenger-102-continue20-measure30 | core | X17-challenger-102-continue20, FV-x-tier2.b-102 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X17-challenger-103-continue20-measure30 | core | X17-challenger-103-continue20, FV-x-tier2.b-103 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X18-challenger-101-continue20-measure30 | core | X18-challenger-101-continue20, FV-x-tier2.b-p2-101 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X18-challenger-102-continue20-measure30 | core | X18-challenger-102-continue20, FV-x-tier2.b-p2-102 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X18-challenger-103-continue20-measure30 | core | X18-challenger-103-continue20, FV-x-tier2.b-p2-103 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X17-challenger-101-continue30 | core | X17-challenger-101-continue20-measure30 | afterok | None; split stage; included in full-stage envelope, measured split pending |
| X17-challenger-102-continue30 | core | X17-challenger-102-continue20-measure30 | afterok | None; split stage; included in full-stage envelope, measured split pending |
| X17-challenger-103-continue30 | core | X17-challenger-103-continue20-measure30 | afterok | None; split stage; included in full-stage envelope, measured split pending |
| X18-challenger-101-continue30 | core | X18-challenger-101-continue20-measure30 | afterok | None; split stage; included in full-stage envelope, measured split pending |
| X18-challenger-102-continue30 | core | X18-challenger-102-continue20-measure30 | afterok | None; split stage; included in full-stage envelope, measured split pending |
| X18-challenger-103-continue30 | core | X18-challenger-103-continue20-measure30 | afterok | None; split stage; included in full-stage envelope, measured split pending |
| X17-challenger-101-continue30-measure40 | core | X17-challenger-101-continue30, FV-x-tier2.b-101 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X17-challenger-102-continue30-measure40 | core | X17-challenger-102-continue30, FV-x-tier2.b-102 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X17-challenger-103-continue30-measure40 | core | X17-challenger-103-continue30, FV-x-tier2.b-103 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X18-challenger-101-continue30-measure40 | core | X18-challenger-101-continue30, FV-x-tier2.b-p2-101 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X18-challenger-102-continue30-measure40 | core | X18-challenger-102-continue30, FV-x-tier2.b-p2-102 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X18-challenger-103-continue30-measure40 | core | X18-challenger-103-continue30, FV-x-tier2.b-p2-103 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X17-challenger-101-continue30-measure40-score | core | X17-challenger-101-continue30-measure40, FV-x-tier2.b-101 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X17-challenger-102-continue30-measure40-score | core | X17-challenger-102-continue30-measure40, FV-x-tier2.b-102 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X17-challenger-103-continue30-measure40-score | core | X17-challenger-103-continue30-measure40, FV-x-tier2.b-103 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X18-challenger-101-continue30-measure40-score | core | X18-challenger-101-continue30-measure40, FV-x-tier2.b-p2-101 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X18-challenger-102-continue30-measure40-score | core | X18-challenger-102-continue30-measure40, FV-x-tier2.b-p2-102 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X18-challenger-103-continue30-measure40-score | core | X18-challenger-103-continue30-measure40, FV-x-tier2.b-p2-103 | afterok | None; core measured scoring/preparation/qualification allocation pending |
| X23 | core | X12, X15-control-101-continue10-score, X15-challenger-101-continue10-score, X15-control-102-continue10-score, X15-challenger-102-continue10-score, X15-control-103-continue10-score, X15-challenger-103-continue10-score, X16-control-101-train-score, X16-challenger-101-train-score, X16-control-102-train-score, X16-challenger-102-train-score, X16-control-103-train-score, X16-challenger-103-train-score, X17-control-101-continue10-score, X17-challenger-101-continue30-measure40-score, X17-control-102-continue10-score, X17-challenger-102-continue30-measure40-score, X17-control-103-continue10-score, X17-challenger-103-continue30-measure40-score, X18-control-101-continue10-score, X18-challenger-101-continue30-measure40-score, X18-control-102-continue10-score, X18-challenger-102-continue30-measure40-score, X18-control-103-continue10-score, X18-challenger-103-continue30-measure40-score, X19-control-101-continue10-score, X19-challenger-101-continue10-score, X19-control-102-continue10-score, X19-challenger-102-continue10-score, X19-control-103-continue10-score, X19-challenger-103-continue10-score, X20-control-101-continue10-score, X20-challenger-101-continue10-score, X20-control-102-continue10-score, X20-challenger-102-continue10-score, X20-control-103-continue10-score, X20-challenger-103-continue10-score | afterany | [0, 0]; CPU/handoff; no GPU compute |
| M3-init | kcr | none | afterok | [0, 0]; K/CR CPU/handoff |
| M3-bind-inputs | kcr | M3-init | afterok | [0, 0]; K/CR CPU/handoff |
| M3-hellaswag | kcr | M3-init | afterok | [0, 0]; K/CR CPU/handoff |
| M3-bind-base | kcr | M3-init | afterok | [0, 0]; K/CR CPU/handoff |
| M3-math-targets | kcr | M3-init | afterok | [10.0, 30.0]; K/CR shared generation; compatible committed core targets remove new math generation charge |
| M3-assign | kcr | M3-init | afterok | [0, 0]; K/CR CPU/handoff |
| M3-verify-superset | kcr | M3-bind-inputs | afterok | [0, 0]; K/CR CPU/handoff |
| M3-finqa-targets | kcr | M3-bind-inputs | afterok | [0, 0]; K/CR CPU/handoff |
| M3-contract | kcr | M3-bind-inputs, M3-hellaswag | afterok | [0, 0]; K/CR CPU/handoff |
| M3-finqa-replay | kcr | M3-bind-inputs, M3-hellaswag | afterok | [0, 0]; K/CR CPU/handoff |
| M3-broad | kcr | M3-contract, M3-finqa-replay | afterok | [2.0, 25.0]; K/CR shared generation; compatible committed core targets remove new math generation charge |
| M3-pools-shared | kcr | M3-bind-inputs, M3-verify-superset, M3-hellaswag | afterok | [0, 0]; K/CR CPU/handoff |
| M3-pools | kcr | M3-broad, M3-finqa-replay | afterok | [0, 0]; K/CR CPU/handoff |
| M3-schedules | kcr | M3-pools-shared, M3-finqa-targets, M3-bind-inputs | afterok | [0, 0]; K/CR CPU/handoff |
| M3-schedules-math | kcr | M3-pools-shared, M3-math-targets | afterok | [0, 0]; K/CR CPU/handoff |
| M3-schedules-coverage | kcr | M3-pools, M3-schedules, M3-finqa-targets | afterok | [0, 0]; K/CR CPU/handoff |
| M3-schedules-coverage-math | kcr | M3-pools, M3-schedules-math, M3-math-targets | afterok | [0, 0]; K/CR CPU/handoff |
| M3-preflight-C0 | kcr | M3-bind-base, M3-schedules | afterok | [4.8, 9.16]; K/CR one qualification per arm; includes diagnostic/setup reserve |
| M3-preflight-K | kcr | M3-bind-base, M3-schedules | afterok | [4.94, 9.79]; K/CR one qualification per arm; includes diagnostic/setup reserve |
| M3-base-101 | kcr | M3-assign, M3-bind-base, M3-schedules, M3-hellaswag | afterok | None; K/CR shared base/validation total 2.10–4.20; per-job measurement pending |
| M3-base-102 | kcr | M3-assign, M3-bind-base, M3-schedules, M3-hellaswag | afterok | None; K/CR shared base/validation total 2.10–4.20; per-job measurement pending |
| M3-base-103 | kcr | M3-assign, M3-bind-base, M3-schedules, M3-hellaswag | afterok | None; K/CR shared base/validation total 2.10–4.20; per-job measurement pending |
| M3-preflight-G | kcr | M3-bind-base, M3-schedules-coverage | afterok | [6.69, 12.24]; K/CR one qualification per arm; includes diagnostic/setup reserve |
| M3-preflight-CR | kcr | M3-bind-base, M3-schedules-coverage | afterok | [6.69, 12.24]; K/CR one qualification per arm; includes diagnostic/setup reserve |
| M3-C0-101-finqa-train | kcr | M3-base-101, M3-preflight-C0 | afterok | [15.49, 25.56]; K/CR measured-smoke calibrated stage; setup/scoring/qualification are separate envelope components |
| M3-K-101-finqa-train | kcr | M3-base-101, M3-preflight-K | afterok | [16.06, 28.09]; K/CR measured-smoke calibrated stage; setup/scoring/qualification are separate envelope components |
| M3-C0-102-finqa-train | kcr | M3-base-102, M3-preflight-C0 | afterok | [15.49, 25.56]; K/CR measured-smoke calibrated stage; setup/scoring/qualification are separate envelope components |
| M3-K-102-finqa-train | kcr | M3-base-102, M3-preflight-K | afterok | [16.06, 28.09]; K/CR measured-smoke calibrated stage; setup/scoring/qualification are separate envelope components |
| M3-C0-103-finqa-train | kcr | M3-base-103, M3-preflight-C0 | afterok | [15.49, 25.56]; K/CR measured-smoke calibrated stage; setup/scoring/qualification are separate envelope components |
| M3-K-103-finqa-train | kcr | M3-base-103, M3-preflight-K | afterok | [16.06, 28.09]; K/CR measured-smoke calibrated stage; setup/scoring/qualification are separate envelope components |
| M3-C0-101-finqa-score | kcr | M3-C0-101-finqa-train | afterok | [0.8, 1.6]; K/CR 16 call-equivalent allowance per arm/seed/stage |
| M3-K-101-finqa-score | kcr | M3-K-101-finqa-train | afterok | [0.8, 1.6]; K/CR 16 call-equivalent allowance per arm/seed/stage |
| M3-G-101-finqa-train | kcr | M3-base-101, M3-preflight-G, M3-schedules-coverage | afterok | [19.37, 31.95]; K/CR measured-smoke calibrated stage; setup/scoring/qualification are separate envelope components |
| M3-CR-101-finqa-train | kcr | M3-base-101, M3-preflight-CR, M3-schedules-coverage | afterok | [19.37, 31.95]; K/CR measured-smoke calibrated stage; setup/scoring/qualification are separate envelope components |
| M3-C0-102-finqa-score | kcr | M3-C0-102-finqa-train | afterok | [0.8, 1.6]; K/CR 16 call-equivalent allowance per arm/seed/stage |
| M3-K-102-finqa-score | kcr | M3-K-102-finqa-train | afterok | [0.8, 1.6]; K/CR 16 call-equivalent allowance per arm/seed/stage |
| M3-G-102-finqa-train | kcr | M3-base-102, M3-preflight-G, M3-schedules-coverage | afterok | [19.37, 31.95]; K/CR measured-smoke calibrated stage; setup/scoring/qualification are separate envelope components |
| M3-CR-102-finqa-train | kcr | M3-base-102, M3-preflight-CR, M3-schedules-coverage | afterok | [19.37, 31.95]; K/CR measured-smoke calibrated stage; setup/scoring/qualification are separate envelope components |
| M3-C0-103-finqa-score | kcr | M3-C0-103-finqa-train | afterok | [0.8, 1.6]; K/CR 16 call-equivalent allowance per arm/seed/stage |
| M3-K-103-finqa-score | kcr | M3-K-103-finqa-train | afterok | [0.8, 1.6]; K/CR 16 call-equivalent allowance per arm/seed/stage |
| M3-G-103-finqa-train | kcr | M3-base-103, M3-preflight-G, M3-schedules-coverage | afterok | [19.37, 31.95]; K/CR measured-smoke calibrated stage; setup/scoring/qualification are separate envelope components |
| M3-CR-103-finqa-train | kcr | M3-base-103, M3-preflight-CR, M3-schedules-coverage | afterok | [19.37, 31.95]; K/CR measured-smoke calibrated stage; setup/scoring/qualification are separate envelope components |
| M3-fv-101 | kcr | M3-assign, M3-bind-base, M3-hellaswag, M3-C0-101-finqa-train | afterok | None; K/CR shared base/validation total 2.10–4.20; per-job measurement pending |
| M3-kv-101 | kcr | M3-assign, M3-bind-base, M3-C0-101-finqa-train | afterok | None; K/CR shared base/validation total 2.10–4.20; per-job measurement pending |
| M3-fv-102 | kcr | M3-assign, M3-bind-base, M3-hellaswag, M3-C0-101-finqa-train | afterok | None; K/CR shared base/validation total 2.10–4.20; per-job measurement pending |
| M3-kv-102 | kcr | M3-assign, M3-bind-base, M3-C0-101-finqa-train | afterok | None; K/CR shared base/validation total 2.10–4.20; per-job measurement pending |
| M3-fv-103 | kcr | M3-assign, M3-bind-base, M3-hellaswag, M3-C0-101-finqa-train | afterok | None; K/CR shared base/validation total 2.10–4.20; per-job measurement pending |
| M3-kv-103 | kcr | M3-assign, M3-bind-base, M3-C0-101-finqa-train | afterok | None; K/CR shared base/validation total 2.10–4.20; per-job measurement pending |
| M3-C0-101-contract_nli-train | kcr | M3-C0-101-finqa-score, M3-preflight-C0 | afterok | [15.49, 25.56]; K/CR measured-smoke calibrated stage; setup/scoring/qualification are separate envelope components |
| M3-K-101-contract_nli-train | kcr | M3-K-101-finqa-score, M3-preflight-K | afterok | [16.06, 28.09]; K/CR measured-smoke calibrated stage; setup/scoring/qualification are separate envelope components |
| M3-G-101-finqa-score | kcr | M3-G-101-finqa-train | afterok | [0.8, 1.6]; K/CR 16 call-equivalent allowance per arm/seed/stage |
| M3-CR-101-finqa-score | kcr | M3-CR-101-finqa-train | afterok | [0.8, 1.6]; K/CR 16 call-equivalent allowance per arm/seed/stage |
| M3-C0-102-contract_nli-train | kcr | M3-C0-102-finqa-score, M3-preflight-C0 | afterok | [15.49, 25.56]; K/CR measured-smoke calibrated stage; setup/scoring/qualification are separate envelope components |
| M3-K-102-contract_nli-train | kcr | M3-K-102-finqa-score, M3-preflight-K | afterok | [16.06, 28.09]; K/CR measured-smoke calibrated stage; setup/scoring/qualification are separate envelope components |
| M3-G-102-finqa-score | kcr | M3-G-102-finqa-train | afterok | [0.8, 1.6]; K/CR 16 call-equivalent allowance per arm/seed/stage |
| M3-CR-102-finqa-score | kcr | M3-CR-102-finqa-train | afterok | [0.8, 1.6]; K/CR 16 call-equivalent allowance per arm/seed/stage |
| M3-C0-103-contract_nli-train | kcr | M3-C0-103-finqa-score, M3-preflight-C0 | afterok | [15.49, 25.56]; K/CR measured-smoke calibrated stage; setup/scoring/qualification are separate envelope components |
| M3-K-103-contract_nli-train | kcr | M3-K-103-finqa-score, M3-preflight-K | afterok | [16.06, 28.09]; K/CR measured-smoke calibrated stage; setup/scoring/qualification are separate envelope components |
| M3-G-103-finqa-score | kcr | M3-G-103-finqa-train | afterok | [0.8, 1.6]; K/CR 16 call-equivalent allowance per arm/seed/stage |
| M3-CR-103-finqa-score | kcr | M3-CR-103-finqa-train | afterok | [0.8, 1.6]; K/CR 16 call-equivalent allowance per arm/seed/stage |
| M3-C0-101-contract_nli-score | kcr | M3-C0-101-contract_nli-train | afterok | [0.8, 1.6]; K/CR 16 call-equivalent allowance per arm/seed/stage |
| M3-K-101-contract_nli-score | kcr | M3-K-101-contract_nli-train | afterok | [0.8, 1.6]; K/CR 16 call-equivalent allowance per arm/seed/stage |
| M3-G-101-contract_nli-train | kcr | M3-G-101-finqa-score, M3-preflight-G, M3-schedules-coverage | afterok | [23.24, 38.34]; K/CR measured-smoke calibrated stage; setup/scoring/qualification are separate envelope components |
| M3-CR-101-contract_nli-train | kcr | M3-CR-101-finqa-score, M3-preflight-CR, M3-schedules-coverage | afterok | [23.24, 38.34]; K/CR measured-smoke calibrated stage; setup/scoring/qualification are separate envelope components |
| M3-C0-102-contract_nli-score | kcr | M3-C0-102-contract_nli-train | afterok | [0.8, 1.6]; K/CR 16 call-equivalent allowance per arm/seed/stage |
| M3-K-102-contract_nli-score | kcr | M3-K-102-contract_nli-train | afterok | [0.8, 1.6]; K/CR 16 call-equivalent allowance per arm/seed/stage |
| M3-G-102-contract_nli-train | kcr | M3-G-102-finqa-score, M3-preflight-G, M3-schedules-coverage | afterok | [23.24, 38.34]; K/CR measured-smoke calibrated stage; setup/scoring/qualification are separate envelope components |
| M3-CR-102-contract_nli-train | kcr | M3-CR-102-finqa-score, M3-preflight-CR, M3-schedules-coverage | afterok | [23.24, 38.34]; K/CR measured-smoke calibrated stage; setup/scoring/qualification are separate envelope components |
| M3-C0-103-contract_nli-score | kcr | M3-C0-103-contract_nli-train | afterok | [0.8, 1.6]; K/CR 16 call-equivalent allowance per arm/seed/stage |
| M3-K-103-contract_nli-score | kcr | M3-K-103-contract_nli-train | afterok | [0.8, 1.6]; K/CR 16 call-equivalent allowance per arm/seed/stage |
| M3-G-103-contract_nli-train | kcr | M3-G-103-finqa-score, M3-preflight-G, M3-schedules-coverage | afterok | [23.24, 38.34]; K/CR measured-smoke calibrated stage; setup/scoring/qualification are separate envelope components |
| M3-CR-103-contract_nli-train | kcr | M3-CR-103-finqa-score, M3-preflight-CR, M3-schedules-coverage | afterok | [23.24, 38.34]; K/CR measured-smoke calibrated stage; setup/scoring/qualification are separate envelope components |
| M3-C0-101-math_hard-train | kcr | M3-C0-101-contract_nli-score, M3-preflight-C0, M3-schedules-math | afterok | [15.49, 25.56]; K/CR measured-smoke calibrated stage; setup/scoring/qualification are separate envelope components |
| M3-K-101-math_hard-train | kcr | M3-K-101-contract_nli-score, M3-preflight-K, M3-schedules-math | afterok | [16.06, 28.09]; K/CR measured-smoke calibrated stage; setup/scoring/qualification are separate envelope components |
| M3-G-101-contract_nli-score | kcr | M3-G-101-contract_nli-train | afterok | [0.8, 1.6]; K/CR 16 call-equivalent allowance per arm/seed/stage |
| M3-CR-101-contract_nli-score | kcr | M3-CR-101-contract_nli-train | afterok | [0.8, 1.6]; K/CR 16 call-equivalent allowance per arm/seed/stage |
| M3-C0-102-math_hard-train | kcr | M3-C0-102-contract_nli-score, M3-preflight-C0, M3-schedules-math | afterok | [15.49, 25.56]; K/CR measured-smoke calibrated stage; setup/scoring/qualification are separate envelope components |
| M3-K-102-math_hard-train | kcr | M3-K-102-contract_nli-score, M3-preflight-K, M3-schedules-math | afterok | [16.06, 28.09]; K/CR measured-smoke calibrated stage; setup/scoring/qualification are separate envelope components |
| M3-G-102-contract_nli-score | kcr | M3-G-102-contract_nli-train | afterok | [0.8, 1.6]; K/CR 16 call-equivalent allowance per arm/seed/stage |
| M3-CR-102-contract_nli-score | kcr | M3-CR-102-contract_nli-train | afterok | [0.8, 1.6]; K/CR 16 call-equivalent allowance per arm/seed/stage |
| M3-C0-103-math_hard-train | kcr | M3-C0-103-contract_nli-score, M3-preflight-C0, M3-schedules-math | afterok | [15.49, 25.56]; K/CR measured-smoke calibrated stage; setup/scoring/qualification are separate envelope components |
| M3-K-103-math_hard-train | kcr | M3-K-103-contract_nli-score, M3-preflight-K, M3-schedules-math | afterok | [16.06, 28.09]; K/CR measured-smoke calibrated stage; setup/scoring/qualification are separate envelope components |
| M3-G-103-contract_nli-score | kcr | M3-G-103-contract_nli-train | afterok | [0.8, 1.6]; K/CR 16 call-equivalent allowance per arm/seed/stage |
| M3-CR-103-contract_nli-score | kcr | M3-CR-103-contract_nli-train | afterok | [0.8, 1.6]; K/CR 16 call-equivalent allowance per arm/seed/stage |
| M3-C0-101-math_hard-score | kcr | M3-C0-101-math_hard-train | afterok | [0.8, 1.6]; K/CR 16 call-equivalent allowance per arm/seed/stage |
| M3-K-101-math_hard-score | kcr | M3-K-101-math_hard-train | afterok | [0.8, 1.6]; K/CR 16 call-equivalent allowance per arm/seed/stage |
| M3-G-101-math_hard-train | kcr | M3-G-101-contract_nli-score, M3-preflight-G, M3-schedules-coverage-math | afterok | [27.11, 44.73]; K/CR measured-smoke calibrated stage; setup/scoring/qualification are separate envelope components |
| M3-CR-101-math_hard-train | kcr | M3-CR-101-contract_nli-score, M3-preflight-CR, M3-schedules-coverage-math | afterok | [27.11, 44.73]; K/CR measured-smoke calibrated stage; setup/scoring/qualification are separate envelope components |
| M3-C0-102-math_hard-score | kcr | M3-C0-102-math_hard-train | afterok | [0.8, 1.6]; K/CR 16 call-equivalent allowance per arm/seed/stage |
| M3-K-102-math_hard-score | kcr | M3-K-102-math_hard-train | afterok | [0.8, 1.6]; K/CR 16 call-equivalent allowance per arm/seed/stage |
| M3-G-102-math_hard-train | kcr | M3-G-102-contract_nli-score, M3-preflight-G, M3-schedules-coverage-math | afterok | [27.11, 44.73]; K/CR measured-smoke calibrated stage; setup/scoring/qualification are separate envelope components |
| M3-CR-102-math_hard-train | kcr | M3-CR-102-contract_nli-score, M3-preflight-CR, M3-schedules-coverage-math | afterok | [27.11, 44.73]; K/CR measured-smoke calibrated stage; setup/scoring/qualification are separate envelope components |
| M3-C0-103-math_hard-score | kcr | M3-C0-103-math_hard-train | afterok | [0.8, 1.6]; K/CR 16 call-equivalent allowance per arm/seed/stage |
| M3-K-103-math_hard-score | kcr | M3-K-103-math_hard-train | afterok | [0.8, 1.6]; K/CR 16 call-equivalent allowance per arm/seed/stage |
| M3-G-103-math_hard-train | kcr | M3-G-103-contract_nli-score, M3-preflight-G, M3-schedules-coverage-math | afterok | [27.11, 44.73]; K/CR measured-smoke calibrated stage; setup/scoring/qualification are separate envelope components |
| M3-CR-103-math_hard-train | kcr | M3-CR-103-contract_nli-score, M3-preflight-CR, M3-schedules-coverage-math | afterok | [27.11, 44.73]; K/CR measured-smoke calibrated stage; setup/scoring/qualification are separate envelope components |
| M3-G-101-math_hard-score | kcr | M3-G-101-math_hard-train | afterok | [0.8, 1.6]; K/CR 16 call-equivalent allowance per arm/seed/stage |
| M3-CR-101-math_hard-score | kcr | M3-CR-101-math_hard-train | afterok | [0.8, 1.6]; K/CR 16 call-equivalent allowance per arm/seed/stage |
| M3-G-102-math_hard-score | kcr | M3-G-102-math_hard-train | afterok | [0.8, 1.6]; K/CR 16 call-equivalent allowance per arm/seed/stage |
| M3-CR-102-math_hard-score | kcr | M3-CR-102-math_hard-train | afterok | [0.8, 1.6]; K/CR 16 call-equivalent allowance per arm/seed/stage |
| M3-G-103-math_hard-score | kcr | M3-G-103-math_hard-train | afterok | [0.8, 1.6]; K/CR 16 call-equivalent allowance per arm/seed/stage |
| M3-CR-103-math_hard-score | kcr | M3-CR-103-math_hard-train | afterok | [0.8, 1.6]; K/CR 16 call-equivalent allowance per arm/seed/stage |
| M3-read | kcr | M3-pools, M3-C0-101-math_hard-score, M3-K-101-math_hard-score, M3-G-101-math_hard-score, M3-CR-101-math_hard-score, M3-C0-102-math_hard-score, M3-K-102-math_hard-score, M3-G-102-math_hard-score, M3-CR-102-math_hard-score, M3-C0-103-math_hard-score, M3-K-103-math_hard-score, M3-G-103-math_hard-score, M3-CR-103-math_hard-score, M3-fv-101, M3-kv-101, M3-fv-102, M3-kv-102, M3-fv-103, M3-kv-103 | afterany | [0, 0]; K/CR CPU/handoff |
| SP-verify-pools | sp | M-pools, M3-pools, R-broad, R-nominate | afterany | [0, 0]; SP(b) .25–1 GPU-h/entry; startup separate; unresolved coverage has no finite upper total |
| SP-panel-core-s101 | sp | M10-101-incoming, M10-101-score, M4-101, M6-101-incoming, M6-101-score, M7-101-incoming, M7-101-score, M8-101-incoming, M8-101-score, M9-101-incoming, M9-101-score | afterany | [4.5, 18]; SP(b) .25–1 GPU-h/entry; startup separate; unresolved coverage has no finite upper total |
| SP-panel-core-s102 | sp | M10-102-incoming, M10-102-score, M4-102, M6-102-incoming, M6-102-score, M7-102-incoming, M7-102-score, M8-102-incoming, M8-102-score, M9-102-incoming, M9-102-score | afterany | [4.5, 18]; SP(b) .25–1 GPU-h/entry; startup separate; unresolved coverage has no finite upper total |
| SP-panel-core-s103 | sp | M10-103-incoming, M10-103-score, M4-103, M6-103-incoming, M6-103-score, M7-103-incoming, M7-103-score, M8-103-incoming, M8-103-score, M9-103-incoming, M9-103-score | afterany | [4.5, 18]; SP(b) .25–1 GPU-h/entry; startup separate; unresolved coverage has no finite upper total |
| SP-panel-m3-s101 | sp | M3-C0-101-contract_nli-score, M3-C0-101-contract_nli-train, M3-C0-101-finqa-score, M3-C0-101-finqa-train, M3-C0-101-math_hard-score, M3-C0-101-math_hard-train, M3-CR-101-contract_nli-score, M3-CR-101-contract_nli-train, M3-CR-101-finqa-score, M3-CR-101-finqa-train, M3-CR-101-math_hard-score, M3-CR-101-math_hard-train, M3-G-101-contract_nli-score, M3-G-101-contract_nli-train, M3-G-101-finqa-score, M3-G-101-finqa-train, M3-G-101-math_hard-score, M3-G-101-math_hard-train, M3-K-101-contract_nli-score, M3-K-101-contract_nli-train, M3-K-101-finqa-score, M3-K-101-finqa-train, M3-K-101-math_hard-score, M3-K-101-math_hard-train, M3-base-101 | afterany | [3.25, 13]; SP(b) .25–1 GPU-h/entry; startup separate; unresolved coverage has no finite upper total |
| SP-panel-m3-s102 | sp | M3-C0-102-contract_nli-score, M3-C0-102-contract_nli-train, M3-C0-102-finqa-score, M3-C0-102-finqa-train, M3-C0-102-math_hard-score, M3-C0-102-math_hard-train, M3-CR-102-contract_nli-score, M3-CR-102-contract_nli-train, M3-CR-102-finqa-score, M3-CR-102-finqa-train, M3-CR-102-math_hard-score, M3-CR-102-math_hard-train, M3-G-102-contract_nli-score, M3-G-102-contract_nli-train, M3-G-102-finqa-score, M3-G-102-finqa-train, M3-G-102-math_hard-score, M3-G-102-math_hard-train, M3-K-102-contract_nli-score, M3-K-102-contract_nli-train, M3-K-102-finqa-score, M3-K-102-finqa-train, M3-K-102-math_hard-score, M3-K-102-math_hard-train, M3-base-102 | afterany | [3.25, 13]; SP(b) .25–1 GPU-h/entry; startup separate; unresolved coverage has no finite upper total |
| SP-panel-m3-s103 | sp | M3-C0-103-contract_nli-score, M3-C0-103-contract_nli-train, M3-C0-103-finqa-score, M3-C0-103-finqa-train, M3-C0-103-math_hard-score, M3-C0-103-math_hard-train, M3-CR-103-contract_nli-score, M3-CR-103-contract_nli-train, M3-CR-103-finqa-score, M3-CR-103-finqa-train, M3-CR-103-math_hard-score, M3-CR-103-math_hard-train, M3-G-103-contract_nli-score, M3-G-103-contract_nli-train, M3-G-103-finqa-score, M3-G-103-finqa-train, M3-G-103-math_hard-score, M3-G-103-math_hard-train, M3-K-103-contract_nli-score, M3-K-103-contract_nli-train, M3-K-103-finqa-score, M3-K-103-finqa-train, M3-K-103-math_hard-score, M3-K-103-math_hard-train, M3-base-103 | afterany | [3.25, 13]; SP(b) .25–1 GPU-h/entry; startup separate; unresolved coverage has no finite upper total |
| SP-panel-reprobe-s101 | sp | R-both-101-score, R-broad-101-score, R-earlier-101-score, R-generic-101-score, R-incoming-101 | afterany | [2.25, 9]; SP(b) .25–1 GPU-h/entry; startup separate; unresolved coverage has no finite upper total |
| SP-panel-reprobe-s102 | sp | R-both-102-score, R-broad-102-score, R-earlier-102-score, R-generic-102-score, R-incoming-102 | afterany | [2.25, 9]; SP(b) .25–1 GPU-h/entry; startup separate; unresolved coverage has no finite upper total |
| SP-panel-reprobe-s103 | sp | R-both-103-score, R-broad-103-score, R-earlier-103-score, R-generic-103-score, R-incoming-103 | afterany | [2.25, 9]; SP(b) .25–1 GPU-h/entry; startup separate; unresolved coverage has no finite upper total |
| SP-panel-return1.A101-s101 | sp | RA101 | afterany | [1.75, 7]; SP(b) .25–1 GPU-h/entry; startup separate; unresolved coverage has no finite upper total |
| SP-panel-return1.A102-s102 | sp | RA102 | afterany | [1.5, 6]; SP(b) .25–1 GPU-h/entry; startup separate; unresolved coverage has no finite upper total |
| SP-panel-return1.A103-s103 | sp | RA103 | afterany | [2.5, 10]; SP(b) .25–1 GPU-h/entry; startup separate; unresolved coverage has no finite upper total |
| SP-panel-x-tier1-s101 | sp | X-incoming-101, X5-control-101-train-score, X6-p-101-train-score, X8-101-continue-score, X9-plasticity-101-train-score | afterany | [1.5, 6]; SP(b) .25–1 GPU-h/entry; startup separate; unresolved coverage has no finite upper total |
| SP-panel-x-tier1-s102 | sp | X-incoming-102, X5-control-102-train-score, X6-p-102-train-score, X8-102-continue-score, X9-plasticity-102-train-score | afterany | [1.5, 6]; SP(b) .25–1 GPU-h/entry; startup separate; unresolved coverage has no finite upper total |
| SP-panel-x-tier1-s103 | sp | X-incoming-103, X5-control-103-train-score, X6-p-103-train-score, X8-103-continue-score, X9-plasticity-103-train-score | afterany | [1.5, 6]; SP(b) .25–1 GPU-h/entry; startup separate; unresolved coverage has no finite upper total |
| SP-panel-x-tier2.b-p2-s101 | sp | X18-101-incoming, X18-challenger-101-continue30-measure40-score, X18-control-101-continue10-score | afterany | [0.75, 3]; SP(b) .25–1 GPU-h/entry; startup separate; unresolved coverage has no finite upper total |
| SP-panel-x-tier2.b-p2-s102 | sp | X18-102-incoming, X18-challenger-102-continue30-measure40-score, X18-control-102-continue10-score | afterany | [0.75, 3]; SP(b) .25–1 GPU-h/entry; startup separate; unresolved coverage has no finite upper total |
| SP-panel-x-tier2.b-p2-s103 | sp | X18-103-incoming, X18-challenger-103-continue30-measure40-score, X18-control-103-continue10-score | afterany | [0.75, 3]; SP(b) .25–1 GPU-h/entry; startup separate; unresolved coverage has no finite upper total |
| SP-panel-x-tier2.b-s101 | sp | X17-101-incoming, X17-challenger-101-continue30-measure40-score, X17-control-101-continue10-score | afterany | [0.75, 3]; SP(b) .25–1 GPU-h/entry; startup separate; unresolved coverage has no finite upper total |
| SP-panel-x-tier2.b-s102 | sp | X17-102-incoming, X17-challenger-102-continue30-measure40-score, X17-control-102-continue10-score | afterany | [0.75, 3]; SP(b) .25–1 GPU-h/entry; startup separate; unresolved coverage has no finite upper total |
| SP-panel-x-tier2.b-s103 | sp | X17-103-incoming, X17-challenger-103-continue30-measure40-score, X17-control-103-continue10-score | afterany | [0.75, 3]; SP(b) .25–1 GPU-h/entry; startup separate; unresolved coverage has no finite upper total |
| SP-panel-x-tier2.incremental-p2-s101 | sp | X20-101-incoming, X20-challenger-101-continue10-score, X20-control-101-continue10-score | afterany | [0.75, 3]; SP(b) .25–1 GPU-h/entry; startup separate; unresolved coverage has no finite upper total |
| SP-panel-x-tier2.incremental-p2-s102 | sp | X20-102-incoming, X20-challenger-102-continue10-score, X20-control-102-continue10-score | afterany | [0.75, 3]; SP(b) .25–1 GPU-h/entry; startup separate; unresolved coverage has no finite upper total |
| SP-panel-x-tier2.incremental-p2-s103 | sp | X20-103-incoming, X20-challenger-103-continue10-score, X20-control-103-continue10-score | afterany | [0.75, 3]; SP(b) .25–1 GPU-h/entry; startup separate; unresolved coverage has no finite upper total |
| SP-panel-x-tier2.incremental-s101 | sp | X19-101-incoming, X19-challenger-101-continue10-score, X19-control-101-continue10-score | afterany | [0.75, 3]; SP(b) .25–1 GPU-h/entry; startup separate; unresolved coverage has no finite upper total |
| SP-panel-x-tier2.incremental-s102 | sp | X19-102-incoming, X19-challenger-102-continue10-score, X19-control-102-continue10-score | afterany | [0.75, 3]; SP(b) .25–1 GPU-h/entry; startup separate; unresolved coverage has no finite upper total |
| SP-panel-x-tier2.incremental-s103 | sp | X19-103-incoming, X19-challenger-103-continue10-score, X19-control-103-continue10-score | afterany | [0.75, 3]; SP(b) .25–1 GPU-h/entry; startup separate; unresolved coverage has no finite upper total |
| SP-panel-x-tier2.stack-drift-s101 | sp | X15-101-incoming, X15-challenger-101-continue10-score, X15-control-101-continue10-score | afterany | [0.75, 3]; SP(b) .25–1 GPU-h/entry; startup separate; unresolved coverage has no finite upper total |
| SP-panel-x-tier2.stack-drift-s102 | sp | X15-102-incoming, X15-challenger-102-continue10-score, X15-control-102-continue10-score | afterany | [0.75, 3]; SP(b) .25–1 GPU-h/entry; startup separate; unresolved coverage has no finite upper total |
| SP-panel-x-tier2.stack-drift-s103 | sp | X15-103-incoming, X15-challenger-103-continue10-score, X15-control-103-continue10-score | afterany | [0.75, 3]; SP(b) .25–1 GPU-h/entry; startup separate; unresolved coverage has no finite upper total |
| SP-panel-x-tier2.stack-plasticity-s101 | sp | X16-101-incoming, X16-challenger-101-train-score, X16-control-101-train-score | afterany | [0.75, 3]; SP(b) .25–1 GPU-h/entry; startup separate; unresolved coverage has no finite upper total |
| SP-panel-x-tier2.stack-plasticity-s102 | sp | X16-102-incoming, X16-challenger-102-train-score, X16-control-102-train-score | afterany | [0.75, 3]; SP(b) .25–1 GPU-h/entry; startup separate; unresolved coverage has no finite upper total |
| SP-panel-x-tier2.stack-plasticity-s103 | sp | X16-103-incoming, X16-challenger-103-train-score, X16-control-103-train-score | afterany | [0.75, 3]; SP(b) .25–1 GPU-h/entry; startup separate; unresolved coverage has no finite upper total |
| SP-length | sp | M10-101-incoming, M10-101-score, M10-102-incoming, M10-102-score, M10-103-incoming, M10-103-score, M3-C0-101-contract_nli-score, M3-C0-101-contract_nli-train, M3-C0-101-finqa-score, M3-C0-101-finqa-train, M3-C0-101-math_hard-score, M3-C0-101-math_hard-train, M3-C0-102-contract_nli-score, M3-C0-102-contract_nli-train, M3-C0-102-finqa-score, M3-C0-102-finqa-train, M3-C0-102-math_hard-score, M3-C0-102-math_hard-train, M3-C0-103-contract_nli-score, M3-C0-103-contract_nli-train, M3-C0-103-finqa-score, M3-C0-103-finqa-train, M3-C0-103-math_hard-score, M3-C0-103-math_hard-train, M3-CR-101-contract_nli-score, M3-CR-101-contract_nli-train, M3-CR-101-finqa-score, M3-CR-101-finqa-train, M3-CR-101-math_hard-score, M3-CR-101-math_hard-train, M3-CR-102-contract_nli-score, M3-CR-102-contract_nli-train, M3-CR-102-finqa-score, M3-CR-102-finqa-train, M3-CR-102-math_hard-score, M3-CR-102-math_hard-train, M3-CR-103-contract_nli-score, M3-CR-103-contract_nli-train, M3-CR-103-finqa-score, M3-CR-103-finqa-train, M3-CR-103-math_hard-score, M3-CR-103-math_hard-train, M3-G-101-contract_nli-score, M3-G-101-contract_nli-train, M3-G-101-finqa-score, M3-G-101-finqa-train, M3-G-101-math_hard-score, M3-G-101-math_hard-train, M3-G-102-contract_nli-score, M3-G-102-contract_nli-train, M3-G-102-finqa-score, M3-G-102-finqa-train, M3-G-102-math_hard-score, M3-G-102-math_hard-train, M3-G-103-contract_nli-score, M3-G-103-contract_nli-train, M3-G-103-finqa-score, M3-G-103-finqa-train, M3-G-103-math_hard-score, M3-G-103-math_hard-train, M3-K-101-contract_nli-score, M3-K-101-contract_nli-train, M3-K-101-finqa-score, M3-K-101-finqa-train, M3-K-101-math_hard-score, M3-K-101-math_hard-train, M3-K-102-contract_nli-score, M3-K-102-contract_nli-train, M3-K-102-finqa-score, M3-K-102-finqa-train, M3-K-102-math_hard-score, M3-K-102-math_hard-train, M3-K-103-contract_nli-score, M3-K-103-contract_nli-train, M3-K-103-finqa-score, M3-K-103-finqa-train, M3-K-103-math_hard-score, M3-K-103-math_hard-train, M3-base-101, M3-base-102, M3-base-103, M4-101, M4-102, M4-103, M6-101-incoming, M6-101-score, M6-102-incoming, M6-102-score, M6-103-incoming, M6-103-score, M7-101-incoming, M7-101-score, M7-102-incoming, M7-102-score, M7-103-incoming, M7-103-score, M8-101-incoming, M8-101-score, M8-102-incoming, M8-102-score, M8-103-incoming, M8-103-score, M9-101-incoming, M9-101-score, M9-102-incoming, M9-102-score, M9-103-incoming, M9-103-score, R-both-101-score, R-both-102-score, R-both-103-score, R-broad-101-score, R-broad-102-score, R-broad-103-score, R-earlier-101-score, R-earlier-102-score, R-earlier-103-score, R-generic-101-score, R-generic-102-score, R-generic-103-score, R-incoming-101, R-incoming-102, R-incoming-103, RA101, RA102, RA103, X-incoming-101, X-incoming-102, X-incoming-103, X15-101-incoming, X15-102-incoming, X15-103-incoming, X15-challenger-101-continue10-score, X15-challenger-102-continue10-score, X15-challenger-103-continue10-score, X15-control-101-continue10-score, X15-control-102-continue10-score, X15-control-103-continue10-score, X16-101-incoming, X16-102-incoming, X16-103-incoming, X16-challenger-101-train-score, X16-challenger-102-train-score, X16-challenger-103-train-score, X16-control-101-train-score, X16-control-102-train-score, X16-control-103-train-score, X17-101-incoming, X17-102-incoming, X17-103-incoming, X17-challenger-101-continue30-measure40-score, X17-challenger-102-continue30-measure40-score, X17-challenger-103-continue30-measure40-score, X17-control-101-continue10-score, X17-control-102-continue10-score, X17-control-103-continue10-score, X18-101-incoming, X18-102-incoming, X18-103-incoming, X18-challenger-101-continue30-measure40-score, X18-challenger-102-continue30-measure40-score, X18-challenger-103-continue30-measure40-score, X18-control-101-continue10-score, X18-control-102-continue10-score, X18-control-103-continue10-score, X19-101-incoming, X19-102-incoming, X19-103-incoming, X19-challenger-101-continue10-score, X19-challenger-102-continue10-score, X19-challenger-103-continue10-score, X19-control-101-continue10-score, X19-control-102-continue10-score, X19-control-103-continue10-score, X20-101-incoming, X20-102-incoming, X20-103-incoming, X20-challenger-101-continue10-score, X20-challenger-102-continue10-score, X20-challenger-103-continue10-score, X20-control-101-continue10-score, X20-control-102-continue10-score, X20-control-103-continue10-score, X5-control-101-train-score, X5-control-102-train-score, X5-control-103-train-score, X6-p-101-train-score, X6-p-102-train-score, X6-p-103-train-score, X8-101-continue-score, X8-102-continue-score, X8-103-continue-score, X9-plasticity-101-train-score, X9-plasticity-102-train-score, X9-plasticity-103-train-score | afterany | [0, 0]; SP(b) .25–1 GPU-h/entry; startup separate; unresolved coverage has no finite upper total |
| SP-read | sp | SP-length, SP-panel-core-s101, SP-panel-core-s102, SP-panel-core-s103, SP-panel-m3-s101, SP-panel-m3-s102, SP-panel-m3-s103, SP-panel-reprobe-s101, SP-panel-reprobe-s102, SP-panel-reprobe-s103, SP-panel-return1.A101-s101, SP-panel-return1.A102-s102, SP-panel-return1.A103-s103, SP-panel-x-tier1-s101, SP-panel-x-tier1-s102, SP-panel-x-tier1-s103, SP-panel-x-tier2.b-p2-s101, SP-panel-x-tier2.b-p2-s102, SP-panel-x-tier2.b-p2-s103, SP-panel-x-tier2.b-s101, SP-panel-x-tier2.b-s102, SP-panel-x-tier2.b-s103, SP-panel-x-tier2.incremental-p2-s101, SP-panel-x-tier2.incremental-p2-s102, SP-panel-x-tier2.incremental-p2-s103, SP-panel-x-tier2.incremental-s101, SP-panel-x-tier2.incremental-s102, SP-panel-x-tier2.incremental-s103, SP-panel-x-tier2.stack-drift-s101, SP-panel-x-tier2.stack-drift-s102, SP-panel-x-tier2.stack-drift-s103, SP-panel-x-tier2.stack-plasticity-s101, SP-panel-x-tier2.stack-plasticity-s102, SP-panel-x-tier2.stack-plasticity-s103, SP-verify-pools | afterany | [0, 0]; SP(b) .25–1 GPU-h/entry; startup separate; unresolved coverage has no finite upper total |
| PD-101 | pd | none | after core-submitted + PF-tier on node | [73.864972, 87.873469]; PD measured reprice, per seed; within approved tier component allowance |
| PD-102 | pd | none | after core-submitted + PF-tier on node | [73.864972, 87.873469]; PD measured reprice, per seed; within approved tier component allowance |
| PD-103 | pd | none | after core-submitted + PF-tier on node | [73.864972, 87.873469]; PD measured reprice, per seed; within approved tier component allowance |

Component envelopes (overlap per-job components; not additive):

- Core M: 109.6 floor includes 46.4 MATH +10 targets +30 scoring/export +23.2 legal; +23.2 per confirmation set; fresh 12-stage re-probe +92.8 proxy; independent 31-call rescore/overhead unpriced
- Screen S: 30–50 envelope; v2 scoring/transfer measurement pending
- X tier 1: 12 x 7.733333 = 92.8 training; probes/generation/scoring/X4/startup additional
- X tier 2: 36 x 7.733333 = 278.4 training; 295.2–316.8 historical envelope; v2 overheads remain measured separately
- K/CR: [786.297517, 1387.157525]
- PD: [221.594915, 263.620406]
- SP: [54.75, 216.0]; entry sum 53.75–215 plus 1 GPU-h startup =54.75–216; SP(a)=0

Approved tier allowance: 1,800 GPU-h; component limits K/CR <=1,400, PD <=264, SP <=216. The estimated upper case is 1,866.78 GPU-h, about 67 above the allowance; a late PD whole-seed restart adds up to 87.87 (about 88) GPU-h. These are warnings monitored in partner receipts, never launch/scientific gates; no cells are dropped. Core is separate. Queue/restore/transfer time and failed attempts must be charged separately.
