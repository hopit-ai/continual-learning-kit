# Send 5: package 4, one campaign, one archive back

The tag is named in our message. This send runs in the **same WORK folder as your pilot** (send 4,
`campaigns/k8b-pilot.yaml`): it reads the pilot's runs, its report and the runner's records there, and writes
everything new under `$WORK/k8b4/` and `$WORK/runs/p4-*`.

## Why this send

Your pilot report says which recipe and task order lost the first task in both of its runs. Package 4 takes the two
stage-1 checkpoints of that group and runs two experiments that we registered before any pilot number existed:

| experiment | what it asks | runs | GPU-hour ceiling |
|---|---|---|---|
| prevention screen | does training the second task with a 2,048-token limit, and reward only for answers that end by themselves, keep the first task without learning the second one less? | 2 qualification runs of 2 steps, then 8 runs of 40 steps (two lineages, two seeds, control and intervention) | 75 |
| bridges | from one of those checkpoints, which change of setting (minibatch, learning rate) and which change of algorithm at equal settings moves the loss? | 6 runs of 40 steps (two of the eight configurations are the screen's controls and are not trained again) | 125 |

Every new checkpoint, the untrained model and the pilot's selected checkpoints are scored on both tasks and the
general panel, at 8,192 new tokens with token ids, on one GPU. No pilot scoring is reused (amendment 3).

There are four campaign files, `campaigns/k8b-p4-{g8,sema}-{chemtool,toolchem}.yaml`, one per group the selection
can name. **We will tell you which one to run** after we have read your pilot archive. Its first row runs the selector
on your own pilot and stops at once, naming the right file, if your pilot selects another group.

## Before the command, once

- Keep the pilot's WORK folder exactly as the pilot left it (`/work/k8b-work`): its `runs/` (the merged models
  `runs/*/hf-step40` of the selected group are the starting checkpoints, and each run's `env/argv.txt` is the recipe
  the campaign checks against), `k8b/` (the pilot report and its scorings) and `campaign/k8b-pilot/` (the pilot's
  recorded times, from which the reservation is computed). Do not move, rename or rerun anything in it: the first row
  freezes a record of the finished pilot, and any later change to it makes the campaign refuse to go on.
- The same node, the same environment as the pilot: the same `MODEL_DIR` (the first row checks it is the untrained
  model your pilot's first stages started from) and the same `SDPO_DIR` checkout, at the same commit and **clean**:
  `git -C $SDPO_DIR status --short` must print nothing (the third row checks the commit and the clean state against
  what your pilot recorded. Your pilot's launchers recorded no hash of the task data, so the third row runs the
  authors' own preprocessing again into a scratch folder and requires the task files in place to be byte for byte
  what it writes; it changes nothing in `$SDPO_DIR`. If any of this cannot be established, the third row stops the
  campaign and says why -- please send us that message).
  You do **not** set `NGPU`, `TP` or `OFFLOAD` for this send: the campaign reads them from your pilot's own recorded
  commands and uses those values.
- `nvidia-smi` must be on `PATH` (the first row checks); it is on a standard GPU node.

```bash
export KIT=/work/continual-learning-kit/kit SDPO_DIR=/work/SDPO
cd /work/continual-learning-kit && git fetch --tags && git checkout <TAG>      # the tag named in our message
```

No new package to install.

## The command

```bash
WORK=/work/k8b-work MODEL_DIR=/work/models/Qwen3-8B python $KIT/runner.py batch $KIT/campaigns/k8b-p4-<GROUP>.yaml
```

`<GROUP>` is the one we name (for example `sema-chemtool`). One line, nothing else.

**One job at a time on the node; scoring on one physical GPU.** A training row takes the GPUs your pilot's runs took
(the count is read from your pilot's recorded settings, not set by you or by the campaign) and a scoring row takes GPU 0 (`CUDA_VISIBLE_DEVICES=0`, every scoring after the one before it), because the report refuses to compare
scorings whose machine fingerprints differ. The runner above runs rows one after another, and every GPU row first waits
(up to two minutes) until `nvidia-smi` shows no compute process on the GPUs it will use; if one is still there it fails
with "GPU still busy" and launches nothing. Please run nothing else on the node while the campaign runs.

## What the first rows do

The first nine rows are pilots; if one fails, everything after it stops:

1. `select` checks that your pilot is the released one and is finished (every row has a verdict, the report is the
   latest), writes `k8b4/selection/pilot-finalization.json` and the selection, and stops unless the selection names this
   file's group. Both records are frozen: running the line again recomputes them and goes on only if nothing changed.
2. `reserve` computes the GPU-hour reservation from your pilot's recorded times and prints it.
3. `recipe-check` compares every training command of this send with your pilot's own recorded command for the same
   recipe, setting by setting (the run name, its output folders, the starting checkpoint, the seeds and, where
   registered, the task data, the minibatch of one configuration, the 2,048 limit, the reward path of the finish gate
   and the two steps of the qualification may differ, each by exactly its registered value; nothing else), and the
   trainer's resolved configuration of both the same way. It takes the launcher settings (`NGPU`, `TP`, `OFFLOAD`, ...)
   from your pilot's records. Its first result is frozen, whether it passed or failed: a failed comparison is not
   repeated by running the line again.
4. `containment-check` freezes the Slurm configuration and allocation receipt.
5. `containment-selftest` uses one GPU and a fixed 300-second admission from prevention to test descendant cleanup after killing its clients.
6. to 9. two 2-step qualification runs of the intervention and their scorings.

## How the time limits are enforced

Run the batch line inside a Slurm allocation with `SLURM_JOB_ID` set and `srun`, `scontrol`, `scancel` and `squeue` on PATH. If Slurm removes finished steps from `scontrol`, `sacct` must provide their exact step IDs, terminal states and start/end times; missing accounting prevents verification. The node must permit independently timed job steps. The first containment row requires `proctrack/cgroup`, `task/cgroup`, a nonnegative integer KillWait, and global/partition overtime zero or absent. It records SignalChildrenProcesses rather than inferring its behaviour from the plugin name. An unqualified cluster stops at that pilot row and prints the failed checks; no training starts.

The next row starts a one-minute adversarial step, kills its own wrapper, real watchdog and srun client, and observes detached, environment-clearing, TERM-ignoring and orphaned workers from outside the step, including a worker that acquires CUDA later. Outside discovery is deliberately paused across the timeout; recorded death times are observation upper bounds. Missing torch/CUDA makes this check fail. The expected probe timeout is inside its separate 300-second spending admission. Both receipts are frozen; a failed selftest is final in this WORK.

Each later GPU row is parked in one step until Slurm reports its actual start and granted time. Whole minutes are rounded down to fit both campaign deadlines, allowing at least 60 seconds for enforcement/clock delay and the recorded KillWait. `KIT_P4_SLURM_ALLOWANCE` may raise that allowance. A late start or changed physical GPU identity cancels that step without releasing the command. Startup may wait at most 120 seconds; identified abandoned pending steps are cancelled and verified absent. An unidentifiable submitted step or unverified cancellation stops further admission with accounting unavailable; its parked command is never released. The armed scheduler timeout survives loss of all clients. Every exit cancels only its owning step; the row is charged until scheduler termination and idle GPUs are observed. A scheduler timeout in a training/scoring row permanently stops that block. No passing smoke test establishes a bound on arbitrary scheduler delays.

## The two ceilings, and how the campaign keeps to them

The prevention screen may use at most **75 GPU-hours** and the bridges at most **125**, 200 in all; neither is paid
from the other. Every GPU row -- training and scoring -- asks `kit/p4_budget.py` first whether its block still fits,
and runs in its own independently timed Slurm step under a time limit computed so that the block's outstanding scorings stay paid for
and the 60-second shutdown fits. When the row's command ends -- by itself, by its limit, or because the line was
interrupted -- its owning step is cancelled and verified ended, and the GPUs are polled until `nvidia-smi`
shows them idle; the row is charged until that moment. If a row is refused for its budget, is ended by its limit, or
leaves a process on its GPUs that cannot be shown ended, that block is **stopped for good**: `k8b4/budget/stop-<block>.json` is written, every later GPU row of that block is refused (retries included),
and the report labels the block "budget-incomplete". The other block goes on. Each decision is one line of
`$WORK/k8b4/budget/ledger.jsonl`.

What the campaign does not do by itself: it cannot stop work started outside it, and the conservative enforcement/clock allowance is a registered assumption rather than a measured cluster guarantee; please do not start other GPU work on the node while it runs.

Any GPU-hour figure for this send is an estimate until it has run. The second row prints the reservation the
registered rule derives from your pilot's records: for the screen, nine times T (the mean GPU-hours of your pilot's
40-step second-stage runs of the selected recipe) plus 45 times E (1.5 times the mean GPU-hours of your pilot's
separable long scorings, amendment 3); for the bridges, seven times the largest such recipe mean plus 20 E. The
ceilings above are the only bounds.

## If a row fails

After the pilots, a failed training row costs only its own statistics, scorings and sweeps: every other run, the
bridges and the report still run, and the runner exits non-zero.

**Run the same line once more; the campaign itself decides whether a failed training row may be retried and says why
if not.** Rows that passed are skipped and every row that read a re-run row runs again by itself. What a re-run checks
although it skips them: every GPU row, the budget rows and the report first re-hash the frozen records of the first three
rows and every pilot file they were computed from (the pilot report, the pilot's scorings and sweeps, each pilot run's
summary and recorded command, commit, clean state and data hashes, the runner's records of the pilot) and refuse,
naming the file, if any of them changed; each training row also re-checks the `SDPO_DIR` commit and clean state, the task
data hashes, and the starting checkpoint's content (the sha256 of every file of it, recorded by the recipe check: a
changed or replaced file refuses the launch). Every GPU row runs under a time limit computed from its experiment's ledger, followed by a reserved 60-second
shutdown allowance that ends at the row's hard deadline. Shutdown begins when the admitted time ends, on a fixed
schedule: at that moment the row's process group is sent SIGTERM, and 30 seconds later -- in any case 15 seconds before
the deadline -- SIGKILL. These two signals go to the group the row was started in, remembered from its start, and are
sent before anything is looked up or written: no slow `nvidia-smi`, process listing or disk can delay them. Alongside,
the campaign looks for the rest of the row's processes, and each one it finds is stopped the same way (asked before the
SIGKILL time, killed after it); every lookup is cut off when the next signal or the deadline is due. These process-group signals and scans are a secondary line. Processes with the marker `KIT_P4_JOB` include descendants that keep it, and the scans include every process of yours that holds one of the row's GPUs and was started while the row ran. Those observations alone cannot prove every descendant gone. Persistent containment comes from the independently timed Slurm step, whose owning step alone is cancelled FIRST at the limit, before any scan. By the deadline every one of them must be shown gone; if that cannot be shown (for example
because `nvidia-smi` does not answer), the row is "workers not verified terminated", the experiment's budget accounting
is marked unavailable and nothing more of that experiment runs. A small watchdog process, started in its own session
with the row's process group before the row's command runs, keeps the same schedule by itself: if the line itself is
killed (a closed terminal, a killed runner), the watchdog still ends that row's processes when the row's admitted time
ends and records when it found them all gone and the GPUs idle. A watchdog only ever signals its own row's group, the
processes carrying its own row's marker, and GPU holders that started while its own row ran -- never the next row's
worker. A row that was still running when its admitted time ended is a spending-limit termination: that verdict and
the experiment's stop record are written to disk right after the first SIGTERM, before the row is marked ended, and
the next run of the line completes any record it finds in that state.

When a row ends, the line waits until that row's watchdog has written to disk that it stood down (or, if it acted,
its own result) before going on; and before admitting any GPU row it checks that no watchdog of the WORK folder is still
running without having done so, waits for it (up to 30 seconds, or until that row's deadline if its time has run out),
and otherwise refuses the row ("watchdog unacknowledged", exit 3, nothing launched; run the line again). The next run
of the line first reconciles a row whose line was killed: it is charged up to the time the watchdog recorded; if no such
time was recorded, the next run ends whatever is left of that row, checks that it is all gone and the GPUs idle, tells
its watchdog to stand down, waits for the watchdog to confirm, and charges the row up to that moment; if it cannot show
the GPUs idle (for example a process that is not yours still holds one), the experiment's budget accounting is marked
unavailable and nothing more of that experiment runs. A row whose watchdog had to end it, or that is found and charged
only at or after the end of its admitted time, counts as ended by its limit: it is never retried, nothing more of its
experiment runs, and the experiment gets no registered label, even if its stop record were lost. Campaign GPU rows run with `KIT_P4_CONTAINMENT=slurm-step`. Explicit `KIT_P4_CONTAINMENT=process-group` execution has descriptive reporting only: budget compliance is unavailable and there are no registered labels. A training run that
already left a complete model is never trained again (even if the runner was interrupted before recording it). At most
one training retry is allowed per experiment, only for a 40-step run whose settings are recorded and correct (its
summary, and its actual command and resolved configuration compared with your pilot's as in the recipe check), whose
model is missing or incomplete, and which was not stopped by its budget; a qualification run is never retried. When a
retry is not allowed, the row's record (`k8b4/budget/attempts/<row>-a<N>.json`) and its log say why. Running the line
a third time changes nothing that is final. A failed scoring or sweep is simply run again by the same line.

What you do by hand: keep the WORK folder and the environment as above, run the one line (again after a failure), run
nothing else on the node meanwhile, run the collect line at the end, and send any file it names as INCOMPLETE separately.

## What can withhold a label

The report gives an experiment no registered label -- it prints every number it can, descriptively -- when any of
these holds, and says which:

- any GPU attempt lacks verified Slurm containment, passing frozen receipts or complete scheduler evidence ("process-group-only execution", descriptive reporting only);
- the experiment was stopped by its budget: a row refused for its budget, ended by its limit, or found running at the
  end of its admitted time ("budget-incomplete");
- a row's processes could not be shown gone by its deadline, or a killed line's row could not be reconciled ("budget
  compliance unavailable": missing accounting evidence, not a demonstrated overrun);
- a run of record is missing, failed, has wrong or unrecorded settings, or a scoring, sweep or prefix check it needs is
  missing or does not match its recorded hashes;
- a frozen input changed after it was frozen (the selection, the reservation, the recipe check and its baseline);
- a training attempt -- failed ones included -- has no reduced rollout records, or its records do not match what the
  trainer did (a completed run's steps each need their full 256 rollouts; a gated run's rows all need the cut flag and
  both rewards);
- **a failed training attempt with no rollout records, unless it is positively shown that it rolled nothing out.**
  Each launcher writes its own progress to `runs/<run>/env/stage.json` as it goes (`started`, `config-resolved`,
  `trainer-invoked`, `trainer-exited`, `merged`), each step written to disk before it goes on, and on any exit appends
  `exited` with its status and the last step reached. A failed attempt with no rows counts as "no rollout before the
  failure" only if the line never started its launcher, or that file -- naming this attempt and its `KIT_P4_JOB`
  marker -- ends before `trainer-invoked` with the `exited` entry. In every other case (no such file, no `exited`
  entry, the trainer invoked, metrics empty or without a step) its rollout evidence is "unavailable" and the experiment
  gets no registered label. This includes a trainer that was started and crashed before it recorded its first step:
  such a crash costs the experiment its labels, by the registered rule (amendment 3 B8). Please do not delete or edit
  anything under `runs/`.

## Disk

About 315 GB: 14 runs and 2 qualification runs each keep a merged model of about 17.6 GB, and one trainer checkpoint
of about 33 GB exists at a time (it is deleted as soon as its merged model is complete). The first row refuses to start
with less than 320 GB free under `$WORK` and says so (`K8B4_MIN_FREE_GB` overrides the number; please do not lower it
without telling us).

## What to send back

```bash
python $KIT/collect.py --work /work/k8b-work --out send5-k8b4.tar.gz
```

Text only, no weights: the selection and the pilot's finalization record, the reservation, the ledger, the attempt
records and any stop record, the recipe check, every scoring's answers and token ids, the sweeps and prefix checks,
each run's command, resolved configuration, stage record, summary and metrics, a reduced record of every training rollout (lengths
and rewards, no text), the report, the log of every row that did not pass, and a copy of the analysis code
(`kit-snapshot/`). It packs the pilot's tree again too (we check that nothing in it changed). If a line starting
`INCOMPLETE:` is printed, a file was over 50 MB and only its end is in the archive: please send that file separately.
