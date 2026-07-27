# The analysis worker

Snoocle's server doesn't analyze songs. It holds a queue and hands jobs out; a
worker on your Mac claims them, runs them, and reports back.

You never see this. You add a song in the player, and it appears. The rest of
this document exists for when it doesn't.

## Why it works this way

Analysis is CPU-bound (MIR) and, once stems arrive, ML-shaped. Two facts pushed
the work off Cloud Run:

**Background work on Cloud Run is expensive out of proportion to itself.** Under
the default request-based billing, CPU is throttled to near-zero outside a
request — a background task doesn't merely risk being scaled away, it *stalls*.
Keeping one alive needs instance-based billing plus a warm instance
(`--no-cpu-throttling --min-instances=1`), which bills every second of the month
at the full rate: **~$53/month at 1 vCPU / 2 GiB, before a single song is
analyzed.**

**And the laptop is faster anyway.** Demucs separates a 7-minute song in ~12
seconds on Apple Silicon versus ~6 minutes on a Cloud Run vCPU. Paying $53 a
month to do the work 30× slower is a bad trade in both directions.

So the server brokers and the Mac works. Cloud Run stays at
`--min-instances=0`, request-based, and costs approximately nothing when nobody
is playing a song. The container never needs torch, so the image stays small and
cold starts stay quick.

Single-song analysis (`POST /v1/songs/analyze`, the "Add song" button) is
**unchanged** and still runs on the server. It completes inside its own request,
so it needs no worker and no always-on CPU — which means adding one song from
the iPad works instantly, anywhere, whether or not the Mac is awake.

## Setup

Double-click `scripts/install_worker.command`. It asks for the server URL and
your API token, builds a private virtualenv, and registers a launchd agent that
starts at login and stays running.

Prerequisites: Python 3.10+ and `ffmpeg` (`brew install ffmpeg`).

To check on it:

```sh
tail -f ~/Library/Logs/Snoocle/worker.log
launchctl print gui/$UID/com.snoocle.worker
```

To stop it permanently:

```sh
launchctl bootout gui/$UID/com.snoocle.worker
```

To run it once by hand, in the foreground, which is the fastest way to see why
a job is failing:

```sh
SNOOCLE_SERVER_URL=https://… SNOOCLE_API_TOKEN=… snoocle-worker --once -v
```

## The protocol

Every exchange is an HTTPS request the worker initiates. Nothing on the Mac
listens, no port is opened, no tunnel is needed, and the machine stays invisible
from the internet. The bearer token is the same one the admin UI uses.

| Call | Meaning |
|---|---|
| `POST /v1/queue/claim` | "Anything for me?" `204` means no — the ordinary case |
| `POST /v1/queue/{id}/heartbeat` | "Still working." Extends the lease |
| `POST /v1/queue/{id}/complete` | "Done, here's the song id" |
| `POST /v1/queue/{id}/fail` | "Broke. Worth retrying / not worth retrying" |

A claim is a **lease**, not an assignment — it expires after 5 minutes without a
heartbeat. That single decision is what makes closing the laptop mid-song a
non-event: the heartbeat stops, the lease expires, the job returns to the queue,
and any worker (including the same Mac when it wakes) picks it up. Nothing has
to notice the failure, so nothing can fail to notice it. Leases are reclaimed
lazily when the queue is read, so there is no cron, no scheduler, and no
always-on CPU anywhere in the system.

A worker that wakes up still holding a stale lease gets `409` on its next
heartbeat, drops the work in progress, and asks for something new — it never
races the machine that took over.

## Capabilities

A worker advertises what it can actually do, proven by imports rather than
configuration:

| Capability | Present when | Used for |
|---|---|---|
| `analyze` | always | the reconcile pipeline |
| `stems` | `demucs` imports | separation (Phase B4) |
| `align` | `whisperx` imports | forced alignment (Phase B2) |
| `melody` | `torchcrepe` imports | reference vocal melody (Phase H1) |

A job carries `wants`, and the broker only offers it to a worker that satisfies
all of them. So queueing a song with `wants: ["stems"]` on a Mac without demucs
leaves it queued rather than claimed-then-failed — and installing demucs later
makes it run with no other change.

## When something is wrong

**Jobs sit at "queued" and the dashboard says no worker has checked in.** The
Mac is asleep, offline, or the agent isn't running. Check the log.

**A job says "lease expired — worker stopped responding".** The Mac went to
sleep mid-analysis. It has already been requeued; it will run when the Mac is
back. After three of these the job parks as an error so the queue moves on, and
the Retry button gives it a fresh budget.

**Every job fails immediately.** Almost always `ffmpeg` missing, or a wrong
token. `snoocle-worker --once -v` in Terminal shows which within seconds.

**Two Macs.** Fine, and unplanned-for in no way: leases are atomic, so two
workers simply drain the queue faster.
