# Stems and forced alignment (B4, B2)

Two engines that both need a torch-sized dependency stack and minutes of CPU
per song, and therefore both run on a worker rather than in the Cloud Run
image.

## What they give you

**Stems (B4)** split a song into `vocals / drums / bass / other` with demucs,
then render two practice mixes with ffmpeg:

| mix | what it is |
| --- | --- |
| `backing_no_vocals` | drums + bass + other — sing over it |
| `backing_no_guitar` | vocals + drums + bass — play over it |

`backing_no_guitar` is an **approximation** with the default 4-stem model.
There is no guitar stem: guitar lives in `other`, along with keys, strings and
horns, so removing the guitar removes those too. The API says so in the mix's
`description` field rather than leaving you to discover it. `htdemucs_6s`
(`?model=htdemucs_6s`) emits a real guitar stem and makes the mix exact; it is
slower and marginally less clean on the other sources, which is why it is not
the default.

**Alignment (B2)** times known lyrics against the audio with WhisperX's
alignment-only path. It is not transcription — the words come from the song, so
the model is only asked *when*, never *what*. That matters because the lyrics
may already have been corrected by hand, and an ASR pass would throw that away.

Alignment prefers the vocals stem when one exists, because how audible the
voice is dominates the result. The response records which input was used
(`"source": "vocals" | "mix"`), so two runs are comparable.

## Endpoints

```
POST   /v1/songs/{id}/stems[?model=htdemucs&force=1]   202 queued (or 200 cached)
GET    /v1/songs/{id}/stems                            what exists
GET    /v1/songs/{id}/stems/{name}                     audio, Range supported
DELETE /v1/songs/{id}/stems                            reclaim the disk

POST   /v1/songs/{id}/align[?language=en]              202 queued
```

Serving needs nothing installed. The API process only reads files off disk,
which is the whole reason the split works: the Mac separates, the server (or
the Mac) serves.

## Installing the engines

On the machine that runs the worker:

```sh
pip install 'snoocle-server[stems]' --index-url https://download.pytorch.org/whl/cpu
pip install 'snoocle-server[align]' --index-url https://download.pytorch.org/whl/cpu
```

The `--index-url` is not optional unless you have CUDA and want it — without
it pip fetches a multi-gigabyte CUDA build of torch to run on a CPU.

Capabilities are detected by import at worker startup, so after installing
either extra the worker has to be restarted before the broker will hand it
those jobs:

```sh
launchctl kickstart -k gui/$UID/com.snoocle.worker
```

## The honest limitation

`SNOOCLE_STEMS_DIR` defaults to `data/stems`, which on Cloud Run is **tmpfs —
RAM, wiped on restart**. Stems produced by a worker live on that worker's disk
and are served by whatever process can see them. There is deliberately no
upload path to the cloud service yet: a four-minute song is roughly 150 MB of
WAV across stems and mixes, so shipping them to Cloud Run means a blob store
(GCS) and a bill, which is a decision to make deliberately rather than a
default to inherit.

Until that is decided, the working arrangement is: run the worker on the Mac,
and reach the stems from the same machine. `F7` (the iOS stems player) is what
forces the question, and it is the right moment to answer it.
