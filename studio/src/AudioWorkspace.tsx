import { useEffect, useRef, useState } from "react";
import { apiFetch, getBearerToken } from "./api";
import { createWaveform, type WaveformController } from "./waveform";

type AudioArtifact = {
  audioRef: string;
  filename: string;
  contentType: string;
  durationSeconds: number;
  sizeBytes: number;
  expiresAt: string;
  playbackUrl: string;
};

async function artifactResponse(response: Response): Promise<AudioArtifact> {
  const body = await response.json();
  if (!response.ok) {
    throw new Error(body.detail ?? body.reason ?? `Request failed (${response.status})`);
  }
  return body.artifact as AudioArtifact;
}

export function AudioWorkspace() {
  const [file, setFile] = useState<File | null>(null);
  const [youtube, setYoutube] = useState("");
  const [artifact, setArtifact] = useState<AudioArtifact | null>(null);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const waveformElement = useRef<HTMLDivElement | null>(null);
  const waveform = useRef<WaveformController | null>(null);

  useEffect(() => {
    if (!artifact || !waveformElement.current) return;
    waveform.current?.destroy();
    waveform.current = createWaveform(
      waveformElement.current,
      artifact.playbackUrl,
      getBearerToken(),
    );
    return () => {
      waveform.current?.destroy();
      waveform.current = null;
    };
  }, [artifact]);

  const preview = (next: AudioArtifact) => {
    setArtifact(next);
    setMessage("");
  };

  const upload = async () => {
    if (!file) return;
    setBusy(true);
    setMessage("");
    try {
      const data = new FormData();
      data.set("file", file);
      preview(await artifactResponse(await apiFetch("/v1/audio/artifacts", { method: "POST", body: data })));
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Upload failed");
    } finally {
      setBusy(false);
    }
  };

  const acquire = async () => {
    if (!youtube.trim()) return;
    setBusy(true);
    setMessage("");
    try {
      preview(await artifactResponse(await apiFetch("/v1/audio/artifacts/acquire", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ youtubeUrlOrId: youtube.trim() }),
      })));
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Acquisition failed");
    } finally {
      setBusy(false);
    }
  };

  const remove = async () => {
    if (!artifact) return;
    setBusy(true);
    setMessage("");
    try {
      const response = await apiFetch(`/v1/audio/artifacts/${artifact.audioRef}`, { method: "DELETE" });
      if (!response.ok) throw new Error(`Delete failed (${response.status})`);
      waveform.current?.destroy();
      waveform.current = null;
      setArtifact(null);
      setMessage("Temporary audio deleted.");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Delete failed");
    } finally {
      setBusy(false);
    }
  };

  return (
    <section className="audio-workspace" aria-labelledby="audio-heading">
      <h3 id="audio-heading">Audio artifacts</h3>
      <p>Upload local audio or acquire a YouTube recording. References expire automatically and never reveal server paths.</p>
      <div className="audio-inputs">
        <label>
          <span>Local audio</span>
          <input
            aria-label="Local audio"
            type="file"
            accept="audio/mpeg,audio/mp4,audio/wav,audio/flac,audio/ogg,audio/webm,.opus"
            onChange={(event) => setFile(event.target.files?.[0] ?? null)}
          />
        </label>
        <button type="button" disabled={!file || busy} onClick={upload}>Upload and preview</button>
        <label>
          <span>YouTube URL or video ID</span>
          <input
            aria-label="YouTube URL or video ID"
            value={youtube}
            onChange={(event) => setYoutube(event.target.value)}
          />
        </label>
        <button type="button" disabled={!youtube.trim() || busy} onClick={acquire}>Acquire and preview</button>
      </div>
      {message && <p role="status">{message}</p>}
      {artifact && (
        <div className="audio-preview">
          <div>
            <strong>{artifact.filename}</strong>
            <span>{artifact.durationSeconds.toFixed(1)}s · {(artifact.sizeBytes / 1_000_000).toFixed(1)} MB</span>
          </div>
          <div ref={waveformElement} role="img" aria-label={`Waveform for ${artifact.filename}`} />
          <div className="audio-actions">
            <button type="button" onClick={() => waveform.current?.playPause()}>Play or pause</button>
            <button type="button" disabled={busy} onClick={remove}>Delete temporary audio</button>
          </div>
        </div>
      )}
    </section>
  );
}
