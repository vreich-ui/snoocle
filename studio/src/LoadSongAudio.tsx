import { useState } from "react";
import { acquireArtifact } from "./audio";
import { isYouTubeAuthFailure } from "./youtube";
import { YouTubeAuthNotice } from "./YouTubeAuthNotice";
import { type Workbench } from "./workbench";

interface LoadSongAudioProps {
  bench: Workbench;
  token: string;
  onChange(next: Workbench): void;
}

/**
 * The missing link between "a song is selected" and "the audio steps can run".
 *
 * The store already records which recording a song was built from, so asking
 * the operator to find that video id, open another section and paste it back
 * was work the app could do itself — and work that invited pasting the wrong
 * one. Nothing appears when the song has no known recording; there is nothing
 * honest to offer in that case.
 */
export function LoadSongAudio({ bench, token, onChange }: LoadSongAudioProps) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [authBlocked, setAuthBlocked] = useState(false);
  const song = bench.song;
  const videoId = song?.youtubeVideoId;
  if (!song || !videoId) return null;

  const alreadyLoaded = bench.audio?.youtubeVideoId === videoId && bench.audio.songId === song.id;

  const load = async () => {
    setBusy(true);
    setError("");
    setAuthBlocked(false);
    try {
      const artifact = await acquireArtifact(videoId);
      onChange({
        ...bench,
        audio: {
          audioRef: artifact.audioRef,
          filename: artifact.filename,
          durationSeconds: artifact.durationSeconds,
          songId: song.id,
          youtubeVideoId: artifact.youtubeVideoId,
          videoTitle: artifact.videoTitle,
        },
      });
    } catch (caught) {
      setAuthBlocked(isYouTubeAuthFailure(caught));
      setError(caught instanceof Error ? caught.message : "Could not load this song's audio");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="load-song-audio">
      <div className="form-actions">
        <button type="button" disabled={busy || alreadyLoaded} onClick={load}>
          {busy ? "Loading audio…" : alreadyLoaded ? "Audio loaded" : "Load this song's audio"}
        </button>
      </div>
      <p className="field-help">
        {alreadyLoaded
          ? `Using the song's own recording (YouTube ${videoId}).`
          : `Fetches YouTube ${videoId}, the recording on file for this song.`}
      </p>
      {authBlocked
        ? <YouTubeAuthNotice token={token} detail={error} onReconnected={() => setAuthBlocked(false)} />
        : error && <p className="error" role="alert">{error}</p>}
    </div>
  );
}
