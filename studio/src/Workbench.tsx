import { useState } from "react";
import { type SongsResponse, type SongSummary, type VersionsResponse } from "./client";
import { formatDateTime } from "./format";
import { useApi } from "./useApi";
import { LoadSongAudio } from "./LoadSongAudio";
import { benchMismatches, type Workbench as WorkbenchState } from "./workbench";

interface WorkbenchProps {
  bench: WorkbenchState;
  token: string;
  onChange(next: WorkbenchState): void;
}

function matchesQuery(item: SongSummary, query: string): boolean {
  return `${item.title} ${item.artist} ${item.id}`.toLocaleLowerCase().includes(query);
}

function formatDuration(seconds: number | undefined): string {
  if (seconds === undefined) return "unknown duration";
  const minutes = Math.floor(seconds / 60);
  const rest = Math.round(seconds % 60);
  return `${minutes}:${String(rest).padStart(2, "0")}`;
}

function mirByteLength(mirJson: string): number {
  return new TextEncoder().encode(mirJson).length;
}

export function WorkbenchBar({ bench, token, onChange }: WorkbenchProps) {
  const [pickerOpen, setPickerOpen] = useState(false);
  const [search, setSearch] = useState("");
  const songs = useApi<SongsResponse>(pickerOpen ? "/v1/songs" : null, token);
  const versions = useApi<VersionsResponse>(
    bench.song ? `/v1/songs/${encodeURIComponent(bench.song.id)}/versions` : null,
    token,
  );

  const items = songs.data?.items ?? [];
  const query = search.trim().toLocaleLowerCase();
  const visibleSongs = query ? items.filter((item) => matchesQuery(item, query)) : items;

  // A 404 here means the song has no recorded versions yet — that is not an
  // error state, it just leaves "Latest" as the only option.
  const noVersions = versions.state === "error" && versions.error?.status === 404;
  const versionList = noVersions ? [] : versions.data?.versions ?? [];

  const pickSong = (item: SongSummary) => {
    onChange({
      ...bench,
      song: {
        id: item.id,
        title: item.title,
        artist: item.artist,
        youtubeVideoId: item.youtubeVideoId ?? undefined,
      },
    });
    setPickerOpen(false);
    setSearch("");
  };

  const clearSong = () => onChange({ ...bench, song: undefined });
  const clearAudio = () => onChange({ ...bench, audio: undefined });
  const clearMir = () => onChange({ ...bench, mir: undefined });

  // Slots left over from a previous song are the quiet failure mode: nothing
  // errors, the wrong recording is simply used. Name them, and offer the one
  // action that fixes it.
  const mismatches = benchMismatches(bench);
  const dropStaleSlots = () => onChange({
    ...bench,
    audio: bench.audio?.songId && bench.audio.songId !== bench.song?.id ? undefined : bench.audio,
    mir: undefined,
  });

  return (
    <section className="workbench-bar" aria-label="Workbench">
      <p className="eyebrow">Workbench</p>
      {mismatches.length > 0 && (
        <div className="workbench-mismatch" role="alert">
          <strong>These slots do not belong to the selected song.</strong>
          <ul>{mismatches.map((problem) => <li key={problem}>{problem}</li>)}</ul>
          <div className="form-actions">
            <button type="button" onClick={dropStaleSlots}>Clear the mismatched slots</button>
          </div>
        </div>
      )}
      <div className="workbench-slots">
        <div className="tool-card workbench-slot">
          <p className="eyebrow">Song</p>
          {bench.song ? (
            <>
              <strong>{bench.song.title} — {bench.song.artist}</strong>
              <code>{bench.song.id}</code>
              <label>
                <span>Version</span>
                <select
                  aria-label="Song version"
                  value={bench.song.version ?? ""}
                  onChange={(event) => onChange({
                    ...bench,
                    song: { ...bench.song!, version: event.target.value || undefined },
                  })}
                >
                  <option value="">Latest</option>
                  {versionList.map((version) => (
                    <option key={version.version} value={version.version}>
                      {version.version.slice(0, 10)} · {formatDateTime(version.timestamp)}
                    </option>
                  ))}
                </select>
              </label>
              {bench.song.youtubeVideoId && (
                <span className="muted">recording YouTube {bench.song.youtubeVideoId}</span>
              )}
              <div className="form-actions">
                <button type="button" onClick={clearSong}>Clear</button>
              </div>
            </>
          ) : (
            <div className="form-actions">
              <button type="button" onClick={() => setPickerOpen(true)}>Pick song</button>
            </div>
          )}
        </div>

        <div className="tool-card workbench-slot">
          <p className="eyebrow">Audio</p>
          {bench.audio ? (
            <>
              <strong>{bench.audio.videoTitle ?? bench.audio.filename}</strong>
              <span className="muted">
                {formatDuration(bench.audio.durationSeconds)}
                {bench.audio.youtubeVideoId ? ` · YouTube ${bench.audio.youtubeVideoId}` : ""}
              </span>
              {bench.audio.videoTitle && <span className="muted">{bench.audio.filename}</span>}
              <div className="form-actions">
                <button type="button" onClick={clearAudio}>Clear</button>
              </div>
            </>
          ) : (
            <>
              <span className="muted">none</span>
              <LoadSongAudio bench={bench} onChange={onChange} />
            </>
          )}
        </div>

        <div className="tool-card workbench-slot">
          <p className="eyebrow">MIR</p>
          {bench.mir ? (
            <>
              <strong>Captured</strong>
              <span className="muted">{mirByteLength(bench.mir.json)} bytes</span>
              {bench.mir.audioRef && <span className="muted">from {bench.mir.audioRef}</span>}
              <div className="form-actions">
                <button type="button" onClick={clearMir}>Clear</button>
              </div>
            </>
          ) : <span className="muted">none</span>}
        </div>
      </div>

      {pickerOpen && (
        <div className="workbench-picker" role="region" aria-label="Pick a song">
          <label>
            <span>Search songs</span>
            <input type="search" value={search} onChange={(event) => setSearch(event.target.value)} />
          </label>
          {songs.state === "unauthenticated" && <p className="muted">Paste the server's SNOOCLE_API_TOKEN into the sidebar to load songs.</p>}
          {songs.state === "loading" && <p className="muted" role="status">Loading songs…</p>}
          {songs.state === "error" && (
            <div className="error" role="alert">
              <p>{songs.error?.detail}</p>
              <button type="button" onClick={songs.reload}>Retry</button>
            </div>
          )}
          {songs.state === "ready" && (
            <div className="song-list">
              {visibleSongs.map((item) => (
                <button className="song-row" key={item.id} type="button" onClick={() => pickSong(item)}>
                  <strong>{item.title} — {item.artist}</strong>
                  <code>{item.id}</code>
                </button>
              ))}
              {!visibleSongs.length && <p className="muted">No songs match.</p>}
            </div>
          )}
          <div className="form-actions">
            <button type="button" onClick={() => setPickerOpen(false)}>Close</button>
          </div>
        </div>
      )}
    </section>
  );
}
