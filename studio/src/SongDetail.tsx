import { useEffect, useRef, useState } from "react";
import { apiFetch } from "./api";
import {
  apiBlob,
  type GoldResponse,
  type IdentityRenameResponse,
  type NotesResponse,
  type Song,
  type SongRunsResponse,
  type VersionsResponse,
} from "./client";
import { runDetailPath, songDetailPath } from "./navigation";
import { Sheet } from "./Sheet";
import { formatCost, formatDateTime, orDash, pairOrDash } from "./format";
import { useApi } from "./useApi";
import { type WorkbenchSong } from "./workbench";

interface SongDetailProps {
  songId: string;
  token: string;
  onNavigate(path: string): void;
  onSendToToolStudio?(song: WorkbenchSong): void;
}

const EXPORT_EXTENSIONS = { chordpro: "cho", txt: "txt", json: "json" } as const;
type ExportFormat = keyof typeof EXPORT_EXTENSIONS;

function formatDuration(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined) return "unknown";
  const minutes = Math.floor(seconds / 60);
  const rest = Math.round(seconds % 60);
  return `${minutes}:${String(rest).padStart(2, "0")}`;
}

function triggerDownload(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

interface IdentityErrorBody {
  detail: string;
  errorCode?: string;
  missing?: string[];
}

async function submitIdentityRename(
  songId: string,
  artist: string,
  title: string,
): Promise<{ ok: true; data: IdentityRenameResponse } | { ok: false; status: number; body: IdentityErrorBody }> {
  const response = await apiFetch(`/v1/songs/${encodeURIComponent(songId)}/identity`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ artist, title }),
  });
  const body = await response.json();
  if (!response.ok) return { ok: false, status: response.status, body };
  return { ok: true, data: body as IdentityRenameResponse };
}

export function SongDetail({ songId, token, onNavigate, onSendToToolStudio }: SongDetailProps) {
  const song = useApi<Song>(`/v1/songs/${encodeURIComponent(songId)}`, token);
  const versions = useApi<VersionsResponse>(`/v1/songs/${encodeURIComponent(songId)}/versions`, token);
  const runs = useApi<SongRunsResponse>(`/v1/songs/${encodeURIComponent(songId)}/runs`, token);
  const gold = useApi<GoldResponse>(`/v1/songs/${encodeURIComponent(songId)}/gold`, token);
  const notes = useApi<NotesResponse>(`/v1/songs/${encodeURIComponent(songId)}/notes`, token);

  const [artistInput, setArtistInput] = useState("");
  const [titleInput, setTitleInput] = useState("");
  const [renameBusy, setRenameBusy] = useState(false);
  const [renameError, setRenameError] = useState("");
  const [renameMissing, setRenameMissing] = useState<string[]>([]);
  const [renameNotice, setRenameNotice] = useState("");

  // Seed the rename fields once per song. Keying this on song.data alone lets a
  // later re-resolve overwrite whatever the user has typed — which silently
  // reverts an edit in progress and leaves the Rename button disabled.
  const seededFor = useRef("");
  useEffect(() => {
    if (!song.data || seededFor.current === songId) return;
    seededFor.current = songId;
    setArtistInput(song.data.metadata.artist);
    setTitleInput(song.data.metadata.title);
  }, [song.data, songId]);

  // Distinct from versionA/versionB below: this is the one version (if any)
  // the user has deliberately pinned to send along to Tool Studio, not one of
  // the two diff comparators.
  const [pinnedVersion, setPinnedVersion] = useState("");
  const [versionA, setVersionA] = useState("");
  const [versionB, setVersionB] = useState("");
  const [diffText, setDiffText] = useState("");
  const [diffError, setDiffError] = useState("");
  const [diffBusy, setDiffBusy] = useState(false);

  useEffect(() => {
    const list = versions.data?.versions ?? [];
    if (list.length && !versionA && !versionB) {
      setVersionB(list[0].version);
      setVersionA((list[1] ?? list[0]).version);
    }
  }, [versions.data, versionA, versionB]);

  const [exportError, setExportError] = useState("");

  // Everything above is about one song. A rename navigates to the new id
  // without unmounting this component (App has no key on it), and a rename
  // rewrites every version hash — so the comparators kept values matching no
  // option, and the previous song's diff stayed on screen under the new
  // song's heading.
  useEffect(() => {
    setPinnedVersion("");
    setVersionA("");
    setVersionB("");
    setDiffText("");
    setDiffError("");
    setExportError("");
  }, [songId]);

  const noVersions = versions.state === "error" && versions.error?.status === 404;
  const versionList = noVersions ? [] : versions.data?.versions ?? [];

  const rename = async () => {
    setRenameBusy(true);
    setRenameError("");
    setRenameMissing([]);
    try {
      const result = await submitIdentityRename(songId, artistInput.trim(), titleInput.trim());
      if (!result.ok) {
        if (result.body.errorCode === "identity_unresolved") setRenameMissing(result.body.missing ?? []);
        else setRenameError(result.body.detail);
        return;
      }
      const { versionMap, migratedRunIds, songId: newSongId } = result.data;
      setRenameNotice(
        `Renamed to ${newSongId}: moved ${Object.keys(versionMap).length} version(s) and ${migratedRunIds.length} run(s).`,
      );
      onNavigate(songDetailPath(newSongId));
    } catch (error) {
      setRenameError(error instanceof Error ? error.message : String(error));
    } finally {
      setRenameBusy(false);
    }
  };

  const loadDiff = async () => {
    if (!versionA || !versionB) return;
    setDiffBusy(true);
    setDiffError("");
    try {
      const response = await apiFetch(
        `/v1/songs/${encodeURIComponent(songId)}/diff?a=${encodeURIComponent(versionA)}&b=${encodeURIComponent(versionB)}`,
      );
      const text = await response.text();
      if (!response.ok) throw new Error(text || `Request failed (${response.status})`);
      setDiffText(text);
    } catch (error) {
      setDiffError(error instanceof Error ? error.message : String(error));
    } finally {
      setDiffBusy(false);
    }
  };

  const exportSong = async (format: ExportFormat) => {
    setExportError("");
    try {
      const blob = await apiBlob(`/v1/songs/${encodeURIComponent(songId)}/export?format=${format}`);
      triggerDownload(blob, `${songId}.${EXPORT_EXTENSIONS[format]}`);
    } catch (error) {
      setExportError(error instanceof Error ? error.message : String(error));
    }
  };

  if (song.state === "unauthenticated") {
    return (
      <section className="workspace" aria-labelledby="song-detail-heading">
        <h2 id="song-detail-heading">Song</h2>
        <p className="muted">Paste the server's SNOOCLE_API_TOKEN into the sidebar to load this song.</p>
      </section>
    );
  }

  if (song.state === "loading" || song.state === "idle") {
    return (
      <section className="workspace" aria-labelledby="song-detail-heading">
        <h2 id="song-detail-heading">Song</h2>
        <p className="muted" role="status">Loading song…</p>
      </section>
    );
  }

  if (song.state === "error" || !song.data) {
    return (
      <section className="workspace" aria-labelledby="song-detail-heading">
        <h2 id="song-detail-heading">Song</h2>
        <div className="error" role="alert">
          <p>{song.error?.detail}</p>
          <button type="button" onClick={song.reload}>Retry</button>
        </div>
      </section>
    );
  }

  const data = song.data;
  const latestVersion = versionList[0]?.version;
  const isGold = gold.data?.goldVersion !== null && gold.data?.goldVersion !== undefined &&
    gold.data.goldVersion === latestVersion;
  const renameUnchanged = artistInput === data.metadata.artist && titleInput === data.metadata.title;

  return (
    <section className="workspace song-detail" aria-labelledby="song-detail-heading">
      <button type="button" className="back-link" onClick={() => onNavigate("/studio/library")}>← Back to Library</button>
      <p className="eyebrow">Song</p>
      <h2 id="song-detail-heading">{data.metadata.title} — {data.metadata.artist}</h2>
      <code>{data.id}</code>
      {onSendToToolStudio && (
        <div className="form-actions">
          <button
            type="button"
            onClick={() => {
              onSendToToolStudio({
                id: data.id,
                title: data.metadata.title,
                artist: data.metadata.artist,
                version: pinnedVersion || undefined,
                // Without this the workbench knows the song but not the
                // recording it came from, and every audio tool has to be told
                // by hand what the store already knows.
                youtubeVideoId: data.audio.youtubeVideoId ?? data.audio.analyzedVideoId ?? undefined,
              });
              onNavigate("/studio/tool-studio");
            }}
          >Send to Tool Studio</button>
        </div>
      )}
      <dl className="classification-grid song-meta">
        <div><dt>Key</dt><dd>{data.metadata.key ?? "unknown"}</dd></div>
        <div><dt>BPM</dt><dd>{data.metadata.bpm ?? "unknown"}</dd></div>
        <div><dt>Time signature</dt><dd>{data.metadata.timeSignature ?? "unknown"}</dd></div>
        <div><dt>Capo</dt><dd>{data.displayPreferences.capo}</dd></div>
        <div><dt>Duration</dt><dd>{formatDuration(data.audio.durationSeconds)}</dd></div>
        <div><dt>YouTube video</dt><dd>{data.audio.youtubeVideoId ?? "none"}</dd></div>
      </dl>
      <p>
        {isGold && <span className="status-pill">Gold</span>}
        {data.testOnly && <span className="status-pill">Test only</span>}
      </p>

      <section aria-labelledby="rename-heading">
        <h3 id="rename-heading">Rename</h3>
        <p className="warning" role="note">Renaming changes the song id and rewrites its versions and runs.</p>
        {renameNotice && <p className="muted">{renameNotice}</p>}
        {renameMissing.length > 0 && (
          <p className="error" role="alert">Missing: {renameMissing.join(", ")}</p>
        )}
        {renameError && <p className="error" role="alert">{renameError}</p>}
        <label>
          <span>Artist</span>
          <input value={artistInput} onChange={(event) => setArtistInput(event.target.value)} />
        </label>
        <label>
          <span>Title</span>
          <input value={titleInput} onChange={(event) => setTitleInput(event.target.value)} />
        </label>
        <button type="button" disabled={renameBusy || renameUnchanged} onClick={rename}>Rename song</button>
      </section>

      <section aria-labelledby="versions-heading">
        <h3 id="versions-heading">Versions</h3>
        {versions.state === "error" && !noVersions && (
          <div className="error" role="alert">
            <p>{versions.error?.detail}</p>
            <button type="button" onClick={versions.reload}>Retry</button>
          </div>
        )}
        {versionList.length === 0 ? (
          <p className="muted">No versions yet.</p>
        ) : (
          <>
            <div className="table-wrap">
              <table>
                <thead><tr><th>Version</th><th>Timestamp</th><th>Message</th></tr></thead>
                <tbody>
                  {versionList.map((version) => (
                    <tr key={version.version}>
                      <td><code>{version.version.slice(0, 10)}</code></td>
                      <td>{formatDateTime(version.timestamp)}</td>
                      <td>{version.message}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {onSendToToolStudio && (
              <label>
                <span>Version to send</span>
                <select value={pinnedVersion} onChange={(event) => setPinnedVersion(event.target.value)}>
                  <option value="">Latest</option>
                  {versionList.map((version) => (
                    <option key={version.version} value={version.version}>{version.version.slice(0, 10)}</option>
                  ))}
                </select>
              </label>
            )}
            <div className="diff-controls">
              <label>
                <span>A</span>
                <select value={versionA} onChange={(event) => setVersionA(event.target.value)}>
                  {versionList.map((version) => (
                    <option key={version.version} value={version.version}>{version.version.slice(0, 10)}</option>
                  ))}
                </select>
              </label>
              <label>
                <span>B</span>
                <select value={versionB} onChange={(event) => setVersionB(event.target.value)}>
                  {versionList.map((version) => (
                    <option key={version.version} value={version.version}>{version.version.slice(0, 10)}</option>
                  ))}
                </select>
              </label>
              <button type="button" disabled={diffBusy} onClick={loadDiff}>Show diff</button>
            </div>
            {diffError && <p className="error" role="alert">{diffError}</p>}
            {diffText && <pre className="diff-output">{diffText}</pre>}
          </>
        )}
      </section>

      <section aria-labelledby="runs-heading">
        <h3 id="runs-heading">Runs</h3>
        {runs.state === "error" && (
          <div className="error" role="alert">
            <p>{runs.error?.detail}</p>
            <button type="button" onClick={runs.reload}>Retry</button>
          </div>
        )}
        {(runs.data?.runs.length ?? 0) === 0 ? (
          <p className="muted">No runs yet.</p>
        ) : (
          <div className="table-wrap">
            <table>
              <thead>
                <tr><th>Run</th><th>Status</th><th>Provider / model</th><th>Started</th><th>Steps</th><th>Cost</th></tr>
              </thead>
              <tbody>
                {runs.data?.runs.map((run) => (
                  <tr key={run.runId}>
                    <td>
                      <button
                        type="button"
                        className="row-open"
                        onClick={() => onNavigate(runDetailPath(run.runId))}
                      >
                        <code>{run.runId.slice(0, 10)}</code>
                      </button>
                    </td>
                    <td>{run.status}</td>
                    <td>{pairOrDash(run.provider, run.model)}</td>
                    <td>{formatDateTime(run.startedAt)}</td>
                    <td>{orDash(run.stepCount)}</td>
                    <td>{formatCost(run.costUSD)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      <section aria-labelledby="notes-heading">
        <h3 id="notes-heading">Notes</h3>
        {!notes.data?.preference && !notes.data?.correction ? (
          <p className="muted">No notes.</p>
        ) : (
          <>
            {notes.data?.preference && (
              <div><strong>Preference</strong><p>{notes.data.preference.notes}</p></div>
            )}
            {notes.data?.correction && (
              <div><strong>Correction</strong><p>{notes.data.correction.notes}</p></div>
            )}
          </>
        )}
      </section>

      <section aria-labelledby="sheet-heading">
        <h3 id="sheet-heading">Sheet</h3>
        <Sheet song={data} />
      </section>

      <section aria-labelledby="export-heading">
        <h3 id="export-heading">Exports</h3>
        {exportError && <p className="error" role="alert">{exportError}</p>}
        <div className="form-actions">
          <button type="button" onClick={() => exportSong("chordpro")}>Export chordpro</button>
          <button type="button" onClick={() => exportSong("txt")}>Export txt</button>
          <button type="button" onClick={() => exportSong("json")}>Export json</button>
        </div>
      </section>
    </section>
  );
}
