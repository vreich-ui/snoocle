import { useMemo, useState } from "react";
import { type NeedsIdentityResponse, type SongsResponse, type SongSummary } from "./client";
import { songDetailPath } from "./navigation";
import { useApi } from "./useApi";

interface LibraryProps {
  token: string;
  onNavigate(path: string): void;
}

function matchesQuery(item: SongSummary, query: string): boolean {
  return `${item.title} ${item.artist} ${item.id}`.toLocaleLowerCase().includes(query);
}

export function Library({ token, onNavigate }: LibraryProps) {
  const [search, setSearch] = useState("");
  const songs = useApi<SongsResponse>("/v1/songs", token);
  const needsIdentity = useApi<NeedsIdentityResponse>("/v1/songs/needs-identity", token);

  const needsIdentityIds = useMemo(
    () => new Set((needsIdentity.data?.songs ?? []).map((entry) => entry.songId)),
    [needsIdentity.data],
  );

  const items = songs.data?.items ?? [];
  const query = search.trim().toLocaleLowerCase();
  const visible = query ? items.filter((item) => matchesQuery(item, query)) : items;

  return (
    <section className="workspace library" aria-labelledby="library-heading">
      <p className="eyebrow">Song browser</p>
      <h2 id="library-heading">Library</h2>

      {songs.state === "unauthenticated" && (
        <p className="muted">Paste the server's SNOOCLE_API_TOKEN into the sidebar to load the library.</p>
      )}

      {songs.state === "loading" && <p className="muted" role="status">Loading songs…</p>}

      {songs.state === "error" && (
        <div className="error" role="alert">
          <p>{songs.error?.detail}</p>
          <button type="button" onClick={songs.reload}>Retry</button>
        </div>
      )}

      {songs.state === "ready" && (
        items.length === 0 ? (
          <p className="muted">No songs in the store yet.</p>
        ) : (
          <>
            <label className="library-search">
              <span>Search</span>
              <input type="search" value={search} onChange={(event) => setSearch(event.target.value)} />
            </label>
            <p className="catalog-count">Showing {visible.length} of {items.length}</p>
            <div className="song-list">
              {visible.map((item) => (
                <button className="song-row" key={item.id} type="button" onClick={() => onNavigate(songDetailPath(item.id))}>
                  <strong>{item.title} — {item.artist}</strong>
                  <code>{item.id}</code>
                  <span className="song-row-meta">
                    <span>{new Date(item.updatedAt).toLocaleString()}</span>
                    {needsIdentityIds.has(item.id) && <span className="status-pill">Needs identity</span>}
                    {item.hasTiming && <span className="safety-badge safe">Timed</span>}
                  </span>
                </button>
              ))}
            </div>
          </>
        )
      )}
    </section>
  );
}
