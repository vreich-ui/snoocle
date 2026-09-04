import { lazy, Suspense, useEffect, useState } from "react";
import { getBearerToken, saveBearerToken } from "./api";
import { AudioWorkspace } from "./AudioWorkspace";
import { ErrorBoundary } from "./ErrorBoundary";
import { isImplemented, routeFromPath, sectionPath, sectionPlan, studioSections, type StudioRoute, type StudioSection } from "./navigation";
import { WorkbenchBar } from "./Workbench";
import { loadWorkbench, saveWorkbench, type Workbench } from "./workbench";
import "./studio.css";

const ToolStudio = lazy(() => import("./ToolStudio").then((module) => ({ default: module.ToolStudio })));
const SongStudio = lazy(() => import("./SongStudio").then((module) => ({ default: module.SongStudio })));
const Library = lazy(() => import("./Library").then((module) => ({ default: module.Library })));
const SongDetail = lazy(() => import("./SongDetail").then((module) => ({ default: module.SongDetail })));
const Runs = lazy(() => import("./Runs").then((module) => ({ default: module.Runs })));

function useCurrentRoute() {
  const [route, setRoute] = useState<StudioRoute>(() => routeFromPath(window.location.pathname));

  useEffect(() => {
    const sync = () => setRoute(routeFromPath(window.location.pathname));
    window.addEventListener("popstate", sync);
    return () => window.removeEventListener("popstate", sync);
  }, []);

  // Song/run detail routes are not section paths, so navigate() takes an
  // arbitrary path (a back button, a table row) rather than a StudioSection.
  const navigate = (path: string) => {
    window.history.pushState({}, "", path);
    setRoute(routeFromPath(path));
  };

  const navigateSection = (section: StudioSection) => navigate(sectionPath(section));

  return { route, navigate, navigateSection };
}

export function StudioApp() {
  const { route, navigate, navigateSection } = useCurrentRoute();
  const { section, detailId } = route;
  const [token, setToken] = useState(getBearerToken);
  const [bench, setBench] = useState<Workbench>(loadWorkbench);

  const onTokenChange = (value: string) => {
    setToken(value);
    saveBearerToken(value);
  };

  const onBenchChange = (next: Workbench) => {
    setBench(next);
    saveWorkbench(next);
  };

  return (
    <main className="studio-shell">
      <header>
        <p className="eyebrow">Snoocle</p>
        <h1>Studio</h1>
        <label className="token-field">
          <span>Bearer token</span>
          <input
            aria-label="Bearer token"
            type="password"
            autoComplete="off"
            value={token}
            onChange={(event) => onTokenChange(event.target.value)}
          />
        </label>
      </header>
      <nav aria-label="Studio sections">
        {studioSections.map((item) => (
          <button
            aria-current={section === item ? "page" : undefined}
            className={[section === item ? "selected" : "", isImplemented(item) ? "" : "unbuilt"].filter(Boolean).join(" ")}
            key={item}
            onClick={() => navigateSection(item)}
            title={isImplemented(item) ? undefined : `${item} — not built yet`}
            type="button"
          >
            {item}
          </button>
        ))}
      </nav>
      <ErrorBoundary resetKey={`${section}:${detailId ?? ""}`}>
      {section === "Song Studio" ? (
        <Suspense fallback={<section className="workspace" role="status">Loading Song Studio…</section>}>
          <SongStudio songId={detailId} token={token} bench={bench} onBenchChange={onBenchChange} onNavigate={navigate} />
        </Suspense>
      ) : section === "Tool Studio" ? (
        <div className="tool-studio-page">
          <div className="workspace audio-workspace-shell">
            <AudioWorkspace
              onArtifact={(artifact) => onBenchChange({
                ...bench,
                audio: {
                  audioRef: artifact.audioRef,
                  filename: artifact.filename,
                  durationSeconds: artifact.durationSeconds,
                  // Stamped with the song that was selected at the time, and
                  // with what the server says it fetched, so a later step can
                  // tell whether this recording belongs to the song on screen.
                  songId: bench.song?.id,
                  youtubeVideoId: artifact.youtubeVideoId,
                  videoTitle: artifact.videoTitle,
                },
              })}
              onArtifactRemoved={(audioRef) => {
                // The workbench must not keep pointing at a reference the
                // server no longer has.
                if (bench.audio?.audioRef === audioRef) onBenchChange({ ...bench, audio: undefined });
              }}
            />
          </div>
          <WorkbenchBar bench={bench} token={token} onChange={onBenchChange} />
          <Suspense fallback={<section className="workspace" role="status">Loading Tool Studio…</section>}>
            <ToolStudio token={token} bench={bench} onBenchChange={onBenchChange} />
          </Suspense>
        </div>
      ) : section === "Library" ? (
        <Suspense fallback={<section className="workspace" role="status">Loading Library…</section>}>
          {detailId
            ? (
              <SongDetail
                songId={detailId}
                token={token}
                onNavigate={navigate}
                onSendToToolStudio={(song) => onBenchChange({ ...bench, song })}
              />
            )
            : <Library token={token} onNavigate={navigate} />}
        </Suspense>
      ) : section === "Runs" ? (
        <Suspense fallback={<section className="workspace" role="status">Loading Runs…</section>}>
          <Runs token={token} detailId={detailId} onNavigate={navigate} />
        </Suspense>
      ) : (
        <section aria-labelledby="section-heading" className="workspace" tabIndex={-1}>
          <p className="eyebrow">Workspace</p>
          <h2 id="section-heading">{section}</h2>
          <span className="status-pill">Not built yet</span>
          <p>{sectionPlan[section]}</p>
          <p className="muted">Until this section is built, the working equivalent lives in the existing admin at <a href="/ui/">/ui/</a>.</p>
        </section>
      )}
      </ErrorBoundary>
    </main>
  );
}
