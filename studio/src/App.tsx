import { lazy, Suspense, useEffect, useState } from "react";
import { getBearerToken, saveBearerToken } from "./api";
import { AudioWorkspace } from "./AudioWorkspace";
import { ErrorBoundary } from "./ErrorBoundary";
import { isImplemented, routeFromPath, sectionPath, sectionPlan, studioSections, type StudioRoute, type StudioSection } from "./navigation";
import "./studio.css";

const ToolStudio = lazy(() => import("./ToolStudio").then((module) => ({ default: module.ToolStudio })));
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

  const onTokenChange = (value: string) => {
    setToken(value);
    saveBearerToken(value);
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
      {section === "Tool Studio" ? (
        <div className="tool-studio-page">
          <div className="workspace audio-workspace-shell">
            <AudioWorkspace />
          </div>
          <Suspense fallback={<section className="workspace" role="status">Loading Tool Studio…</section>}>
            <ToolStudio token={token} />
          </Suspense>
        </div>
      ) : section === "Library" ? (
        <Suspense fallback={<section className="workspace" role="status">Loading Library…</section>}>
          {detailId
            ? <SongDetail songId={detailId} token={token} onNavigate={navigate} />
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
