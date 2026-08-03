import { lazy, Suspense, useEffect, useState } from "react";
import { getBearerToken, saveBearerToken } from "./api";
import { sectionFromPath, sectionPath, studioSections, type StudioSection } from "./navigation";
import "./studio.css";

const ToolStudio = lazy(() => import("./ToolStudio").then((module) => ({ default: module.ToolStudio })));

function useCurrentSection() {
  const [section, setSection] = useState<StudioSection>(() => sectionFromPath(window.location.pathname));

  useEffect(() => {
    const sync = () => setSection(sectionFromPath(window.location.pathname));
    window.addEventListener("popstate", sync);
    return () => window.removeEventListener("popstate", sync);
  }, []);

  const navigate = (next: StudioSection) => {
    window.history.pushState({}, "", sectionPath(next));
    setSection(next);
  };

  return { section, navigate };
}

export function StudioApp() {
  const { section, navigate } = useCurrentSection();
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
            className={section === item ? "selected" : ""}
            key={item}
            onClick={() => navigate(item)}
            type="button"
          >
            {item}
          </button>
        ))}
      </nav>
      {section === "Tool Studio" ? (
        <Suspense fallback={<section className="workspace" role="status">Loading Tool Studio…</section>}>
          <ToolStudio token={token} />
        </Suspense>
      ) : (
        <section aria-labelledby="section-heading" className="workspace" tabIndex={-1}>
          <p className="eyebrow">Workspace</p>
          <h2 id="section-heading">{section}</h2>
          <p>Connect your Snoocle workflow here. API requests use the bearer token from this tab only.</p>
        </section>
      )}
    </main>
  );
}
