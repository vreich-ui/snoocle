import { useCallback, useEffect, useState } from "react";
import { ApiError } from "./client";
import { formatDateTime } from "./format";
import {
  clearYouTubeCookies,
  countCookieLines,
  fetchYouTubeSession,
  storeYouTubeCookies,
  type YouTubeSessionStatus,
} from "./youtube";

interface YouTubeSessionProps {
  token: string;
  /** Called after a successful upload, so a blocked page can retry without a reload. */
  onReconnected?(): void;
}

type LoadState = "idle" | "loading" | "ready" | "error" | "unconfigured";

/**
 * Manages the YouTube session cookies the server uses for downloads.
 *
 * YouTube bot-checks datacenter IPs, so a Cloud Run deployment cannot fetch
 * anything without a signed-in session. The server has held these routes since
 * before Studio existed; there was simply no way to reach them from a browser,
 * which left the whole pipeline stuck behind an error that named its own fix.
 *
 * The cookies are write-only here on purpose: they are session credentials, so
 * this shows how many lines are stored and when, never their contents.
 */
export function YouTubeSession({ token, onReconnected }: YouTubeSessionProps) {
  const [state, setState] = useState<LoadState>("idle");
  const [status, setStatus] = useState<YouTubeSessionStatus>();
  const [loadError, setLoadError] = useState("");
  const [cookiesTxt, setCookiesTxt] = useState("");
  const [busy, setBusy] = useState(false);
  const [actionError, setActionError] = useState("");
  const [notice, setNotice] = useState("");

  const load = useCallback(async () => {
    if (!token) {
      setState("idle");
      return;
    }
    setState("loading");
    setLoadError("");
    try {
      setStatus(await fetchYouTubeSession());
      setState("ready");
    } catch (error) {
      // 409 is the server refusing to manage session cookies while it is
      // itself unauthenticated. That is a deployment fact, not a failure.
      if (error instanceof ApiError && error.status === 409) {
        setLoadError(error.detail);
        setState("unconfigured");
        return;
      }
      setLoadError(error instanceof ApiError ? error.detail : error instanceof Error ? error.message : String(error));
      setState("error");
    }
  }, [token]);

  useEffect(() => {
    void load();
  }, [load]);

  const lineCount = countCookieLines(cookiesTxt);

  const upload = async () => {
    setBusy(true);
    setActionError("");
    setNotice("");
    try {
      await storeYouTubeCookies(cookiesTxt);
      setCookiesTxt("");
      setNotice("YouTube session stored. Retry the download.");
      await load();
      onReconnected?.();
    } catch (error) {
      setActionError(error instanceof ApiError ? error.detail : error instanceof Error ? error.message : String(error));
    } finally {
      setBusy(false);
    }
  };

  const clear = async () => {
    setBusy(true);
    setActionError("");
    setNotice("");
    try {
      await clearYouTubeCookies();
      setNotice("Stored YouTube session cleared.");
      await load();
    } catch (error) {
      setActionError(error instanceof ApiError ? error.detail : error instanceof Error ? error.message : String(error));
    } finally {
      setBusy(false);
    }
  };

  return (
    <section className="youtube-session" aria-labelledby="youtube-session-heading">
      <p className="eyebrow">YouTube session</p>
      <h3 id="youtube-session-heading">YouTube session</h3>
      <p className="muted">
        YouTube blocks downloads from datacenter addresses unless the request carries a signed-in
        session, so the server needs cookies from a browser that is logged in. Export them with a
        cookies.txt extension while signed in to YouTube, then paste the file below.
      </p>

      {state === "idle" && <p className="muted">Paste the server's SNOOCLE_API_TOKEN into the sidebar to manage the session.</p>}
      {state === "loading" && <p className="muted" role="status">Checking the stored session…</p>}
      {state === "unconfigured" && (
        <div className="warning" role="note">
          <strong>The server will not hold session cookies while it is unauthenticated.</strong>
          <p>{loadError}</p>
        </div>
      )}
      {state === "error" && (
        <div className="error" role="alert">
          <p>{loadError}</p>
          <button type="button" onClick={() => void load()}>Retry</button>
        </div>
      )}

      {state === "ready" && status && (
        <dl className="classification-grid">
          <div><dt>Status</dt><dd>{status.configured ? "Connected" : "Not connected"}</dd></div>
          <div><dt>Stored</dt><dd>{status.updatedAt ? formatDateTime(status.updatedAt) : "—"}</dd></div>
          <div><dt>Source</dt><dd>{status.source ?? "—"}</dd></div>
          <div><dt>Cookie lines</dt><dd>{status.lineCount ?? "—"}</dd></div>
        </dl>
      )}

      {(state === "ready" || state === "unconfigured") && (
        <>
          <label className="schema-field" htmlFor="youtube-cookies">
            <span>cookies.txt</span>
            <textarea
              id="youtube-cookies"
              aria-label="cookies.txt"
              disabled={busy}
              rows={8}
              placeholder="# Netscape HTTP Cookie File"
              value={cookiesTxt}
              onChange={(event) => setCookiesTxt(event.target.value)}
            />
          </label>
          <p className="field-help">
            {cookiesTxt.trim()
              ? `${lineCount} cookie ${lineCount === 1 ? "line" : "lines"} detected.`
              : "The contents are stored server-side and never shown again."}
          </p>
          {notice && <p role="status">{notice}</p>}
          {actionError && <p className="error" role="alert">{actionError}</p>}
          <div className="form-actions">
            <button type="button" disabled={busy || lineCount === 0} onClick={() => void upload()}>
              {busy ? "Working…" : "Store session"}
            </button>
            {status?.configured && (
              <button type="button" disabled={busy} onClick={() => void clear()}>Clear stored session</button>
            )}
          </div>
        </>
      )}
    </section>
  );
}
