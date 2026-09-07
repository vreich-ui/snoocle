import { useState } from "react";
import { YouTubeSession } from "./YouTubeSession";

interface YouTubeAuthNoticeProps {
  token: string;
  /** The server's own words, kept behind a disclosure rather than shown as the headline. */
  detail?: string;
  onReconnected?(): void;
}

/**
 * What a blocked download should look like: the state named, the fix offered
 * where the failure happened. The raw yt-dlp output is still available, but it
 * is the appendix rather than the message — it was previously the whole of it.
 */
export function YouTubeAuthNotice({ token, detail, onReconnected }: YouTubeAuthNoticeProps) {
  const [open, setOpen] = useState(false);

  return (
    <div className="youtube-auth-notice" role="alert">
      <strong>YouTube refused this download.</strong>
      <p>
        It is asking the server to prove it is not a bot, which happens to every request from a
        datacenter address without a signed-in session. Retrying will not help; the server needs
        fresh cookies.
      </p>
      <div className="form-actions">
        <button type="button" onClick={() => setOpen((value) => !value)}>
          {open ? "Hide session settings" : "Reconnect YouTube"}
        </button>
      </div>
      {open && <YouTubeSession token={token} onReconnected={onReconnected} />}
      {detail && <details><summary>What the server reported</summary><pre>{detail}</pre></details>}
    </div>
  );
}
