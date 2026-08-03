const TOKEN_KEY = "snoocle.studio.bearer-token";

/** The bearer token lives only for this browser tab, never in the bundle or URL. */
export function getBearerToken(): string {
  return window.sessionStorage.getItem(TOKEN_KEY) ?? "";
}

export function saveBearerToken(token: string): void {
  if (token) window.sessionStorage.setItem(TOKEN_KEY, token);
  else window.sessionStorage.removeItem(TOKEN_KEY);
}

export async function apiFetch(path: string, init: RequestInit = {}): Promise<Response> {
  const token = getBearerToken();
  const headers = new Headers(init.headers);
  if (token) headers.set("Authorization", `Bearer ${token}`);
  return fetch(path, { ...init, headers });
}
