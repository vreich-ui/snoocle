import { useEffect, useState } from "react";
import { apiJson, ApiError } from "./client";

/**
 * "idle" covers a path that was intentionally withheld — e.g. Runs shows no
 * detail until a run is selected. It is distinct from "unauthenticated": that
 * state says a token is required and missing, while "idle" makes no claim
 * about auth at all (a page can be authenticated and still have nothing
 * selected yet).
 */
export type ApiState = "idle" | "unauthenticated" | "loading" | "ready" | "error";

export interface ApiResult<T> {
  state: ApiState;
  data?: T;
  error?: ApiError;
  reload(): void;
}

export function useApi<T>(path: string | null, token: string): ApiResult<T> {
  const [state, setState] = useState<ApiState>(() => (!token ? "unauthenticated" : path === null ? "idle" : "loading"));
  const [data, setData] = useState<T>();
  const [error, setError] = useState<ApiError>();
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    let active = true;
    if (!token) {
      setState("unauthenticated");
      setData(undefined);
      setError(undefined);
      return () => {
        active = false;
      };
    }
    if (path === null) {
      setState("idle");
      setData(undefined);
      setError(undefined);
      return () => {
        active = false;
      };
    }
    setState("loading");
    setError(undefined);
    apiJson<T>(path).then((result) => {
      if (!active) return;
      setData(result);
      setState("ready");
    }).catch((err: unknown) => {
      if (!active) return;
      setError(err instanceof ApiError ? err : new ApiError(0, err instanceof Error ? err.message : String(err)));
      setState("error");
    });
    return () => {
      active = false;
    };
  }, [path, token, attempt]);

  return { state, data, error, reload: () => setAttempt((value) => value + 1) };
}
