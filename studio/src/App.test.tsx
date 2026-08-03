import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { StudioApp } from "./App";

describe("StudioApp", () => {
  afterEach(cleanup);

  beforeEach(() => {
    window.history.replaceState({}, "", "/studio/");
    window.sessionStorage.clear();
  });

  it("provides every Studio section", () => {
    render(<StudioApp />);
    for (const label of ["Repair", "Build", "Automatic Pipeline", "Tool Studio", "Library", "Runs", "Evaluation", "Configuration"]) {
      expect(screen.getByRole("button", { name: label })).toBeVisible();
    }
  });

  it("supports keyboard navigation with native buttons", async () => {
    render(<StudioApp />);
    const runs = screen.getByRole("button", { name: "Runs" });
    runs.focus();
    await userEvent.keyboard("{Enter}");
    expect(window.location.pathname).toBe("/studio/runs");
    expect(screen.getByRole("heading", { name: "Runs" })).toBeVisible();
    expect(runs).toHaveAttribute("aria-current", "page");
  });

  it("keeps a entered bearer token in session storage only", () => {
    render(<StudioApp />);
    fireEvent.change(screen.getByLabelText("Bearer token"), { target: { value: "temporary-token" } });
    expect(window.sessionStorage.getItem("snoocle.studio.bearer-token")).toBe("temporary-token");
    expect(window.localStorage.getItem("snoocle.studio.bearer-token")).toBeNull();
  });
});
