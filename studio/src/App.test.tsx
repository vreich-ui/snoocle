import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { StudioApp } from "./App";
import { sectionPlan } from "./navigation";

describe("StudioApp", () => {
  afterEach(cleanup);

  beforeEach(() => {
    window.history.replaceState({}, "", "/studio/repair");
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

  it("lands an unknown path and a bare /studio/ on Tool Studio", async () => {
    window.history.replaceState({}, "", "/studio/nope");
    render(<StudioApp />);
    expect(await screen.findByRole("heading", { name: "Tool Studio" })).toBeVisible();
    cleanup();

    window.history.replaceState({}, "", "/studio/");
    render(<StudioApp />);
    expect(await screen.findByRole("heading", { name: "Tool Studio" })).toBeVisible();
  });

  it("shows the Not built yet pill, its plan sentence and a link to /ui/ for a placeholder section", () => {
    render(<StudioApp />);
    expect(screen.getByRole("heading", { name: "Repair" })).toBeVisible();
    expect(screen.getByText("Not built yet")).toBeVisible();
    expect(screen.getByText(sectionPlan.Repair)).toBeVisible();
    const link = screen.getByRole("link", { name: "/ui/" });
    expect(link).toHaveAttribute("href", "/ui/");
  });
});
