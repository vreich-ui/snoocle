import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ErrorBoundary } from "./ErrorBoundary";

function Boom(): never {
  throw new Error("field exploded");
}

describe("ErrorBoundary", () => {
  beforeEach(() => vi.spyOn(console, "error").mockImplementation(() => undefined));
  afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
  });

  it("shows the failure instead of unmounting the app", () => {
    render(<ErrorBoundary><Boom /></ErrorBoundary>);
    expect(screen.getByRole("alert")).toBeVisible();
    expect(screen.getByText("field exploded")).toBeVisible();
  });

  it("renders children when nothing throws", () => {
    render(<ErrorBoundary><p>fine</p></ErrorBoundary>);
    expect(screen.getByText("fine")).toBeVisible();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("clears when the reset key changes so navigating away recovers", () => {
    const view = render(<ErrorBoundary resetKey="a"><Boom /></ErrorBoundary>);
    expect(screen.getByRole("alert")).toBeVisible();
    view.rerender(<ErrorBoundary resetKey="b"><p>recovered</p></ErrorBoundary>);
    expect(screen.getByText("recovered")).toBeVisible();
  });
});
