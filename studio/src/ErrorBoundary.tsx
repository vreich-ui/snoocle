import { Component, type ErrorInfo, type ReactNode } from "react";

interface ErrorBoundaryProps {
  /** Changing this resets the boundary — navigating away from a broken record should not stay broken. */
  resetKey?: string;
  children: ReactNode;
}

interface ErrorBoundaryState {
  message: string;
}

/**
 * Without this, one unexpected field shape anywhere in a section unmounts the
 * whole app and the user sees a blank page with no way back.
 */
export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  state: ErrorBoundaryState = { message: "" };

  static getDerivedStateFromError(error: unknown): ErrorBoundaryState {
    return { message: error instanceof Error ? error.message : String(error) };
  }

  componentDidUpdate(previous: ErrorBoundaryProps): void {
    if (previous.resetKey !== this.props.resetKey && this.state.message) this.setState({ message: "" });
  }

  componentDidCatch(error: unknown, info: ErrorInfo): void {
    console.error("Studio section failed to render", error, info.componentStack);
  }

  render(): ReactNode {
    if (!this.state.message) return this.props.children;
    return (
      <section className="workspace" role="alert">
        <p className="eyebrow">Workspace</p>
        <h2>This section failed to render</h2>
        <p className="error">{this.state.message}</p>
        <p className="muted">The rest of Studio still works — switch sections, or reload the page.</p>
        <button type="button" onClick={() => this.setState({ message: "" })}>Try again</button>
      </section>
    );
  }
}
