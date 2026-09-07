import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { SchemaForm, type JsonSchema } from "./SchemaForm";

const schema: JsonSchema = {
  type: "object",
  properties: {
    title: { type: "string", description: "Song title" },
    attempts: { type: "integer", minimum: 1, default: 2 },
    persist: { type: "boolean" },
    tags: { type: "array", items: { type: "string" } },
  },
  required: ["title"],
};

describe("SchemaForm", () => {
  afterEach(cleanup);

  it("generates typed controls and JSON editors entirely from inputSchema", async () => {
    const onSubmit = vi.fn();
    render(<SchemaForm schema={schema} busy={false} requiresConfirmation={false} onSubmit={onSubmit} onCancel={vi.fn()} />);
    await userEvent.type(screen.getByLabelText(/Title/), "Let It Be");
    expect(screen.getByLabelText("Attempts")).toHaveValue(2);
    await userEvent.selectOptions(screen.getByLabelText("Persist"), "true");
    fireEvent.change(screen.getByLabelText("Tags"), { target: { value: '["classic"]' } });
    await userEvent.click(screen.getByRole("button", { name: "Invoke tool" }));
    expect(onSubmit).toHaveBeenCalledWith({ title: "Let It Be", attempts: 2, persist: true, tags: ["classic"] });
  });

  it("supports raw JSON advanced mode and rejects non-object or invalid input", async () => {
    const onSubmit = vi.fn();
    render(<SchemaForm schema={schema} busy={false} requiresConfirmation={false} onSubmit={onSubmit} onCancel={vi.fn()} />);
    await userEvent.click(screen.getByRole("button", { name: "Raw JSON" }));
    const editor = screen.getByLabelText("Arguments JSON");
    await userEvent.clear(editor);
    fireEvent.change(editor, { target: { value: "[]" } });
    await userEvent.click(screen.getByRole("button", { name: "Invoke tool" }));
    expect(screen.getByRole("alert")).toHaveTextContent("Arguments must be a JSON object");
    expect(onSubmit).not.toHaveBeenCalled();
    fireEvent.change(editor, { target: { value: '{"title":"Across the Universe"}' } });
    await userEvent.click(screen.getByRole("button", { name: "Invoke tool" }));
    expect(onSubmit).toHaveBeenCalledWith({ title: "Across the Universe" });
  });

  it("requires explicit confirmation and exposes cancellation only while running", async () => {
    const onCancel = vi.fn();
    const { rerender } = render(
      <SchemaForm schema={{ type: "object", properties: {} }} busy={false} requiresConfirmation onSubmit={vi.fn()} onCancel={onCancel} />,
    );
    const invoke = screen.getByRole("button", { name: "Confirm and invoke" });
    expect(invoke).toBeDisabled();
    await userEvent.click(screen.getByRole("checkbox"));
    expect(invoke).toBeEnabled();
    rerender(<SchemaForm schema={{ type: "object", properties: {} }} busy requiresConfirmation onSubmit={vi.fn()} onCancel={onCancel} />);
    await userEvent.click(screen.getByRole("button", { name: "Cancel" }));
    expect(onCancel).toHaveBeenCalledOnce();
  });

  // "Confirmation before each run" has to mean each run; the box used to stay
  // ticked, so a second click re-ran a persistent tool unconfirmed.
  it("re-arms the confirmation after every submit", async () => {
    const onSubmit = vi.fn();
    const user = userEvent.setup();
    render(
      <SchemaForm
        schema={{ type: "object", properties: { note: { type: "string" } } }}
        busy={false}
        requiresConfirmation
        onSubmit={onSubmit}
        onCancel={vi.fn()}
      />,
    );

    const submit = screen.getByRole("button", { name: /invoke/i });
    expect(submit).toBeDisabled();
    const box = screen.getByRole("checkbox");
    await user.click(box);
    expect(submit).toBeEnabled();

    await user.click(submit);
    expect(onSubmit).toHaveBeenCalledTimes(1);
    expect(box).not.toBeChecked();
    expect(submit).toBeDisabled();
  });

  it("applies a later seed to untouched fields only", async () => {
    const schema = { type: "object", properties: { a: { type: "string" }, b: { type: "string" } } } as const;
    const user = userEvent.setup();
    const view = render(
      <SchemaForm schema={schema} seed={{ a: "seed-a" }} busy={false} requiresConfirmation={false} onSubmit={vi.fn()} onCancel={vi.fn()} />,
    );
    await user.type(document.getElementById("field-b")!, "typed-b");

    view.rerender(
      <SchemaForm schema={schema} seed={{ a: "seed-a2", b: "seed-b" }} busy={false} requiresConfirmation={false} onSubmit={vi.fn()} onCancel={vi.fn()} />,
    );

    await waitFor(() => expect(document.getElementById("field-a")).toHaveValue("seed-a2"));
    expect(document.getElementById("field-b")).toHaveValue("typed-b");
  });
});
