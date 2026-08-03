import { cleanup, fireEvent, render, screen } from "@testing-library/react";
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
});
