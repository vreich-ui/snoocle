import { useMemo, useState } from "react";
import { formatJson, isRecord } from "./tooling";

export interface JsonSchema {
  type?: string | string[];
  title?: string;
  description?: string;
  default?: unknown;
  enum?: unknown[];
  const?: unknown;
  properties?: Record<string, JsonSchema>;
  required?: string[];
  items?: JsonSchema;
  anyOf?: JsonSchema[];
  oneOf?: JsonSchema[];
  allOf?: JsonSchema[];
  $ref?: string;
  $defs?: Record<string, JsonSchema>;
  definitions?: Record<string, JsonSchema>;
  format?: string;
  minimum?: number;
  maximum?: number;
  minLength?: number;
  maxLength?: number;
  pattern?: string;
}

interface SchemaFormProps {
  schema: JsonSchema;
  initialValue?: Record<string, unknown>;
  busy: boolean;
  blockedReason?: string;
  requiresConfirmation: boolean;
  onSubmit: (args: Record<string, unknown>) => void;
  onCancel: () => void;
}

function humanize(name: string): string {
  return name.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function resolveSchema(schema: JsonSchema, root: JsonSchema): JsonSchema {
  if (schema.$ref?.startsWith("#/$defs/")) return root.$defs?.[schema.$ref.slice(8)] ?? schema;
  if (schema.$ref?.startsWith("#/definitions/")) return root.definitions?.[schema.$ref.slice(14)] ?? schema;
  const union = schema.oneOf ?? schema.anyOf;
  if (union) return union.find((item) => item.type !== "null") ?? union[0] ?? schema;
  if (schema.allOf?.length === 1) return resolveSchema(schema.allOf[0], root);
  return schema;
}

function schemaType(schema: JsonSchema): string {
  if (Array.isArray(schema.type)) return schema.type.find((item) => item !== "null") ?? "string";
  if (schema.type) return schema.type;
  if (schema.properties) return "object";
  if (schema.enum?.length) return typeof schema.enum[0];
  return "string";
}

function defaultsFor(schema: JsonSchema, root: JsonSchema): unknown {
  const resolved = resolveSchema(schema, root);
  if (resolved.default !== undefined) return resolved.default;
  if (resolved.const !== undefined) return resolved.const;
  if (schemaType(resolved) === "object" && resolved.properties) {
    const value: Record<string, unknown> = {};
    for (const [key, child] of Object.entries(resolved.properties)) {
      const childDefault = defaultsFor(child, root);
      if (childDefault !== undefined) value[key] = childDefault;
    }
    return Object.keys(value).length ? value : undefined;
  }
  return undefined;
}

function setAtPath(source: Record<string, unknown>, path: string[], value: unknown): Record<string, unknown> {
  const copy = structuredClone(source);
  let cursor = copy;
  path.slice(0, -1).forEach((segment) => {
    if (!isRecord(cursor[segment])) cursor[segment] = {};
    cursor = cursor[segment] as Record<string, unknown>;
  });
  const key = path.at(-1)!;
  if (value === undefined || value === "") delete cursor[key];
  else cursor[key] = value;
  return copy;
}

function getAtPath(source: Record<string, unknown>, path: string[]): unknown {
  let value: unknown = source;
  for (const segment of path) {
    if (!isRecord(value)) return undefined;
    value = value[segment];
  }
  return value;
}

interface FieldProps {
  name: string;
  path: string[];
  schema: JsonSchema;
  root: JsonSchema;
  value: unknown;
  required: boolean;
  disabled: boolean;
  onChange: (path: string[], value: unknown) => void;
  onJsonError: (path: string, error: string) => void;
}

function Field({ name, path, schema, root, value, required, disabled, onChange, onJsonError }: FieldProps) {
  const resolved = resolveSchema(schema, root);
  const type = schemaType(resolved);
  const id = `field-${path.join("-")}`;
  const label = resolved.title || humanize(name);
  if (type === "object" && resolved.properties) {
    const recordValue = isRecord(value) ? value : {};
    return (
      <fieldset className="schema-group">
        <legend>{label}{required ? " *" : ""}</legend>
        {resolved.description && <p className="field-help">{resolved.description}</p>}
        {Object.entries(resolved.properties).map(([childName, childSchema]) => (
          <Field
            key={childName}
            name={childName}
            path={[...path, childName]}
            schema={childSchema}
            root={root}
            value={recordValue[childName]}
            required={resolved.required?.includes(childName) ?? false}
            disabled={disabled}
            onChange={onChange}
            onJsonError={onJsonError}
          />
        ))}
      </fieldset>
    );
  }

  if (type === "array" || type === "object") {
    const initialText = value === undefined ? "" : formatJson(value);
    return (
      <JsonField
        id={id}
        label={label}
        description={resolved.description}
        initialText={initialText}
        required={required}
        disabled={disabled}
        onChange={(next) => onChange(path, next)}
        onError={(message) => onJsonError(path.join("."), message)}
      />
    );
  }

  const common = {
    id,
    disabled,
    required,
    "aria-describedby": resolved.description ? `${id}-help` : undefined,
  };
  return (
    <label className="schema-field" htmlFor={id}>
      <span>{label}{required ? " *" : ""}</span>
      {resolved.enum ? (
        <select
          {...common}
          value={value === undefined ? "" : String(value)}
          onChange={(event) => onChange(path, event.target.value || undefined)}
        >
          {!required && <option value="">Not set</option>}
          {required && value === undefined && <option value="">Select…</option>}
          {resolved.enum.map((option) => <option key={String(option)} value={String(option)}>{String(option)}</option>)}
        </select>
      ) : type === "boolean" ? (
        <select
          {...common}
          value={value === undefined ? "" : String(value)}
          onChange={(event) => onChange(path, event.target.value === "" ? undefined : event.target.value === "true")}
        >
          {!required && <option value="">Not set</option>}
          {required && value === undefined && <option value="">Select…</option>}
          <option value="true">True</option>
          <option value="false">False</option>
        </select>
      ) : (
        <input
          {...common}
          type={type === "number" || type === "integer" ? "number" : resolved.format === "date" ? "date" : "text"}
          step={type === "integer" ? 1 : type === "number" ? "any" : undefined}
          min={resolved.minimum}
          max={resolved.maximum}
          minLength={resolved.minLength}
          maxLength={resolved.maxLength}
          pattern={resolved.pattern}
          value={value === undefined ? "" : String(value)}
          onChange={(event) => {
            if (type === "number" || type === "integer") {
              onChange(path, event.target.value === "" ? undefined : Number(event.target.value));
            } else onChange(path, event.target.value || undefined);
          }}
        />
      )}
      {resolved.description && <small className="field-help" id={`${id}-help`}>{resolved.description}</small>}
    </label>
  );
}

interface JsonFieldProps {
  id: string;
  label: string;
  description?: string;
  initialText: string;
  required: boolean;
  disabled: boolean;
  onChange: (value: unknown) => void;
  onError: (message: string) => void;
}

function JsonField({ id, label, description, initialText, required, disabled, onChange, onError }: JsonFieldProps) {
  const [text, setText] = useState(initialText);
  return (
    <label className="schema-field" htmlFor={id}>
      <span>{label}{required ? " *" : ""}</span>
      <textarea
        id={id}
        disabled={disabled}
        required={required}
        rows={6}
        value={text}
        onChange={(event) => {
          const next = event.target.value;
          setText(next);
          if (!next.trim()) {
            onError("");
            onChange(undefined);
            return;
          }
          try {
            onChange(JSON.parse(next) as unknown);
            onError("");
          } catch {
            onError(`${label} must contain valid JSON`);
          }
        }}
      />
      {description && <small className="field-help">{description}</small>}
    </label>
  );
}

export function SchemaForm({ schema, initialValue, busy, blockedReason, requiresConfirmation, onSubmit, onCancel }: SchemaFormProps) {
  const initial = useMemo(() => {
    const defaults = defaultsFor(schema, schema);
    return { ...(isRecord(defaults) ? defaults : {}), ...(initialValue ?? {}) };
  }, [schema, initialValue]);
  const [values, setValues] = useState<Record<string, unknown>>(initial);
  const [advanced, setAdvanced] = useState(false);
  const [rawText, setRawText] = useState(() => formatJson(initial));
  const [error, setError] = useState("");
  const [jsonErrors, setJsonErrors] = useState<Record<string, string>>({});
  const [confirmed, setConfirmed] = useState(false);
  const properties = schema.properties ?? {};

  const updateValue = (path: string[], value: unknown) => {
    setValues((current) => setAtPath(current, path, value));
  };

  const updateJsonError = (path: string, message: string) => {
    setJsonErrors((current) => ({ ...current, [path]: message }));
  };

  const submit = (event: React.FormEvent) => {
    event.preventDefault();
    if (Object.values(jsonErrors).some(Boolean)) {
      setError(Object.values(jsonErrors).find(Boolean) ?? "Fix invalid JSON fields");
      return;
    }
    let args = values;
    if (advanced) {
      try {
        const parsed = JSON.parse(rawText) as unknown;
        if (!isRecord(parsed)) throw new Error("Arguments must be a JSON object");
        args = parsed;
      } catch (caught) {
        setError(caught instanceof Error ? caught.message : "Arguments must be valid JSON");
        return;
      }
    }
    setError("");
    onSubmit(args);
  };

  return (
    <form className="schema-form" onSubmit={submit}>
      <div className="mode-switch" role="group" aria-label="Argument editor mode">
        <button
          className={!advanced ? "selected" : ""}
          type="button"
          onClick={() => {
            if (advanced) {
              try {
                const parsed = JSON.parse(rawText) as unknown;
                if (!isRecord(parsed)) throw new Error("Arguments must be a JSON object");
                setValues(parsed);
                setError("");
                setAdvanced(false);
              } catch (caught) {
                setError(caught instanceof Error ? caught.message : "Arguments must be valid JSON");
              }
            } else setAdvanced(false);
          }}
        >Form</button>
        <button
          className={advanced ? "selected" : ""}
          type="button"
          onClick={() => {
            setRawText(formatJson(values));
            setAdvanced(true);
            setError("");
          }}
        >Raw JSON</button>
      </div>
      {advanced ? (
        <label className="schema-field" htmlFor="raw-arguments">
          <span>Arguments JSON</span>
          <textarea
            id="raw-arguments"
            aria-label="Arguments JSON"
            disabled={busy || Boolean(blockedReason)}
            rows={14}
            value={rawText}
            onChange={(event) => setRawText(event.target.value)}
          />
        </label>
      ) : Object.entries(properties).length ? (
        Object.entries(properties).map(([name, property]) => (
          <Field
            key={name}
            name={name}
            path={[name]}
            schema={property}
            root={schema}
            value={getAtPath(values, [name])}
            required={schema.required?.includes(name) ?? false}
            disabled={busy || Boolean(blockedReason)}
            onChange={updateValue}
            onJsonError={updateJsonError}
          />
        ))
      ) : <p className="muted">This tool takes no arguments.</p>}
      {requiresConfirmation && !blockedReason && (
        <label className="confirmation-check">
          <input
            type="checkbox"
            checked={confirmed}
            disabled={busy}
            onChange={(event) => setConfirmed(event.target.checked)}
          />
          I understand this tool can have persistent, costly, or destructive effects.
        </label>
      )}
      {blockedReason && <p className="warning" role="alert">{blockedReason}</p>}
      {error && <p className="error" role="alert">{error}</p>}
      <div className="form-actions">
        <button
          className="primary"
          type="submit"
          disabled={busy || Boolean(blockedReason) || (requiresConfirmation && !confirmed)}
        >{busy ? "Running…" : requiresConfirmation ? "Confirm and invoke" : "Invoke tool"}</button>
        {busy && <button type="button" onClick={onCancel}>Cancel</button>}
      </div>
    </form>
  );
}
