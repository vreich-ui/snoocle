/**
 * Run records are historical: a field added to the trace format later is simply
 * absent from every run written before it, and a render must survive that gap
 * rather than throwing and blanking the page.
 */
export function formatCost(value: number | null | undefined): string {
  return typeof value === "number" ? `$${value.toFixed(4)}` : "—";
}

export function formatDateTime(value: string | null | undefined): string {
  if (!value) return "—";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
}

export function orDash(value: string | number | null | undefined): string {
  return value === null || value === undefined || value === "" ? "—" : String(value);
}

export function pairOrDash(left: string | null | undefined, right: string | null | undefined): string {
  if (!left && !right) return "—";
  return `${orDash(left)} / ${orDash(right)}`;
}
