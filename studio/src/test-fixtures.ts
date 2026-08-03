import type { StudioTool, ToolContract } from "./tooling";

export function contract(overrides: Partial<ToolContract> = {}): ToolContract {
  return {
    schemaVersion: 1,
    title: "Dynamic Echo",
    category: "diagnostics",
    browserSafety: "safe",
    inputArtifactKinds: ["text"],
    outputArtifactKinds: ["json"],
    access: { mode: "read", readOnly: true, destructive: false, idempotent: true },
    networkAccess: [],
    modelUse: "none",
    persistence: [],
    cacheBehavior: "none",
    expectedDuration: "instant",
    specializedRenderer: "json",
    ...overrides,
  };
}

export function tool(name = "dynamic_echo", overrides: Partial<StudioTool> = {}): StudioTool {
  return {
    name,
    title: "Dynamic Echo",
    description: "Echo a dynamically discovered value.",
    inputSchema: {
      type: "object",
      properties: { message: { type: "string", description: "Text to echo" } },
      required: ["message"],
    },
    contract: contract(),
    classificationSource: "capabilities",
    ...overrides,
  };
}
