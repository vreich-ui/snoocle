import { expect, test } from "@playwright/test";

test.beforeEach(async ({ page }) => {
  await page.addInitScript(() => {
    window.sessionStorage.setItem("snoocle.studio.bearer-token", "browser-test-token");
  });
});

test("discovers and invokes the live same-origin MCP server", async ({ page }) => {
  await page.goto("/studio/tool-studio");
  await expect(page.getByRole("status")).toContainText(/Connected · \d+ tools · \d+ browser-runnable/);
  await expect(page.getByText("Official MCP client → same-origin")).toBeVisible();

  await page.getByLabel("Search tools").fill("server status");
  await page.getByRole("button", { name: /server_status/ }).click();
  await expect(page.getByText("This tool takes no arguments.")).toBeVisible();
  await page.getByRole("button", { name: "Invoke tool" }).click();

  const resultPanel = page.getByRole("heading", { name: "Tool result" }).locator("..");
  await expect(resultPanel).toBeVisible();
  await expect(resultPanel.getByTestId("tool-result")).toContainText('"ffmpeg"');
  await expect(resultPanel.getByText("Elapsed", { exact: true })).toBeVisible();
  await expect(resultPanel.getByText("Cache", { exact: true })).toBeVisible();
  await expect(resultPanel.getByText("Model", { exact: true })).toBeVisible();
  await expect(resultPanel.getByText("Cost", { exact: true })).toBeVisible();

  await page.getByRole("button", { name: "Raw MCP result" }).click();
  await expect(resultPanel.getByTestId("tool-result")).toContainText('"content"');
  await expect(page.getByRole("button", { name: /server_status.*success/ })).toBeVisible();
});

test("filters the live registry and refuses server-local path invocation", async ({ page }) => {
  await page.goto("/studio/tool-studio");
  await expect(page.getByRole("status")).toContainText("Connected");
  await page.getByLabel("Browser safety").selectOption("server_filesystem_restricted");
  await expect(page.getByText(/Showing \d+ of \d+/)).toBeVisible();
  await page.getByLabel("Search tools").fill("analyze full track mir");
  await page.getByRole("button", { name: /analyze_full_track_mir/ }).click();
  await expect(page.getByRole("alert")).toContainText("server-local file path");
  await expect(page.getByRole("button", { name: "Invoke tool" })).toBeDisabled();
  await page.getByRole("button", { name: "Raw JSON" }).click();
  await expect(page.getByLabel("Arguments JSON")).toBeDisabled();
});
