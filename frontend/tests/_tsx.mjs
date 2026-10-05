// Load a .tsx component in node:test without adding a test framework.
//
// The frontend has no test runner (PROJECT_HARDENING_PLAN.md C4) and this repo does not
// add dependencies casually, so this uses only what is already installed: TypeScript
// (devDependency) transpiles the component, and react-dom/server renders it. The output
// goes under node_modules/.cache so `react/jsx-runtime` resolves from this package.
import ts from "typescript";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { basename, dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");

export async function importTsx(relativePath) {
  const source = readFileSync(join(root, relativePath), "utf8");
  const { outputText } = ts.transpileModule(source, {
    compilerOptions: {
      jsx: ts.JsxEmit.ReactJSX,
      module: ts.ModuleKind.ESNext,
      target: ts.ScriptTarget.ES2022,
    },
  });
  const dir = join(root, "node_modules", ".cache", "rgh-tests");
  mkdirSync(dir, { recursive: true });
  const file = join(dir, basename(relativePath).replace(/\.tsx$/, ".mjs"));
  writeFileSync(file, outputText);
  return import(`${pathToFileURL(file).href}?t=${Date.now()}`);
}
