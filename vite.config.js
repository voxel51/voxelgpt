import { defineConfig } from "@voxel51/fiftyone-js-plugin-build";
import { dirname } from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const dir = __dirname;

export default defineConfig(dir, {
  buildConfigOverride: { sourcemap: true },
});
