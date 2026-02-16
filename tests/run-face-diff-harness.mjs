import puppeteer from 'puppeteer';
import { spawn } from 'child_process';
import fs from 'fs';
import { dirname, join } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const PROJECT_ROOT = join(__dirname, '..');
const PORT = Number(process.env.FACE_DIFF_PORT || 5177);
const DEFAULT_STEP_FILE = 'step-examples/complex/electronicEnclosure.step';
const DEFAULT_FACE_IDS = [63, 64, 65, 66];

function parseFaceIds(value) {
  if (!value) return DEFAULT_FACE_IDS;
  const parsed = value
    .split(',')
    .map((token) => Number.parseInt(token.trim(), 10))
    .filter((n) => Number.isFinite(n) && n >= 0);
  return parsed.length > 0 ? Array.from(new Set(parsed)).sort((a, b) => a - b) : DEFAULT_FACE_IDS;
}

async function startVite() {
  return new Promise((resolve, reject) => {
    const vite = spawn('npx', ['vite', '--host', '127.0.0.1', '--port', String(PORT)], {
      cwd: PROJECT_ROOT,
      stdio: ['ignore', 'pipe', 'pipe'],
    });

    let started = false;
    let startupOutput = '';
    const markStarted = () => {
      if (started) return;
      started = true;
      clearTimeout(timeout);
      clearTimeout(grace);
      resolve(vite);
    };

    const timeout = setTimeout(() => {
      if (!started) {
        vite.kill('SIGTERM');
        reject(new Error(`Vite startup timeout. Output:\n${startupOutput}`));
      }
    }, 60000);

    const grace = setTimeout(() => {
      markStarted();
    }, 3000);

    const onData = (chunk) => {
      const text = chunk.toString();
      startupOutput += text;
      if (startupOutput.length > 8000) {
        startupOutput = startupOutput.slice(-8000);
      }
      if (text.includes('Local:') || text.includes('ready in')) {
        markStarted();
      }
    };

    vite.stdout.on('data', onData);
    vite.stderr.on('data', onData);
    vite.on('error', (error) => {
      clearTimeout(timeout);
      clearTimeout(grace);
      reject(new Error(`Vite process error: ${error.message}\n${startupOutput}`));
    });
    vite.on('exit', (code) => {
      if (!started && code !== 0) {
        clearTimeout(timeout);
        clearTimeout(grace);
        reject(new Error(`Vite exited before startup (code ${code}). Output:\n${startupOutput}`));
      }
    });
  });
}

function summarizeFaceRow(row) {
  const ours = row.ours;
  const reference = row.reference;
  const oursLabel = ours
    ? `${ours.outputTriangleCount} tris (${ours.status})`
    : 'missing';
  const refLabel = reference
    ? `${reference.triangleCount} tris`
    : 'missing';
  const delta = row.triangleDelta !== undefined ? row.triangleDelta : 'n/a';
  const deltaPct = row.triangleDeltaPct !== undefined
    ? `${row.triangleDeltaPct.toFixed(2)}%`
    : 'n/a';
  return `face ${row.faceIndex}: ours=${oursLabel}, ref=${refLabel}, delta=${delta} (${deltaPct})`;
}

async function main() {
  const stepFile = process.argv[2] || DEFAULT_STEP_FILE;
  const faceIds = parseFaceIds(process.argv[3]);
  const outputPath = process.argv[4] || `/tmp/face-diff-report-${Date.now()}.json`;
  // Oracle mode is opt-in only. Default stays on our TS/GPU path.
  const enableOcctNativeFaceTessellation = process.env.ENABLE_OCCT_NATIVE_FACE_TESSELLATION === '1';
  const enableOcctInspiredTrimGraph = process.env.ENABLE_OCCT_INSPIRED_TRIM_GRAPH === '1';
  const occtInspiredTrimGraphFaceIds = parseFaceIds(process.env.OCCT_INSPIRED_TRIM_GRAPH_FACE_IDS);
  const nativeLinDeflection = process.env.OCCT_NATIVE_LIN_DEFLECTION
    ? Number.parseFloat(process.env.OCCT_NATIVE_LIN_DEFLECTION)
    : undefined;
  const nativeLinDeflectionRatio = process.env.OCCT_NATIVE_LIN_DEFLECTION_RATIO
    ? Number.parseFloat(process.env.OCCT_NATIVE_LIN_DEFLECTION_RATIO)
    : undefined;
  const nativeAngDeflection = process.env.OCCT_NATIVE_ANG_DEFLECTION
    ? Number.parseFloat(process.env.OCCT_NATIVE_ANG_DEFLECTION)
    : undefined;
  const curveVerboseLogs = process.env.CURVE_VERBOSE_LOGS === '1';
  const tessellationVerboseLogs = process.env.TESSELLATION_VERBOSE_LOGS === '1';
  const trimVerboseLogs = process.env.TRIM_VERBOSE_LOGS === '1';
  const localUvClassifierShadow = process.env.LOCAL_UV_CLASSIFIER_SHADOW === '1';
  const perfGeometryOnlyLoad = process.env.PERF_GEOMETRY_ONLY_LOAD === '1';

  const absoluteStepPath = join(PROJECT_ROOT, stepFile);
  if (!fs.existsSync(absoluteStepPath)) {
    throw new Error(`STEP file not found: ${absoluteStepPath}`);
  }

  const vite = await startVite();
  let browser;

  try {
    browser = await puppeteer.launch({
      protocolTimeout: 600000,
      headless: true,
      args: [
        '--enable-unsafe-webgpu',
        '--enable-features=Vulkan',
        '--use-vulkan=swiftshader',
        '--enable-gpu-rasterization',
        '--disable-gpu-sandbox',
        '--no-sandbox',
        '--disable-setuid-sandbox',
        '--window-size=1600,900',
      ],
    });

    const page = await browser.newPage();
    await page.setViewport({ width: 1600, height: 900 });
    page.on('console', (msg) => {
      console.log(`[browser:${msg.type()}] ${msg.text()}`);
    });
    page.on('pageerror', (err) => {
      console.error(`[pageerror] ${err.message}`);
    });

    await page.goto(`http://localhost:${PORT}/`, {
      waitUntil: 'networkidle0',
      timeout: 60000,
    });

    await page.waitForFunction(() => window.visualValidationReady === true, {
      timeout: 60000,
    });

    const report = await page.evaluate(async (testPath, targets, opts) => {
      globalThis.__FACE_DEBUG_MODE__ = 'only';
      globalThis.__FACE_DEBUG_IDS__ = targets;
      globalThis.__ENABLE_CONE_SEAM_SPLIT__ = true;
      globalThis.__CONE_SEAM_SPLIT_FACE_IDS__ = targets;
      globalThis.__CURVE_VERBOSE_LOGS__ = !!opts.curveVerboseLogs;
      globalThis.__TESSELLATION_VERBOSE_LOGS__ = !!opts.tessellationVerboseLogs;
      globalThis.__TRIM_VERBOSE_LOGS__ = !!opts.trimVerboseLogs;
      globalThis.__ENABLE_LOCAL_UV_CLASSIFIER_SHADOW__ = !!opts.localUvClassifierShadow;
      globalThis.__ENABLE_OCCT_INSPIRED_TRIM_GRAPH__ = !!opts.enableOcctInspiredTrimGraph;
      globalThis.__OCCT_INSPIRED_TRIM_GRAPH_FACE_IDS__ = opts.occtInspiredTrimGraphFaceIds;
      globalThis.__PERF_GEOMETRY_ONLY_LOAD__ = !!opts.perfGeometryOnlyLoad;
      globalThis.__ALLOW_OCCT_ORACLE_PATH__ = !!opts.enableOcctNativeFaceTessellation;
      globalThis.__ENABLE_OCCT_NATIVE_FACE_TESSELLATION__ = !!opts.enableOcctNativeFaceTessellation;
      globalThis.__OCCT_NATIVE_FACE_IDS__ = targets;
      if (Number.isFinite(opts.nativeLinDeflection)) {
        globalThis.__OCCT_NATIVE_LIN_DEFLECTION__ = opts.nativeLinDeflection;
      }
      if (Number.isFinite(opts.nativeLinDeflectionRatio)) {
        globalThis.__OCCT_NATIVE_LIN_DEFLECTION_RATIO__ = opts.nativeLinDeflectionRatio;
      }
      if (Number.isFinite(opts.nativeAngDeflection)) {
        globalThis.__OCCT_NATIVE_ANG_DEFLECTION__ = opts.nativeAngDeflection;
      }
      return await window.visualValidation.runFaceDiff(testPath, targets);
    }, stepFile, faceIds, {
      enableOcctInspiredTrimGraph,
      occtInspiredTrimGraphFaceIds,
      enableOcctNativeFaceTessellation,
      nativeLinDeflection,
      nativeLinDeflectionRatio,
      nativeAngDeflection,
      curveVerboseLogs,
      tessellationVerboseLogs,
      trimVerboseLogs,
      localUvClassifierShadow,
      perfGeometryOnlyLoad,
    });

    fs.writeFileSync(outputPath, `${JSON.stringify(report, null, 2)}\n`);

    console.log(`Face diff report written: ${outputPath}`);
    if (enableOcctNativeFaceTessellation) {
      console.log('[harness] OCCT native face triangulation oracle: ENABLED');
    } else {
      console.log('[harness] OCCT native face triangulation oracle: DISABLED (TS/GPU path)');
    }
    if (enableOcctInspiredTrimGraph) {
      console.log(`[harness] OCCT-inspired trim graph path: ENABLED faces=[${occtInspiredTrimGraphFaceIds.join(',')}]`);
    } else {
      console.log('[harness] OCCT-inspired trim graph path: DISABLED');
    }
    for (const row of report.rows) {
      console.log(summarizeFaceRow(row));
    }
  } finally {
    await browser?.close();
    vite.kill('SIGTERM');
  }
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
