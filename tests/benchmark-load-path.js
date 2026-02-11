import puppeteer from 'puppeteer';
import { spawn } from 'child_process';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import fs from 'fs';

const __dirname = dirname(fileURLToPath(import.meta.url));
const PROJECT_ROOT = join(__dirname, '..');

const args = process.argv.slice(2);

const SUITES = {
  canary: [
    { name: 'Plate XLarge (holes)', path: 'step-examples/benchmark/plate-xlarge-20x20.step' },
    { name: 'Conical Surface (complex)', path: 'step-examples/complex/conical-surface.step' },
    { name: 'VM-001', path: 'step-examples/VM-001.STEP' },
  ],
  representative: [
    { name: 'Electronic Enclosure', path: 'step-examples/Electronic Enclousre.STEP' },
    { name: 'VM-001', path: 'step-examples/VM-001.STEP' },
    { name: 'Conical Surface (complex)', path: 'step-examples/complex/conical-surface.step' },
  ],
};

function parseArgs(argv) {
  const out = {
    suite: 'representative',
    timeoutMs: 360000,
    host: '127.0.0.1',
    port: 5175,
    prewarm: true,
  };

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--suite') {
      out.suite = (argv[i + 1] || '').toLowerCase();
      i += 1;
    } else if (arg === '--timeout-ms') {
      out.timeoutMs = Number(argv[i + 1]) || out.timeoutMs;
      i += 1;
    } else if (arg === '--no-prewarm') {
      out.prewarm = false;
    } else if (arg === '--help' || arg === '-h') {
      console.log('Usage: node tests/benchmark-load-path.js [--suite canary|representative] [--timeout-ms N] [--no-prewarm]');
      process.exit(0);
    }
  }

  if (!SUITES[out.suite]) {
    throw new Error(`Unknown suite: ${out.suite}`);
  }

  return out;
}

function loadStepFile(relativePath) {
  const fullPath = join(PROJECT_ROOT, relativePath);
  const content = fs.readFileSync(fullPath, 'utf8');
  const size = fs.statSync(fullPath).size;
  return { content, size };
}

async function startViteServer(host, port) {
  return new Promise((resolve, reject) => {
    const vite = spawn('npx', ['vite', '--host', host, '--port', String(port)], {
      cwd: PROJECT_ROOT,
      stdio: ['ignore', 'pipe', 'pipe'],
    });

    let started = false;
    let stderr = '';
    const timeout = setTimeout(() => {
      if (!started) {
        vite.kill();
        reject(new Error('Vite startup timeout'));
      }
    }, 60000);

    vite.stdout.on('data', (chunk) => {
      const text = chunk.toString();
      if (text.includes('Local:') && !started) {
        started = true;
        clearTimeout(timeout);
        const match = text.match(/http:\/\/[^:]+:(\d+)\//);
        resolve({ process: vite, port: match ? Number(match[1]) : port });
      }
    });

    vite.stderr.on('data', (chunk) => {
      stderr += chunk.toString();
    });

    vite.on('close', (code) => {
      if (started) return;
      clearTimeout(timeout);
      reject(new Error(`Vite exited early (code ${code}): ${stderr}`));
    });
  });
}

async function run() {
  const cfg = parseArgs(args);
  const models = SUITES[cfg.suite];

  console.log(`Load-path benchmark suite=${cfg.suite}, models=${models.length}`);

  let vite = null;
  let browser = null;
  const results = [];

  try {
    vite = await startViteServer(cfg.host, cfg.port);
    browser = await puppeteer.launch({
      headless: true,
      args: [
        '--enable-unsafe-webgpu',
        '--enable-features=Vulkan',
        '--use-vulkan=swiftshader',
        '--enable-gpu-rasterization',
        '--disable-gpu-sandbox',
        '--no-sandbox',
        '--disable-setuid-sandbox',
      ],
    });

    const page = await browser.newPage();
    page.setDefaultTimeout(cfg.timeoutMs);

    await page.goto(`http://${cfg.host}:${vite.port}/tests/benchmark-comprehensive.html`, {
      waitUntil: 'networkidle0',
      timeout: cfg.timeoutMs,
    });
    await page.waitForFunction(() => window.benchmarkReady === true, { timeout: cfg.timeoutMs });

    if (cfg.prewarm) {
      const warm = loadStepFile('step-examples/benchmark/simple-square.step');
      await page.evaluate(async (content) => {
        try { await window.benchmark.runOCC(content); } catch (_) {}
      }, warm.content);
    }

    for (let i = 0; i < models.length; i++) {
      const model = models[i];
      const step = loadStepFile(model.path);
      console.log(`\n[${i + 1}/${models.length}] ${model.name}`);

      const run = await page.evaluate(async (content) => {
        return await window.benchmark.runOCC(content);
      }, step.content);

      if (!run?.success) {
        results.push({ name: model.name, path: model.path, size: step.size, success: false, error: run?.error || 'unknown error' });
        console.log(`FAIL ${model.name}: ${run?.error || 'unknown error'}`);
        continue;
      }

      const phases = run.phases || {};
      const loadMs = phases.loadStepFile || 0;
      const readMs = phases.loadStepFile_readFile || 0;
      const transferMs = phases.loadStepFile_transfer || 0;
      const oneShapeMs = phases.loadStepFile_oneShape || 0;
      const fsWriteMs = phases.loadStepFile_fsWrite || 0;
      const fsCleanupMs = phases.loadStepFile_fsCleanup || 0;
      const restMs = Math.max(0, run.totalTime - loadMs);

      results.push({
        name: model.name,
        path: model.path,
        size: step.size,
        success: true,
        totalMs: run.totalTime,
        loadMs,
        loadSharePct: run.totalTime > 0 ? (loadMs / run.totalTime) * 100 : 0,
        readMs,
        transferMs,
        oneShapeMs,
        fsWriteMs,
        fsCleanupMs,
        restMs,
        triangles: run.triangleCount,
      });

      console.log(
        `PASS ${model.name} | total=${run.totalTime.toFixed(1)}ms | load=${loadMs.toFixed(1)}ms (${((loadMs / run.totalTime) * 100).toFixed(1)}%) | read=${readMs.toFixed(1)} transfer=${transferMs.toFixed(1)} oneShape=${oneShapeMs.toFixed(1)} fs=${(fsWriteMs + fsCleanupMs).toFixed(1)}`
      );
    }

    console.log('\n' + '-'.repeat(136));
    console.log(
      [
        'Model'.padEnd(34),
        'SizeKB'.padStart(8),
        'Total(ms)'.padStart(10),
        'Load(ms)'.padStart(10),
        'Load%'.padStart(8),
        'Read'.padStart(8),
        'Transfer'.padStart(10),
        'OneShape'.padStart(10),
        'FS'.padStart(8),
        'Rest'.padStart(10),
        'Tris'.padStart(8),
      ].join(' | ')
    );
    console.log('-'.repeat(136));

    for (const r of results) {
      if (!r.success) {
        console.log([r.name.padEnd(34), ((r.size || 0) / 1024).toFixed(1).padStart(8), 'FAILED'.padStart(10)].join(' | '));
        continue;
      }
      console.log(
        [
          r.name.padEnd(34),
          (r.size / 1024).toFixed(1).padStart(8),
          r.totalMs.toFixed(1).padStart(10),
          r.loadMs.toFixed(1).padStart(10),
          r.loadSharePct.toFixed(1).padStart(8),
          r.readMs.toFixed(1).padStart(8),
          r.transferMs.toFixed(1).padStart(10),
          r.oneShapeMs.toFixed(1).padStart(10),
          (r.fsWriteMs + r.fsCleanupMs).toFixed(1).padStart(8),
          r.restMs.toFixed(1).padStart(10),
          String(r.triangles).padStart(8),
        ].join(' | ')
      );
    }

    const out = {
      timestamp: new Date().toISOString(),
      suite: cfg.suite,
      timeoutMs: cfg.timeoutMs,
      prewarm: cfg.prewarm,
      results,
    };

    const outPath = join(PROJECT_ROOT, 'tests', 'benchmark-load-results.json');
    fs.writeFileSync(outPath, JSON.stringify(out, null, 2));
    console.log(`\nSaved ${outPath}`);
  } finally {
    if (browser) await browser.close();
    if (vite?.process) vite.process.kill();
  }
}

run().catch((err) => {
  console.error(err);
  process.exit(1);
});
