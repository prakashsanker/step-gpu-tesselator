#!/usr/bin/env node

import { spawn, spawnSync } from 'child_process';
import fs from 'fs';
import os from 'os';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const PROJECT_ROOT = path.resolve(__dirname, '..');

function parseArgs(argv) {
    const args = {
        date: null,
        outDir: null,
        skipAi: false,
        aiFilter: null,
        aiSaveScreenshots: false,
        allBench: false,
        benchSuite: 'representative',
    };

    for (let i = 0; i < argv.length; i++) {
        const arg = argv[i];
        if (arg === '--skip-ai') {
            args.skipAi = true;
        } else if (arg === '--all-bench') {
            args.allBench = true;
            args.benchSuite = 'full';
        } else if (arg === '--bench-suite') {
            args.benchSuite = argv[++i] ?? 'representative';
        } else if (arg === '--ai-save') {
            args.aiSaveScreenshots = true;
        } else if (arg === '--date') {
            args.date = argv[++i] ?? null;
        } else if (arg === '--out-dir') {
            args.outDir = argv[++i] ?? null;
        } else if (arg === '--ai-filter') {
            args.aiFilter = argv[++i] ?? null;
        } else if (arg === '--help' || arg === '-h') {
            printUsage();
            process.exit(0);
        } else {
            console.error(`Unknown argument: ${arg}`);
            printUsage();
            process.exit(2);
        }
    }

    return args;
}

function printUsage() {
    console.log(`Usage:
  node scripts/run-beat-occt-baseline.mjs [options]

Options:
  --date YYYY-MM-DD          Use explicit date folder name
  --out-dir PATH             Override output directory root
  --skip-ai                  Skip AI visual test step
  --ai-filter PATTERN        Pass a filter pattern to run-visual-tests-ai.js
  --ai-save                  Save AI test screenshots
  --all-bench                Run benchmark-comprehensive with --all
  --bench-suite NAME         Benchmark suite: canary|representative|full (default: representative)
  -h, --help                 Show this help
`);
}

function runGit(cmdArgs) {
    const res = spawnSync('git', cmdArgs, {
        cwd: PROJECT_ROOT,
        encoding: 'utf8',
    });
    if (res.status !== 0) {
        return '';
    }
    return (res.stdout || '').trim();
}

function nowIsoSafe() {
    return new Date().toISOString().replace(/:/g, '-').replace(/\./g, '-');
}

function makeRunPaths(args) {
    const datePart = args.date ?? new Date().toISOString().slice(0, 10);
    const sha = runGit(['rev-parse', '--short', 'HEAD']) || 'unknown';
    const stamp = nowIsoSafe();
    const root =
        args.outDir ??
        path.join(PROJECT_ROOT, 'diagnostics', 'beat-occt-import-js', datePart);
    const runDir = path.join(root, `${stamp}-${sha}`);
    return { root, runDir, datePart, sha, stamp };
}

function writeJson(filePath, value) {
    fs.writeFileSync(filePath, JSON.stringify(value, null, 2));
}

async function runStep(step, runDir) {
    const logPath = path.join(runDir, `${step.id}.log`);
    const logStream = fs.createWriteStream(logPath, { flags: 'w' });
    const startedAt = new Date();
    const startedMs = Date.now();

    logStream.write(`# ${step.name}\n`);
    logStream.write(`# cmd: ${step.cmd.join(' ')}\n`);
    logStream.write(`# started_at: ${startedAt.toISOString()}\n\n`);

    console.log(`\n[baseline] ${step.name}`);
    console.log(`[baseline] log -> ${path.relative(PROJECT_ROOT, logPath)}`);

    return new Promise((resolve) => {
        const child = spawn(step.cmd[0], step.cmd.slice(1), {
            cwd: PROJECT_ROOT,
            env: process.env,
            stdio: ['ignore', 'pipe', 'pipe'],
        });

        child.stdout.on('data', (chunk) => {
            process.stdout.write(chunk);
            logStream.write(chunk);
        });

        child.stderr.on('data', (chunk) => {
            process.stderr.write(chunk);
            logStream.write(chunk);
        });

        child.on('close', (code, signal) => {
            const durationMs = Date.now() - startedMs;
            const finishedAt = new Date();

            logStream.write(`\n# finished_at: ${finishedAt.toISOString()}\n`);
            logStream.write(`# duration_ms: ${durationMs}\n`);
            logStream.write(`# exit_code: ${code}\n`);
            if (signal) logStream.write(`# signal: ${signal}\n`);
            logStream.end();

            resolve({
                id: step.id,
                name: step.name,
                cmd: step.cmd,
                startedAt: startedAt.toISOString(),
                finishedAt: finishedAt.toISOString(),
                durationMs,
                exitCode: code,
                signal: signal ?? null,
                logPath: path.relative(PROJECT_ROOT, logPath),
                ok: code === 0,
            });
        });
    });
}

async function main() {
    const args = parseArgs(process.argv.slice(2));
    const validSuites = new Set(['canary', 'representative', 'full']);
    if (!validSuites.has(args.benchSuite)) {
        console.error(`Invalid --bench-suite value: ${args.benchSuite}`);
        console.error('Expected one of: canary, representative, full');
        process.exit(2);
    }

    const { runDir, datePart, sha } = makeRunPaths(args);
    fs.mkdirSync(runDir, { recursive: true });

    const branch = runGit(['rev-parse', '--abbrev-ref', 'HEAD']) || 'unknown';
    const status = runGit(['status', '--porcelain']);

    const metadata = {
        createdAt: new Date().toISOString(),
        date: datePart,
        git: {
            branch,
            commit: sha,
            dirty: status.length > 0,
        },
        host: {
            platform: process.platform,
            release: os.release(),
            arch: process.arch,
            cpus: os.cpus()?.length ?? null,
            node: process.version,
        },
        args,
    };
    writeJson(path.join(runDir, 'meta.json'), metadata);

    const steps = [
        {
            id: '01-unit-tests',
            name: 'Unit/Correctness: tests/run-tests.js',
            cmd: ['node', '--experimental-vm-modules', 'tests/run-tests.js'],
        },
        {
            id: '02-benchmark-comprehensive',
            name: `Performance: tests/benchmark-comprehensive.js --suite ${args.benchSuite}`,
            cmd: ['node', 'tests/benchmark-comprehensive.js', '--suite', args.benchSuite],
        },
        {
            id: '03-benchmark-core',
            name: 'Performance: tests/benchmark.js',
            cmd: ['node', 'tests/benchmark.js'],
        },
    ];

    if (!args.skipAi) {
        const aiCmd = ['node', 'tests/run-visual-tests-ai.js'];
        if (args.aiSaveScreenshots) aiCmd.push('--save');
        if (args.aiFilter) aiCmd.push(args.aiFilter);

        if (!process.env.OPENROUTER_API_KEY) {
            const skipped = {
                id: '04-ai-visual',
                name: 'AI Visual: tests/run-visual-tests-ai.js',
                skipped: true,
                reason: 'OPENROUTER_API_KEY is not set',
            };
            writeJson(path.join(runDir, 'ai-visual-skipped.json'), skipped);
            console.log('\n[baseline] Skipping AI visual step (OPENROUTER_API_KEY not set).');
        } else {
            steps.push({
                id: '04-ai-visual',
                name: 'AI Visual: tests/run-visual-tests-ai.js',
                cmd: aiCmd,
            });
        }
    }

    const results = [];
    for (const step of steps) {
        const result = await runStep(step, runDir);
        results.push(result);
    }

    const summary = {
        createdAt: new Date().toISOString(),
        runDir: path.relative(PROJECT_ROOT, runDir),
        git: {
            branch,
            commit: sha,
        },
        steps: results,
        ok: results.every((r) => r.ok),
    };
    writeJson(path.join(runDir, 'summary.json'), summary);

    console.log('\n[baseline] Summary');
    for (const r of results) {
        const statusLabel = r.ok ? 'PASS' : 'FAIL';
        console.log(
            `[baseline] ${statusLabel} ${r.id} (${(r.durationMs / 1000).toFixed(1)}s) -> ${r.logPath}`,
        );
    }
    console.log(`[baseline] Output directory: ${path.relative(PROJECT_ROOT, runDir)}`);

    process.exit(summary.ok ? 0 : 1);
}

main().catch((err) => {
    console.error(err);
    process.exit(1);
});
