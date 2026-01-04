import puppeteer from 'puppeteer';
import { spawn } from 'child_process';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const PROJECT_ROOT = join(__dirname, '..');

async function main() {
    console.log('Starting Vite dev server...');

    const vite = spawn('npx', ['vite', '--port', '5176'], {
        cwd: PROJECT_ROOT,
        stdio: ['ignore', 'pipe', 'pipe'],
    });

    await new Promise((resolve, reject) => {
        const timeout = setTimeout(() => reject(new Error('Vite timeout')), 30000);
        vite.stdout.on('data', (data) => {
            if (data.toString().includes('Local:')) {
                clearTimeout(timeout);
                resolve();
            }
        });
    });

    console.log('Launching browser...');
    const browser = await puppeteer.launch({
        headless: true,
        args: [
            '--enable-unsafe-webgpu',
            '--enable-features=Vulkan',
            '--use-vulkan=swiftshader',
            '--disable-gpu-sandbox',
            '--no-sandbox',
        ],
    });

    const page = await browser.newPage();

    page.on('console', msg => {
        const text = msg.text();
        if (text.includes('PASS')) {
            console.log('\x1b[32m' + text + '\x1b[0m');
        } else if (text.includes('FAIL')) {
            console.log('\x1b[31m' + text + '\x1b[0m');
        } else {
            console.log(text);
        }
    });

    page.on('pageerror', err => {
        console.log('\x1b[31m[Error] ' + err.message + '\x1b[0m');
    });

    console.log('Navigating to CDT test page...');
    await page.goto('http://localhost:5176/tests/test-cdt.html', {
        waitUntil: 'networkidle0',
        timeout: 60000,
    });

    // Wait for tests to complete
    await page.waitForFunction(() => window.testsDone === true, { timeout: 30000 });

    await browser.close();
    vite.kill();
    console.log('\nCDT tests complete.');
}

main().catch(e => {
    console.error(e);
    process.exit(1);
});
