// Run occ-test.html and capture console output
import puppeteer from 'puppeteer';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

async function run() {
  const browser = await puppeteer.launch({ headless: true });
  const page = await browser.newPage();

  // Collect ALL console output
  page.on('console', msg => {
    console.log('[Browser]', msg.text());
  });

  page.on('pageerror', err => {
    console.error('[Browser Error]', err.message);
  });

  console.log('Loading occ-test.html...');
  await page.goto('http://localhost:5173/occ-test.html', {
    waitUntil: 'domcontentloaded',
    timeout: 60000
  });

  console.log('Page loaded, waiting for it to settle...');
  await new Promise(r => setTimeout(r, 3000));

  // Upload a STEP file
  console.log('Uploading colored-solid.step...');
  const stepFilePath = path.resolve(__dirname, '../step-examples/c8-solids/colored-solid.step');

  const fileInput = await page.$('#file-input');
  if (!fileInput) {
    console.log('File input not found, dumping page content...');
    const content = await page.content();
    console.log(content.slice(0, 2000));
    await browser.close();
    return;
  }

  await fileInput.uploadFile(stepFilePath);

  // Wait for processing
  console.log('Waiting for processing (30s)...');
  await new Promise(r => setTimeout(r, 30000));

  await browser.close();
  console.log('\nDone');
}

run().catch(err => {
  console.error('Error:', err);
  process.exit(1);
});
