// Run the OCC API check in a browser via puppeteer
import puppeteer from 'puppeteer';

async function run() {
  const browser = await puppeteer.launch({ headless: true });
  const page = await browser.newPage();

  // Collect console output
  page.on('console', msg => {
    console.log('[Browser]', msg.text());
  });

  page.on('pageerror', err => {
    console.error('[Browser Error]', err.message);
  });

  // Navigate to the test page
  console.log('Loading test page...');
  await page.goto('http://localhost:5173/tests/check-occ-apis.html', {
    waitUntil: 'networkidle0',
    timeout: 180000
  });

  // Wait a bit for OCC to load
  console.log('Waiting for OpenCascade.js to initialize...');
  await new Promise(r => setTimeout(r, 15000));

  // Get whatever output we have
  const output = await page.$eval('#output', el => el.textContent).catch(() => '(no output)');
  const status = await page.$eval('#status', el => el.textContent).catch(() => '(no status)');

  console.log('\nStatus:', status);
  console.log('\nOutput:\n' + output);

  await browser.close();
}

run().catch(err => {
  console.error('Error:', err);
  process.exit(1);
});
