/**
 * Debug script to test B-spline surface handling in VM-001.STEP
 */
const puppeteer = require('puppeteer');
const fs = require('fs');
const path = require('path');

async function main() {
  const browser = await puppeteer.launch({ headless: true });
  const page = await browser.newPage();

  // Collect console messages
  const logs = [];
  page.on('console', msg => {
    const text = msg.text();
    logs.push(text);
    console.log('BROWSER:', text);
  });

  page.on('pageerror', err => {
    console.error('PAGE ERROR:', err.message);
  });

  // Navigate to app
  await page.goto('http://localhost:5173', { waitUntil: 'networkidle0' });

  // Read the STEP file
  const stepFilePath = path.resolve(__dirname, '../step-examples/VM-001.STEP');

  // Upload the file
  const fileInput = await page.$('#file-input');
  await fileInput.uploadFile(stepFilePath);

  // Wait for processing
  await page.waitForFunction(() => {
    const stats = document.getElementById('stats');
    return stats && (stats.innerHTML.includes('Vertices:') || stats.innerHTML.includes('Error:'));
  }, { timeout: 120000 });

  // Get stats
  const stats = await page.$eval('#stats', el => el.innerHTML);
  console.log('\n=== STATS ===');
  console.log(stats.replace(/<br>/g, '\n'));

  // Check for B-spline related logs
  console.log('\n=== B-SPLINE RELATED LOGS ===');
  for (const log of logs) {
    if (log.includes('B-spline') || log.includes('bspline') ||
        log.includes('C6b') || log.includes('UV boundary') ||
        log.includes('tessellate') || log.includes('Trimmed') ||
        log.includes('vertices:') || log.includes('triangles:')) {
      console.log(log);
    }
  }

  // Check for warnings and errors
  console.log('\n=== WARNINGS/ERRORS ===');
  for (const log of logs) {
    if (log.includes('warn') || log.includes('error') ||
        log.includes('NaN') || log.includes('Invalid')) {
      console.log(log);
    }
  }

  await browser.close();
}

main().catch(err => {
  console.error('Test failed:', err);
  process.exit(1);
});
