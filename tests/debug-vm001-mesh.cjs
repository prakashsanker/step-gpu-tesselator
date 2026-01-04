/**
 * Debug script to analyze mesh assembly for VM-001.STEP
 */
const puppeteer = require('puppeteer');
const path = require('path');

async function main() {
  const browser = await puppeteer.launch({ headless: true });
  const page = await browser.newPage();

  // Collect console messages
  page.on('console', msg => {
    const text = msg.text();
    console.log('BROWSER:', text);
  });

  page.on('pageerror', err => {
    console.error('PAGE ERROR:', err.message);
  });

  // Navigate to app
  await page.goto('http://localhost:5173', { waitUntil: 'networkidle0' });

  // Add debug code to check mesh validity before rendering
  await page.evaluate(() => {
    // Override render function to add validation
    const origRender = window.createThreeMeshFromTesselation;
    if (origRender) {
      window.createThreeMeshFromTesselation = function(mesh) {
        const numVertices = mesh.positions.length / 3;
        const numIndices = mesh.indices.length;
        let maxIndex = 0;
        let invalidIndices = 0;

        for (let i = 0; i < numIndices; i++) {
          const idx = mesh.indices[i];
          if (idx >= numVertices) {
            invalidIndices++;
            if (invalidIndices <= 10) {
              console.log(`[MESH VALIDATION] Invalid index ${idx} at position ${i}, numVertices=${numVertices}`);
            }
          }
          maxIndex = Math.max(maxIndex, idx);
        }

        console.log(`[MESH VALIDATION] vertices=${numVertices}, indices=${numIndices}, maxIndex=${maxIndex}, invalidIndices=${invalidIndices}`);

        return origRender(mesh);
      };
    }
  });

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

  await browser.close();
}

main().catch(err => {
  console.error('Test failed:', err);
  process.exit(1);
});
