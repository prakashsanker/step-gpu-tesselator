/**
 * Quick test to verify optimized ear clipping produces correct results
 */

import puppeteer from 'puppeteer';

async function main() {
    console.log('\n=== Testing Optimized Ear Clipping ===\n');

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
        if (text.includes('[') || text.includes('Error') || text.includes('error')) {
            console.log(`[Browser] ${text}`);
        }
    });

    await page.goto('http://localhost:5173/tests/benchmark-harness.html', {
        waitUntil: 'networkidle0',
        timeout: 30000,
    });

    await page.waitForFunction(() => window.benchmarkReady === true, { timeout: 30000 });
    console.log('Harness ready\n');

    const tests = [
        { name: 'Triangle', points: [[0, 0], [1, 0], [0.5, 1]], expectedTriangles: 1 },
        { name: 'Square', points: [[0, 0], [1, 0], [1, 1], [0, 1]], expectedTriangles: 2 },
        { name: 'Pentagon', points: [[0, 0], [2, 0], [2.5, 1.5], [1, 2.5], [-0.5, 1.5]], expectedTriangles: 3 },
        { name: 'Hexagon', points: [[1, 0], [2, 0], [2.5, 1], [2, 2], [1, 2], [0.5, 1]], expectedTriangles: 4 },
        {
            name: 'Octagon',
            points: (() => {
                const pts = [];
                for (let i = 0; i < 8; i++) {
                    const angle = (i / 8) * Math.PI * 2;
                    pts.push([Math.cos(angle), Math.sin(angle)]);
                }
                return pts;
            })(),
            expectedTriangles: 6,
        },
        {
            name: 'L-Shape (concave)',
            points: [[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]],
            expectedTriangles: 4,
        },
        {
            name: '20-gon',
            points: (() => {
                const pts = [];
                for (let i = 0; i < 20; i++) {
                    const angle = (i / 20) * Math.PI * 2;
                    pts.push([Math.cos(angle), Math.sin(angle)]);
                }
                return pts;
            })(),
            expectedTriangles: 18,
        },
        {
            name: '50-gon',
            points: (() => {
                const pts = [];
                for (let i = 0; i < 50; i++) {
                    const angle = (i / 50) * Math.PI * 2;
                    pts.push([Math.cos(angle), Math.sin(angle)]);
                }
                return pts;
            })(),
            expectedTriangles: 48,
        },
    ];

    let passed = 0;
    let failed = 0;

    for (const test of tests) {
        try {
            const result = await page.evaluate(async (points) => {
                try {
                    const triangles = await window.benchmarkHarness.earClippingOptimized(points);
                    return { success: true, triangleCount: triangles.length, triangles };
                } catch (e) {
                    return { success: false, error: e.message };
                }
            }, test.points);

            if (!result.success) {
                console.log(`❌ ${test.name}: ${result.error}`);
                failed++;
                continue;
            }

            if (result.triangleCount === test.expectedTriangles) {
                console.log(`✅ ${test.name}: ${result.triangleCount} triangles`);
                passed++;
            } else {
                console.log(`❌ ${test.name}: expected ${test.expectedTriangles} triangles, got ${result.triangleCount}`);
                failed++;
            }
        } catch (e) {
            console.log(`❌ ${test.name}: ${e.message}`);
            failed++;
        }
    }

    console.log(`\n=== Results: ${passed} passed, ${failed} failed ===\n`);

    await browser.close();
    process.exit(failed > 0 ? 1 : 0);
}

main().catch(e => {
    console.error(e);
    process.exit(1);
});
