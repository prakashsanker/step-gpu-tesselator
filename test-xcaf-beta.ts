/**
 * Minimal test to verify beta OpenCascade.js XCAF color extraction APIs
 */

// @ts-ignore
import initOpenCascade from 'opencascade.js';

async function main() {
  const outputEl = document.getElementById('output')!;

  function log(msg: string, type: 'info' | 'success' | 'error' = 'info') {
    const line = document.createElement('div');
    line.className = type;
    line.textContent = msg;
    outputEl.appendChild(line);
    console.log(msg);
  }

  log('Loading OpenCascade.js beta...');

  try {
    const oc = await initOpenCascade();
    log('OpenCascade.js loaded!', 'success');

    // Check critical XCAF APIs
    log('\n=== XCAF API Availability ===');

    // TDF_LabelSequence - the key missing API in previous builds
    const tdfApis = Object.keys(oc).filter(k => k.includes('TDF_LabelSequence'));
    log(`TDF_LabelSequence APIs: ${tdfApis.length > 0 ? tdfApis.join(', ') : 'NONE'}`,
        tdfApis.length > 0 ? 'success' : 'error');

    // XCAFDoc_DocumentTool
    const hasDocTool = typeof oc.XCAFDoc_DocumentTool !== 'undefined';
    log(`XCAFDoc_DocumentTool: ${hasDocTool ? 'YES' : 'NO'}`,
        hasDocTool ? 'success' : 'error');

    // XCAFDoc_ColorTool
    const colorToolApis = Object.keys(oc).filter(k => k.includes('XCAFDoc_ColorTool'));
    log(`XCAFDoc_ColorTool APIs: ${colorToolApis.length > 0 ? colorToolApis.join(', ') : 'NONE'}`,
        colorToolApis.length > 0 ? 'success' : 'error');

    // XCAFDoc_ShapeTool
    const shapeToolApis = Object.keys(oc).filter(k => k.includes('XCAFDoc_ShapeTool'));
    log(`XCAFDoc_ShapeTool APIs: ${shapeToolApis.length > 0 ? shapeToolApis.join(', ') : 'NONE'}`,
        shapeToolApis.length > 0 ? 'success' : 'error');

    // Try to create TDF_LabelSequence
    log('\n=== Testing TDF_LabelSequence Creation ===');
    let labelSeq = null;
    try {
      if (oc.TDF_LabelSequence_1) {
        labelSeq = new oc.TDF_LabelSequence_1();
        log('Created TDF_LabelSequence via TDF_LabelSequence_1()', 'success');
      } else if (oc.TDF_LabelSequence) {
        labelSeq = new oc.TDF_LabelSequence();
        log('Created TDF_LabelSequence via TDF_LabelSequence()', 'success');
      } else {
        log('No TDF_LabelSequence constructor found', 'error');
      }

      if (labelSeq) {
        // Check methods available on the sequence
        const proto = Object.getPrototypeOf(labelSeq);
        const methods = Object.getOwnPropertyNames(proto).filter(m => m !== 'constructor');
        log(`TDF_LabelSequence methods: ${methods.slice(0, 10).join(', ')}...`, 'info');
      }
    } catch (err: any) {
      log(`Failed to create TDF_LabelSequence: ${err.message}`, 'error');
    }

    // Test with a simple STEP file
    log('\n=== Testing STEP Color Extraction ===');

    // Fetch a test file
    const testFile = '/step-examples/c8-solids/colored-solid.step';
    try {
      const resp = await fetch(testFile);
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const stepContent = await resp.text();
      log(`Loaded test file: ${testFile} (${(stepContent.length/1024).toFixed(1)} KB)`, 'info');

      // Create document and read STEP
      const doc = new oc.TDocStd_Document(new oc.TCollection_ExtendedString_1());

      // Write to virtual file
      const fileName = 'test.step';
      oc.FS.writeFile('/' + fileName, stepContent);

      // Create reader
      const reader = new oc.STEPCAFControl_Reader_1();
      reader.SetColorMode(true);
      reader.SetNameMode(true);

      const readStatus = reader.ReadFile(fileName);
      log(`Reader.ReadFile status: ${readStatus}`, readStatus === 1 ? 'success' : 'error');

      if (readStatus === 1) {
        const transferOk = reader.Transfer_1(doc, new oc.Message_ProgressRange_1());
        log(`Reader.Transfer status: ${transferOk}`, transferOk ? 'success' : 'error');

        if (transferOk) {
          // Get tools
          const shapeTool = oc.XCAFDoc_DocumentTool.ShapeTool(doc.Main());
          const colorTool = oc.XCAFDoc_DocumentTool.ColorTool(doc.Main());

          log('Got ShapeTool and ColorTool from document', 'success');

          // Try to get colors using TDF_LabelSequence
          if (oc.TDF_LabelSequence_1) {
            try {
              const labels = new oc.TDF_LabelSequence_1();

              // Get all free shapes
              const actualShapeTool = shapeTool.get ? shapeTool.get() : shapeTool;
              actualShapeTool.GetFreeShapes(labels);

              log(`GetFreeShapes found ${labels.Length()} shapes`, 'success');

              // Get colors
              const colorLabels = new oc.TDF_LabelSequence_1();
              const actualColorTool = colorTool.get ? colorTool.get() : colorTool;
              actualColorTool.GetColors(colorLabels);

              log(`GetColors found ${colorLabels.Length()} colors`, 'success');

              // List the colors
              for (let i = 1; i <= colorLabels.Length(); i++) {
                const label = colorLabels.Value(i);
                const color = new oc.Quantity_Color_1();

                // Try to get color from label
                const gotColor = actualColorTool.GetColor_1(label, color);
                if (gotColor) {
                  const r = color.Red();
                  const g = color.Green();
                  const b = color.Blue();
                  log(`  Color ${i}: RGB(${(r*255).toFixed(0)}, ${(g*255).toFixed(0)}, ${(b*255).toFixed(0)})`, 'success');
                }
              }

            } catch (err: any) {
              log(`Error during color extraction: ${err.message}`, 'error');
              console.error(err);
            }
          } else {
            log('Cannot test color extraction - TDF_LabelSequence not available', 'error');
          }
        }
      }

      // Cleanup
      oc.FS.unlink('/' + fileName);

    } catch (err: any) {
      log(`Failed to load test file: ${err.message}`, 'error');
    }

    log('\n=== Test Complete ===');

  } catch (err: any) {
    log(`Failed to initialize OpenCascade: ${err.message}`, 'error');
    console.error(err);
  }
}

main();
