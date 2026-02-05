// Check what OpenCascade.js APIs are available for Transfer/WorkSession access
import initOpenCascade from 'opencascade.js';

async function checkAPIs() {
  console.log('Initializing OpenCascade.js...');
  const oc = await initOpenCascade();

  // Check what STEP reader APIs are available
  const stepApis = Object.keys(oc).filter(k =>
    k.includes('STEP') || k.includes('XSControl') || k.includes('Transfer')
  );

  console.log('\n=== STEP/Transfer/XSControl related APIs ===');
  stepApis.sort().forEach(api => console.log(api));

  // Check STEPCAFControl_Reader methods
  console.log('\n=== STEPCAFControl_Reader_1 prototype methods ===');
  if (oc.STEPCAFControl_Reader_1) {
    const reader = new oc.STEPCAFControl_Reader_1();
    const proto = Object.getPrototypeOf(reader);
    const methods = Object.getOwnPropertyNames(proto).filter(m => typeof reader[m] === 'function');
    methods.sort().forEach(m => console.log('  ' + m));

    // Check if Reader() method exists (returns underlying STEPControl_Reader)
    if (reader.Reader) {
      console.log('\n=== reader.Reader() (STEPControl_Reader) methods ===');
      try {
        const innerReader = reader.Reader();
        const innerProto = Object.getPrototypeOf(innerReader);
        const innerMethods = Object.getOwnPropertyNames(innerProto).filter(m => typeof innerReader[m] === 'function');
        innerMethods.sort().forEach(m => console.log('  ' + m));

        // Check if WS() exists (returns XSControl_WorkSession)
        if (innerReader.WS) {
          console.log('\n=== innerReader.WS() (XSControl_WorkSession) methods ===');
          const ws = innerReader.WS();
          const wsProto = Object.getPrototypeOf(ws);
          const wsMethods = Object.getOwnPropertyNames(wsProto).filter(m => typeof ws[m] === 'function');
          wsMethods.sort().forEach(m => console.log('  ' + m));

          // Check TransferReader
          if (ws.TransferReader) {
            console.log('\n=== ws.TransferReader() methods ===');
            const tr = ws.TransferReader();
            const trProto = Object.getPrototypeOf(tr);
            const trMethods = Object.getOwnPropertyNames(trProto).filter(m => typeof tr[m] === 'function');
            trMethods.sort().forEach(m => console.log('  ' + m));

            // Check TransientProcess
            if (tr.TransientProcess) {
              console.log('\n=== tr.TransientProcess() methods ===');
              const tp = tr.TransientProcess();
              const tpProto = Object.getPrototypeOf(tp);
              const tpMethods = Object.getOwnPropertyNames(tpProto).filter(m => typeof tp[m] === 'function');
              tpMethods.sort().forEach(m => console.log('  ' + m));
            } else {
              console.log('\nTransferReader does NOT have TransientProcess method');
            }
          } else {
            console.log('\nWorkSession does NOT have TransferReader method');
          }
        } else {
          console.log('\nSTEPControl_Reader does NOT have WS method');
        }
      } catch (e) {
        console.log('Error accessing inner reader:', e);
      }
    } else {
      console.log('\nSTEPCAFControl_Reader does NOT have Reader method');
    }

    reader.delete();
  }

  console.log('\n=== Done ===');
}

checkAPIs().catch(console.error);
