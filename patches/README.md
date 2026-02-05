# occt-import-js Color Extraction Fix

## Problem

The original `occt-import-js` library doesn't extract per-face colors from STEP files properly. When STYLED_ITEM entities are used to color entire solids or shells (which is the common case in STEP files), the individual faces within those solids don't inherit the color.

### Root Cause

In the original code (`importer-xcaf.cpp`), when `XcafFace::GetColor()` is called:
1. It searches for a label for the face using `shapeTool->Search(face, label)`
2. If the face doesn't have its own label (which is common for faces within solids), it returns false
3. No fallback to the parent shape's color exists

### Evidence

Testing showed:
- `colored-solid.step`: mesh.color works (1/1), but `brep_faces[].color` is always null (0/12)
- `rocky_house.step`: Only 2/161 meshes have mesh.color, all 8285 `brep_faces[].color` are null

## Solution

The fix modifies `XcafFace` to:
1. Store a pointer to its parent shape (solid/shell)
2. When getting a face's color, first try the face's own color
3. If not found, fall back to the parent shape's color

### Files Modified

- `occt-import-js/src/importer-xcaf.cpp` - Added parent shape parameter to `XcafFace` and fallback logic in `GetColor()`

### Key Changes

```cpp
// Added to XcafFace constructor
XcafFace (const TopoDS_Face& face, ..., const TopoDS_Shape* parent) :
    ...,
    parentShape (parent)
{ }

// Modified GetColor()
virtual bool GetColor (Color& color) const override
{
    // First try to get the face's own color
    if (GetShapeColor ((const TopoDS_Shape&) face, shapeTool, colorTool, color)) {
        return true;
    }
    // Fall back to parent shape's color
    if (parentShape != nullptr) {
        return GetShapeColor (*parentShape, shapeTool, colorTool, color);
    }
    return false;
}
```

## Building

1. Ensure Emscripten SDK is set up (run `tools/setup_emscripten_mac.sh` first)
2. Run the build script:
   ```bash
   cd occt-import-js-fork
   ./tools/build_wasm_mac.sh
   ```
3. Output files will be in `dist/`:
   - `occt-import-js.js`
   - `occt-import-js.wasm`

## Usage

Copy the built `dist/occt-import-js.js` and `dist/occt-import-js.wasm` to your project. After parsing a STEP file, each face in `brep_faces` should now have a `color` property that inherits from the parent solid if the face doesn't have its own color.

```javascript
import occtimportjs from './occt-import-js.js';

const occ = await occtimportjs();
const result = occ.ReadStepFile(stepData, { linearUnit: 'millimeter' });

// Now brep_faces[i].color will be populated with inherited colors
for (const mesh of result.meshes) {
    for (const face of mesh.brep_faces) {
        if (face.color) {
            console.log(`Face color: RGB(${face.color.r}, ${face.color.g}, ${face.color.b})`);
        }
    }
}
```

## Applying the Patch

If you need to apply this fix to a fresh clone of occt-import-js:

```bash
cd memphis/patches
./apply-color-fix.sh
```

Or manually copy `importer-xcaf-patched.cpp` to replace `occt-import-js/src/importer-xcaf.cpp`.
