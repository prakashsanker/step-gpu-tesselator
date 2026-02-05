#!/usr/bin/env python3
"""
Test XCAF color extraction using native OCCT (via pythonocc).
This proves that native OCCT can properly extract colors from STEP files,
which OpenCascade.js cannot do due to missing TDF_LabelSequence bindings.

Install: pip install pythonocc-core
"""

import sys
import os

try:
    from OCC.Core.STEPCAFControl import STEPCAFControl_Reader
    from OCC.Core.TDocStd import TDocStd_Document
    from OCC.Core.XCAFDoc import XCAFDoc_DocumentTool, XCAFDoc_ColorType
    from OCC.Core.TDF import TDF_LabelSequence, TDF_Label
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_SOLID, TopAbs_FACE, TopAbs_SHAPE
    from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB
    from OCC.Core.IFSelect import IFSelect_RetDone
except ImportError:
    print("ERROR: pythonocc-core is not installed.")
    print("Install with: pip install pythonocc-core")
    print("Or use conda: conda install -c conda-forge pythonocc-core")
    sys.exit(1)


def extract_colors_from_step(step_file_path: str) -> dict:
    """
    Extract colors from a STEP file using XCAF (the gold standard approach).
    Returns a dict with color extraction statistics and sample colors.
    """
    if not os.path.exists(step_file_path):
        raise FileNotFoundError(f"STEP file not found: {step_file_path}")

    # Create XCAF document
    doc = TDocStd_Document("pythonocc-doc")

    # Initialize reader with all modes enabled
    reader = STEPCAFControl_Reader()
    reader.SetColorMode(True)
    reader.SetLayerMode(True)
    reader.SetNameMode(True)
    reader.SetMatMode(True)
    reader.SetGDTMode(True)

    print(f"[XCAF] Reading STEP file: {step_file_path}")
    status = reader.ReadFile(step_file_path)

    if status != IFSelect_RetDone:
        raise RuntimeError(f"Failed to read STEP file. Status: {status}")

    print("[XCAF] Transferring to document...")
    reader.Transfer(doc)

    # Get tools
    shape_tool = XCAFDoc_DocumentTool.ShapeTool(doc.Main())
    color_tool = XCAFDoc_DocumentTool.ColorTool(doc.Main())

    # Get all free shapes (top-level shapes)
    labels = TDF_LabelSequence()
    shape_tool.GetFreeShapes(labels)

    print(f"[XCAF] Found {labels.Length()} free shapes in document")

    # Statistics
    stats = {
        "total_labels": labels.Length(),
        "labels_with_color": 0,
        "solids_found": 0,
        "solids_with_color": 0,
        "faces_found": 0,
        "faces_with_color": 0,
        "colors": {},  # unique colors found
        "sample_colors": [],  # sample color assignments
    }

    # Helper to get color from label or shape
    def get_color(label_or_shape, color_type):
        """Try to get color from label or shape."""
        color = Quantity_Color()

        # Try different color types
        for ct in [XCAFDoc_ColorType.XCAFDoc_ColorSurf,
                   XCAFDoc_ColorType.XCAFDoc_ColorGen,
                   XCAFDoc_ColorType.XCAFDoc_ColorCurv]:
            try:
                if isinstance(label_or_shape, TDF_Label):
                    if color_tool.GetColor(label_or_shape, ct, color):
                        return color
                else:
                    if color_tool.GetColor(label_or_shape, ct, color):
                        return color
            except Exception:
                pass
        return None

    # Process labels recursively
    def process_label(label: TDF_Label, depth: int = 0):
        """Process a label and its children for colors."""
        # Get shape for this label
        shape = shape_tool.GetShape(label)

        # Check if this label has a color
        color = get_color(label, None)
        if color:
            stats["labels_with_color"] += 1
            r, g, b = color.Red(), color.Green(), color.Blue()
            color_key = f"({r:.2f}, {g:.2f}, {b:.2f})"
            stats["colors"][color_key] = stats["colors"].get(color_key, 0) + 1

            if len(stats["sample_colors"]) < 10:
                stats["sample_colors"].append({
                    "depth": depth,
                    "r": r, "g": g, "b": b,
                    "type": "label"
                })

        # Explore solids within this shape
        if shape:
            solid_explorer = TopExp_Explorer(shape, TopAbs_SOLID)
            while solid_explorer.More():
                solid = solid_explorer.Current()
                stats["solids_found"] += 1

                # Try to get color for this solid
                solid_color = get_color(solid, None)
                if solid_color:
                    stats["solids_with_color"] += 1
                    r, g, b = solid_color.Red(), solid_color.Green(), solid_color.Blue()
                    color_key = f"({r:.2f}, {g:.2f}, {b:.2f})"
                    stats["colors"][color_key] = stats["colors"].get(color_key, 0) + 1

                    if len(stats["sample_colors"]) < 20:
                        stats["sample_colors"].append({
                            "depth": depth,
                            "r": r, "g": g, "b": b,
                            "type": "solid"
                        })

                solid_explorer.Next()

            # Explore faces within this shape
            face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
            while face_explorer.More():
                face = face_explorer.Current()
                stats["faces_found"] += 1

                # Try to get color for this face
                face_color = get_color(face, None)
                if face_color:
                    stats["faces_with_color"] += 1

                face_explorer.Next()

        # Process children
        from OCC.Core.TDF import TDF_ChildIterator
        child_iter = TDF_ChildIterator(label, False)
        while child_iter.More():
            process_label(child_iter.Value(), depth + 1)
            child_iter.Next()

    # Process all free shapes
    for i in range(1, labels.Length() + 1):
        label = labels.Value(i)
        process_label(label)

    return stats


def main():
    # Default test file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # Try different test files
    test_files = [
        os.path.join(project_root, "step-examples/complex/rocky_house.step"),
        os.path.join(project_root, "step-examples/c8-solids/colored-solid.step"),
    ]

    # Allow command line argument
    if len(sys.argv) > 1:
        test_files = [sys.argv[1]]

    for step_file in test_files:
        if not os.path.exists(step_file):
            print(f"Skipping {step_file} (not found)")
            continue

        print(f"\n{'='*60}")
        print(f"Testing: {os.path.basename(step_file)}")
        print(f"{'='*60}")

        try:
            stats = extract_colors_from_step(step_file)

            print(f"\n[RESULTS]")
            print(f"  Total labels: {stats['total_labels']}")
            print(f"  Labels with color: {stats['labels_with_color']}")
            print(f"  Solids found: {stats['solids_found']}")
            print(f"  Solids with color: {stats['solids_with_color']}")
            print(f"  Faces found: {stats['faces_found']}")
            print(f"  Faces with color: {stats['faces_with_color']}")
            print(f"  Unique colors: {len(stats['colors'])}")

            if stats['colors']:
                print(f"\n[COLORS FOUND]")
                for color_key, count in sorted(stats['colors'].items(), key=lambda x: -x[1])[:10]:
                    print(f"  RGB{color_key}: {count} occurrences")

            if stats['sample_colors']:
                print(f"\n[SAMPLE COLOR ASSIGNMENTS]")
                for sample in stats['sample_colors'][:5]:
                    print(f"  {sample['type']} at depth {sample['depth']}: RGB({sample['r']:.2f}, {sample['g']:.2f}, {sample['b']:.2f})")

            # Success criteria
            if stats['labels_with_color'] > 0 or stats['solids_with_color'] > 0:
                print(f"\n[SUCCESS] Native OCCT extracted colors from XCAF!")
            else:
                print(f"\n[WARNING] No colors found via XCAF - check file or method")

        except Exception as e:
            print(f"[ERROR] {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
