/**
 * STEP file color parser - extracts STYLED_ITEM colors from raw STEP text
 *
 * This parser follows the STEP color chain:
 * STYLED_ITEM -> PRESENTATION_STYLE_ASSIGNMENT -> SURFACE_STYLE_USAGE
 *   -> SURFACE_SIDE_STYLE -> SURFACE_STYLE_FILL_AREA -> FILL_AREA_STYLE
 *   -> FILL_AREA_STYLE_COLOUR -> COLOUR_RGB
 *
 * Also handles: SURFACE_STYLE_RENDERING_WITH_PROPERTIES -> COLOUR_RGB
 */

export interface RGBColor {
  r: number;
  g: number;
  b: number;
}

export interface StyledItemColor {
  styledItemId: number;
  targetId: number;  // The entity ID this style applies to (face, shell, solid)
  color: RGBColor;
}

interface StepEntity {
  id: number;
  type: string;
  data: string;
}

/**
 * Parse STEP entities from raw STEP file content
 */
function parseStepEntities(stepContent: string): Map<number, StepEntity> {
  const entities = new Map<number, StepEntity>();

  // Match entity definitions: #123=ENTITY_TYPE(data);
  // Handle multi-line entities by removing newlines first
  const cleanContent = stepContent.replace(/\r?\n/g, ' ');

  const entityRegex = /#(\d+)\s*=\s*([A-Z_][A-Z0-9_]*)\s*\(([^;]*)\)\s*;/g;

  let match;
  while ((match = entityRegex.exec(cleanContent)) !== null) {
    const id = parseInt(match[1], 10);
    const type = match[2];
    const data = match[3].trim();
    entities.set(id, { id, type, data });
  }

  return entities;
}

/**
 * Parse a reference like #123 and return the numeric ID
 */
function parseRef(ref: string): number | null {
  const match = ref.trim().match(/^#(\d+)$/);
  if (match) {
    return parseInt(match[1], 10);
  }
  return null;
}

/**
 * Parse a list of references like (#123,#456,#789)
 */
function parseRefList(data: string): number[] {
  const refs: number[] = [];
  const matches = data.matchAll(/#(\d+)/g);
  for (const match of matches) {
    refs.push(parseInt(match[1], 10));
  }
  return refs;
}

/**
 * Extract RGB color from a COLOUR_RGB entity
 * Format: COLOUR_RGB('name', r, g, b)
 */
function extractColorRGB(entity: StepEntity): RGBColor | null {
  if (entity.type !== 'COLOUR_RGB') return null;

  // Parse: 'name', r, g, b
  const parts = entity.data.split(',');
  if (parts.length < 4) return null;

  // Skip the name, get r, g, b
  const r = parseFloat(parts[1].trim());
  const g = parseFloat(parts[2].trim());
  const b = parseFloat(parts[3].trim());

  if (isNaN(r) || isNaN(g) || isNaN(b)) return null;

  return { r, g, b };
}

/**
 * Follow the color chain from a STYLED_ITEM to find the COLOUR_RGB
 */
function resolveColor(
  styledItemData: string,
  entities: Map<number, StepEntity>
): RGBColor | null {
  // STYLED_ITEM('name', (style_refs), item_ref)
  // We need to follow the style_refs to find the color

  // Extract the style references (second parameter)
  const styleRefs = parseRefList(styledItemData);
  if (styleRefs.length === 0) return null;

  // Follow the first style reference
  const styleRef = styleRefs[0];
  const styleEntity = entities.get(styleRef);
  if (!styleEntity) return null;

  // Navigate through the style chain
  return followStyleChain(styleEntity, entities);
}

/**
 * Follow the style chain from any style entity to find the color
 */
function followStyleChain(
  entity: StepEntity,
  entities: Map<number, StepEntity>
): RGBColor | null {
  if (!entity) return null;

  switch (entity.type) {
    case 'COLOUR_RGB':
      return extractColorRGB(entity);

    case 'PRESENTATION_STYLE_ASSIGNMENT': {
      // Format: ((style_refs))
      const refs = parseRefList(entity.data);
      for (const ref of refs) {
        const next = entities.get(ref);
        if (next) {
          const color = followStyleChain(next, entities);
          if (color) return color;
        }
      }
      return null;
    }

    case 'SURFACE_STYLE_USAGE': {
      // Format: .BOTH., surface_side_style_ref
      const refs = parseRefList(entity.data);
      for (const ref of refs) {
        const next = entities.get(ref);
        if (next) {
          const color = followStyleChain(next, entities);
          if (color) return color;
        }
      }
      return null;
    }

    case 'SURFACE_SIDE_STYLE': {
      // Format: 'name', (style_refs)
      const refs = parseRefList(entity.data);
      for (const ref of refs) {
        const next = entities.get(ref);
        if (next) {
          const color = followStyleChain(next, entities);
          if (color) return color;
        }
      }
      return null;
    }

    case 'SURFACE_STYLE_FILL_AREA': {
      // Format: fill_area_style_ref
      const refs = parseRefList(entity.data);
      if (refs.length > 0) {
        const next = entities.get(refs[0]);
        if (next) return followStyleChain(next, entities);
      }
      return null;
    }

    case 'FILL_AREA_STYLE': {
      // Format: 'name', (fill_area_style_colour_refs)
      const refs = parseRefList(entity.data);
      for (const ref of refs) {
        const next = entities.get(ref);
        if (next) {
          const color = followStyleChain(next, entities);
          if (color) return color;
        }
      }
      return null;
    }

    case 'FILL_AREA_STYLE_COLOUR': {
      // Format: 'name', colour_ref
      const refs = parseRefList(entity.data);
      if (refs.length > 0) {
        const next = entities.get(refs[0]);
        if (next) return followStyleChain(next, entities);
      }
      return null;
    }

    case 'SURFACE_STYLE_RENDERING_WITH_PROPERTIES': {
      // Format: .CONSTANT_SHADING., colour_ref, ...
      const refs = parseRefList(entity.data);
      if (refs.length > 0) {
        const next = entities.get(refs[0]);
        if (next) return followStyleChain(next, entities);
      }
      return null;
    }

    default:
      // Unknown entity type in chain, try following any refs
      const refs = parseRefList(entity.data);
      for (const ref of refs) {
        const next = entities.get(ref);
        if (next) {
          const color = followStyleChain(next, entities);
          if (color) return color;
        }
      }
      return null;
  }
}

/**
 * Extract the target entity ID from STYLED_ITEM
 * Format: STYLED_ITEM('name', (styles), target_ref)
 */
function extractStyledItemTarget(data: string): number | null {
  // The target is typically the last reference in the data
  const refs = parseRefList(data);
  if (refs.length > 0) {
    return refs[refs.length - 1];
  }
  return null;
}

/**
 * Parse all STYLED_ITEM entities and extract their colors
 */
export function parseStepColors(stepContent: string): Map<number, RGBColor> {
  console.log('[parseStepColors] Starting...');
  const startTime = performance.now();

  const entities = parseStepEntities(stepContent);
  console.log(`[parseStepColors] Parsed ${entities.size} entities in ${(performance.now() - startTime).toFixed(0)}ms`);

  // Find all STYLED_ITEM entities
  const styledItems: StyledItemColor[] = [];

  for (const entity of entities.values()) {
    if (entity.type === 'STYLED_ITEM') {
      const color = resolveColor(entity.data, entities);
      const targetId = extractStyledItemTarget(entity.data);

      if (color && targetId !== null) {
        styledItems.push({
          styledItemId: entity.id,
          targetId,
          color
        });
      }
    }
  }

  console.log(`[parseStepColors] Found ${styledItems.length} styled items with colors`);

  // Build target ID -> color map
  // Note: A target can have multiple styled items, we use the last one
  const targetColorMap = new Map<number, RGBColor>();
  for (const item of styledItems) {
    targetColorMap.set(item.targetId, item.color);
  }

  // Log unique colors found
  const uniqueColors = new Set<string>();
  for (const color of targetColorMap.values()) {
    uniqueColors.add(`${color.r.toFixed(3)},${color.g.toFixed(3)},${color.b.toFixed(3)}`);
  }
  console.log(`[parseStepColors] ${targetColorMap.size} unique targets, ${uniqueColors.size} unique colors`);

  const elapsed = performance.now() - startTime;
  console.log(`[parseStepColors] Complete in ${elapsed.toFixed(0)}ms`);

  return targetColorMap;
}

/**
 * Get a list of all ADVANCED_FACE entity IDs in order of appearance
 */
export function getAdvancedFaceIds(stepContent: string): number[] {
  const faceIds: number[] = [];

  // Match ADVANCED_FACE entities - they can be multi-line so use a simpler regex
  const cleanContent = stepContent.replace(/\r?\n/g, ' ');
  const faceRegex = /#(\d+)\s*=\s*ADVANCED_FACE/g;

  let match;
  while ((match = faceRegex.exec(cleanContent)) !== null) {
    faceIds.push(parseInt(match[1], 10));
  }

  return faceIds;
}

/**
 * Build a face index -> color mapping for use with occt-import-js brep_faces
 *
 * This assumes brep_faces appear in the same order as ADVANCED_FACE entities
 * in the STEP file.
 */
export function buildFaceColorMap(stepContent: string): RGBColor[] {
  console.log('[buildFaceColorMap] Starting...');

  // Get target -> color mapping
  const targetColorMap = parseStepColors(stepContent);

  // Get ordered list of ADVANCED_FACE IDs
  const faceIds = getAdvancedFaceIds(stepContent);
  console.log(`[buildFaceColorMap] Found ${faceIds.length} ADVANCED_FACE entities`);

  // Build ordered color array
  const faceColors: RGBColor[] = [];
  let foundColors = 0;

  for (const faceId of faceIds) {
    const color = targetColorMap.get(faceId);
    if (color) {
      faceColors.push(color);
      foundColors++;
    } else {
      // Default gray for faces without color
      faceColors.push({ r: 0.7, g: 0.7, b: 0.7 });
    }
  }

  console.log(`[buildFaceColorMap] Mapped ${foundColors}/${faceIds.length} faces to colors`);

  return faceColors;
}

/**
 * Resolve what faces a target entity contains
 * Used when STYLED_ITEM points to a shell or solid instead of a face
 */
export function resolveFacesForTarget(
  targetId: number,
  stepContent: string
): number[] {
  const entities = parseStepEntities(stepContent);
  const entity = entities.get(targetId);

  if (!entity) return [];

  switch (entity.type) {
    case 'ADVANCED_FACE':
      return [targetId];

    case 'CLOSED_SHELL':
    case 'OPEN_SHELL': {
      // Shell contains faces: CLOSED_SHELL('name', (face_refs))
      return parseRefList(entity.data);
    }

    case 'MANIFOLD_SOLID_BREP': {
      // Solid contains a shell: MANIFOLD_SOLID_BREP('name', shell_ref)
      const shellRefs = parseRefList(entity.data);
      const faces: number[] = [];
      for (const shellRef of shellRefs) {
        faces.push(...resolveFacesForTarget(shellRef, stepContent));
      }
      return faces;
    }

    default:
      // For other types, return the target itself (might be a face)
      return [targetId];
  }
}

/**
 * Build a comprehensive face color map that handles colors applied at
 * shell/solid level (not just face level)
 */
export function buildComprehensiveFaceColorMap(stepContent: string): Map<number, RGBColor> {
  console.log('[buildComprehensiveFaceColorMap] Starting...');
  const startTime = performance.now();

  const entities = parseStepEntities(stepContent);
  const faceColorMap = new Map<number, RGBColor>();

  // Process all STYLED_ITEM entities
  for (const entity of entities.values()) {
    if (entity.type === 'STYLED_ITEM') {
      const color = resolveColor(entity.data, entities);
      const targetId = extractStyledItemTarget(entity.data);

      if (color && targetId !== null) {
        // Resolve what faces this target contains
        const targetEntity = entities.get(targetId);

        if (targetEntity) {
          switch (targetEntity.type) {
            case 'ADVANCED_FACE':
              faceColorMap.set(targetId, color);
              break;

            case 'CLOSED_SHELL':
            case 'OPEN_SHELL': {
              // Apply color to all faces in the shell
              const faceRefs = parseRefList(targetEntity.data);
              for (const faceRef of faceRefs) {
                faceColorMap.set(faceRef, color);
              }
              break;
            }

            case 'MANIFOLD_SOLID_BREP': {
              // Apply color to all faces in the solid
              const shellRefs = parseRefList(targetEntity.data);
              for (const shellRef of shellRefs) {
                const shellEntity = entities.get(shellRef);
                if (shellEntity && (shellEntity.type === 'CLOSED_SHELL' || shellEntity.type === 'OPEN_SHELL')) {
                  const faceRefs = parseRefList(shellEntity.data);
                  for (const faceRef of faceRefs) {
                    faceColorMap.set(faceRef, color);
                  }
                }
              }
              break;
            }

            default:
              // Try to apply directly (might be a face)
              faceColorMap.set(targetId, color);
          }
        }
      }
    }
  }

  const uniqueColors = new Set<string>();
  for (const color of faceColorMap.values()) {
    uniqueColors.add(`${color.r.toFixed(3)},${color.g.toFixed(3)},${color.b.toFixed(3)}`);
  }

  const elapsed = performance.now() - startTime;
  console.log(`[buildComprehensiveFaceColorMap] Mapped ${faceColorMap.size} faces to ${uniqueColors.size} unique colors in ${elapsed.toFixed(0)}ms`);

  return faceColorMap;
}
