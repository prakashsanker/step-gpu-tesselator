// Minimal Node.js 'fs' type declarations so TypeScript accepts `import * as fs from 'fs'`
// without needing full @types/node. At runtime, the real Node 'fs' module is used.

declare module 'fs' {
  export function readFileSync(
    path: string | number | URL,
    options?: { encoding?: string; flag?: string } | string
  ): string;
}



