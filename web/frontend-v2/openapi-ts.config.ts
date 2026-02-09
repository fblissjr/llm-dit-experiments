import { defineConfig } from '@hey-api/openapi-ts';

export default defineConfig({
  client: false,
  input: 'openapi.json',
  output: {
    path: 'src/generated',
    format: 'prettier',
  },
  types: {
    enums: 'javascript',
  },
  services: false,
});
