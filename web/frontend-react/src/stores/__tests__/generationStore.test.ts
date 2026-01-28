/**
 * Generation Store Tests
 *
 * Tests for form state management, generation lifecycle, and error handling.
 */

import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest'
import { useGenerationStore } from '../generationStore'
import { useHistoryStore } from '../historyStore'
import { usePipelineStore } from '../pipelineStore'
import type { PipelineSchema, FormValues } from '@/types'

// Mock historyStore
vi.mock('../historyStore', () => ({
  useHistoryStore: {
    getState: vi.fn(() => ({
      addItem: vi.fn(),
      getItemsByPipeline: vi.fn(() => []),
    })),
  },
}))

// Mock pipelineStore for variant injection tests
vi.mock('../pipelineStore', () => ({
  usePipelineStore: {
    getState: vi.fn(() => ({
      serverDefaults: {},
    })),
  },
}))

// Mock pipeline schema
const mockSchema: PipelineSchema = {
  id: 'zimage',
  name: 'Z-Image',
  description: 'Fast text-to-image',
  output_type: 'image',
  color: 'blue',
  params: [
    { id: 'prompt', type: 'textarea', label: 'Prompt', default: '', group: 'basic' },
    { id: 'steps', type: 'slider', label: 'Steps', default: 20, min: 1, max: 100, group: 'basic' },
    { id: 'guidance_scale', type: 'slider', label: 'CFG', default: 3.0, min: 0, max: 20, group: 'basic' },
    { id: 'width', type: 'number', label: 'Width', default: 1024, group: 'basic' },
    { id: 'height', type: 'number', label: 'Height', default: 1024, group: 'basic' },
  ],
  supports_history: true,
  supports_img2img: false,
  supports_reference_images: false,
  supports_streaming: false,
  endpoint: '/api/generate',
  category: 'image',
}

describe('generationStore', () => {
  beforeEach(() => {
    // Reset store state
    useGenerationStore.setState({
      formValues: {},
      status: 'idle',
      progress: null,
      currentResult: null,
      error: null,
      abortController: null,
    })

    vi.clearAllMocks()
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  describe('initial state', () => {
    it('starts in idle state with no values', () => {
      const state = useGenerationStore.getState()

      expect(state.formValues).toEqual({})
      expect(state.status).toBe('idle')
      expect(state.progress).toBeNull()
      expect(state.currentResult).toBeNull()
      expect(state.error).toBeNull()
      expect(state.abortController).toBeNull()
    })
  })

  describe('setFormValue', () => {
    it('creates pipeline entry if not exists', () => {
      useGenerationStore.getState().setFormValue('zimage', 'prompt', 'Hello')

      const state = useGenerationStore.getState()
      expect(state.formValues['zimage']).toBeDefined()
      expect(state.formValues['zimage']['prompt']).toBe('Hello')
    })

    it('sets value without overwriting others', () => {
      useGenerationStore.getState().setFormValue('zimage', 'prompt', 'Hello')
      useGenerationStore.getState().setFormValue('zimage', 'steps', 30)

      const state = useGenerationStore.getState()
      expect(state.formValues['zimage']['prompt']).toBe('Hello')
      expect(state.formValues['zimage']['steps']).toBe(30)
    })

    it('overwrites existing value', () => {
      useGenerationStore.getState().setFormValue('zimage', 'prompt', 'Hello')
      useGenerationStore.getState().setFormValue('zimage', 'prompt', 'World')

      const state = useGenerationStore.getState()
      expect(state.formValues['zimage']['prompt']).toBe('World')
    })

    it('keeps pipelines separate', () => {
      useGenerationStore.getState().setFormValue('zimage', 'steps', 30)
      useGenerationStore.getState().setFormValue('ltx2', 'steps', 40)

      const state = useGenerationStore.getState()
      expect(state.formValues['zimage']['steps']).toBe(30)
      expect(state.formValues['ltx2']['steps']).toBe(40)
    })
  })

  describe('setFormValues', () => {
    it('merges values into pipeline', () => {
      useGenerationStore.getState().setFormValue('zimage', 'prompt', 'Hello')
      useGenerationStore.getState().setFormValues('zimage', {
        steps: 30,
        guidance_scale: 7.0,
      })

      const state = useGenerationStore.getState()
      expect(state.formValues['zimage']['prompt']).toBe('Hello')
      expect(state.formValues['zimage']['steps']).toBe(30)
      expect(state.formValues['zimage']['guidance_scale']).toBe(7.0)
    })

    it('overwrites existing values', () => {
      useGenerationStore.getState().setFormValues('zimage', { steps: 20 })
      useGenerationStore.getState().setFormValues('zimage', { steps: 50 })

      const state = useGenerationStore.getState()
      expect(state.formValues['zimage']['steps']).toBe(50)
    })
  })

  describe('resetFormValues', () => {
    it('sets defaults from schema', () => {
      // First set some custom values
      useGenerationStore.getState().setFormValues('zimage', {
        prompt: 'Custom prompt',
        steps: 100,
      })

      // Reset to defaults
      useGenerationStore.getState().resetFormValues('zimage', mockSchema)

      const state = useGenerationStore.getState()
      expect(state.formValues['zimage']['prompt']).toBe('')
      expect(state.formValues['zimage']['steps']).toBe(20)
      expect(state.formValues['zimage']['width']).toBe(1024)
    })

    it('only sets params that have defaults', () => {
      const schemaWithOptional: PipelineSchema = {
        ...mockSchema,
        params: [
          { id: 'prompt', type: 'textarea', label: 'Prompt', group: 'basic' }, // no default
          { id: 'steps', type: 'slider', label: 'Steps', default: 20, group: 'basic' },
        ],
      }

      useGenerationStore.getState().resetFormValues('zimage', schemaWithOptional)

      const state = useGenerationStore.getState()
      expect(state.formValues['zimage']['prompt']).toBeUndefined()
      expect(state.formValues['zimage']['steps']).toBe(20)
    })

    it('injects _variant on reset for zimage pipeline', () => {
      vi.mocked(usePipelineStore.getState).mockReturnValue({
        serverDefaults: { zimage_variant: 'base' },
      } as ReturnType<typeof usePipelineStore.getState>)

      useGenerationStore.getState().resetFormValues('zimage', mockSchema)

      const state = useGenerationStore.getState()
      expect(state.formValues['zimage']['_variant']).toBe('base')
    })

    it('does not inject _variant on reset for non-zimage pipelines', () => {
      vi.mocked(usePipelineStore.getState).mockReturnValue({
        serverDefaults: { zimage_variant: 'base' },
      } as ReturnType<typeof usePipelineStore.getState>)

      useGenerationStore.getState().resetFormValues('ltx2', mockSchema)

      const state = useGenerationStore.getState()
      expect(state.formValues['ltx2']['_variant']).toBeUndefined()
    })

    it('applies variant-aware defaults on reset for zimage', () => {
      vi.mocked(usePipelineStore.getState).mockReturnValue({
        serverDefaults: {
          zimage_variant: 'base',
          steps: 40,
          guidance_scale: 4.0,
          shift: 6.0,
        },
      } as ReturnType<typeof usePipelineStore.getState>)

      useGenerationStore.getState().resetFormValues('zimage', mockSchema)

      const state = useGenerationStore.getState()
      expect(state.formValues['zimage']['steps']).toBe(40)
      expect(state.formValues['zimage']['guidance_scale']).toBe(4.0)
      expect(state.formValues['zimage']['shift']).toBe(6.0)
      expect(state.formValues['zimage']['_variant']).toBe('base')
    })
  })

  describe('restoreFromHistory', () => {
    it('replaces all values for pipeline', () => {
      // First set some values
      useGenerationStore.getState().setFormValues('zimage', {
        prompt: 'Original',
        steps: 20,
      })

      // Restore from history
      const historyParams: FormValues = {
        prompt: 'From history',
        steps: 50,
        guidance_scale: 7.0,
        new_param: 'also restored',
      }
      useGenerationStore.getState().restoreFromHistory('zimage', historyParams)

      const state = useGenerationStore.getState()
      expect(state.formValues['zimage']).toEqual(historyParams)
    })
  })

  describe('getFormValues', () => {
    it('returns stored values if exist', () => {
      useGenerationStore.getState().setFormValues('zimage', {
        prompt: 'Stored prompt',
        steps: 50,
      })

      const values = useGenerationStore.getState().getFormValues('zimage', mockSchema)

      expect(values['prompt']).toBe('Stored prompt')
      expect(values['steps']).toBe(50)
    })

    it('merges stored values with schema defaults', () => {
      // Only store prompt, not steps/guidance_scale
      useGenerationStore.getState().setFormValues('zimage', {
        prompt: 'Just a prompt',
      })

      // Reset mock to return empty serverDefaults (don't override schema)
      vi.mocked(usePipelineStore.getState).mockReturnValue({
        serverDefaults: {},
      } as ReturnType<typeof usePipelineStore.getState>)

      const values = useGenerationStore.getState().getFormValues('zimage', mockSchema)

      // Stored value should be present
      expect(values['prompt']).toBe('Just a prompt')
      // Schema defaults should also be present (not undefined)
      expect(values['steps']).toBe(20)  // Schema default
      expect(values['guidance_scale']).toBe(3.0)  // Schema default
      expect(values['width']).toBe(1024)  // Schema default
    })

    it('returns defaults from schema if no stored values and no server defaults', () => {
      // Reset mock to return empty serverDefaults
      vi.mocked(usePipelineStore.getState).mockReturnValue({
        serverDefaults: {},
      } as ReturnType<typeof usePipelineStore.getState>)

      const values = useGenerationStore.getState().getFormValues('zimage', mockSchema)

      expect(values['prompt']).toBe('')
      expect(values['steps']).toBe(20)  // Schema default
      expect(values['guidance_scale']).toBe(3.0)  // Schema default
      expect(values['width']).toBe(1024)
    })

    it('returns defaults for unset pipeline', () => {
      const values = useGenerationStore.getState().getFormValues('nonexistent', mockSchema)

      expect(values['steps']).toBe(20)
    })

    it('injects _variant from serverDefaults for zimage pipeline', () => {
      // Mock pipelineStore to return base variant
      vi.mocked(usePipelineStore.getState).mockReturnValue({
        serverDefaults: { zimage_variant: 'base' },
      } as ReturnType<typeof usePipelineStore.getState>)

      const values = useGenerationStore.getState().getFormValues('zimage', mockSchema)

      expect(values['_variant']).toBe('base')
    })

    it('injects turbo variant from serverDefaults', () => {
      vi.mocked(usePipelineStore.getState).mockReturnValue({
        serverDefaults: { zimage_variant: 'turbo' },
      } as ReturnType<typeof usePipelineStore.getState>)

      const values = useGenerationStore.getState().getFormValues('zimage', mockSchema)

      expect(values['_variant']).toBe('turbo')
    })

    it('does not inject _variant for non-zimage pipelines', () => {
      vi.mocked(usePipelineStore.getState).mockReturnValue({
        serverDefaults: { zimage_variant: 'base' },
      } as ReturnType<typeof usePipelineStore.getState>)

      const values = useGenerationStore.getState().getFormValues('ltx2', mockSchema)

      expect(values['_variant']).toBeUndefined()
    })

    it('does not inject _variant when serverDefaults is empty', () => {
      vi.mocked(usePipelineStore.getState).mockReturnValue({
        serverDefaults: {},
      } as ReturnType<typeof usePipelineStore.getState>)

      const values = useGenerationStore.getState().getFormValues('zimage', mockSchema)

      expect(values['_variant']).toBeUndefined()
    })

    it('applies variant-aware steps/guidance_scale/shift from serverDefaults for zimage', () => {
      vi.mocked(usePipelineStore.getState).mockReturnValue({
        serverDefaults: {
          zimage_variant: 'base',
          steps: 40,
          guidance_scale: 4.0,
          shift: 6.0,
        },
      } as ReturnType<typeof usePipelineStore.getState>)

      const values = useGenerationStore.getState().getFormValues('zimage', mockSchema)

      // Should use server defaults, not schema defaults
      expect(values['steps']).toBe(40)  // Not schema's 20
      expect(values['guidance_scale']).toBe(4.0)  // Not schema's 3.0
      expect(values['shift']).toBe(6.0)  // From server
      expect(values['_variant']).toBe('base')
    })

    it('applies turbo variant defaults from serverDefaults', () => {
      vi.mocked(usePipelineStore.getState).mockReturnValue({
        serverDefaults: {
          zimage_variant: 'turbo',
          steps: 9,
          guidance_scale: 0.0,
          shift: 3.0,
        },
      } as ReturnType<typeof usePipelineStore.getState>)

      const values = useGenerationStore.getState().getFormValues('zimage', mockSchema)

      expect(values['steps']).toBe(9)
      expect(values['guidance_scale']).toBe(0.0)
      expect(values['shift']).toBe(3.0)
      expect(values['_variant']).toBe('turbo')
    })
  })

  describe('getTimeEstimate', () => {
    it('returns default estimate when no history exists', () => {
      vi.mocked(useHistoryStore.getState).mockReturnValue({
        addItem: vi.fn(),
        getItemsByPipeline: vi.fn(() => []),
      } as unknown as ReturnType<typeof useHistoryStore.getState>)

      const estimate = useGenerationStore.getState().getTimeEstimate('zimage')

      expect(estimate.estimatedSeconds).toBe(30)
      expect(estimate.basedOn).toBe('default')
      expect(estimate.confidence).toBe('low')
    })

    it('calculates estimate from history with medium confidence', () => {
      const mockHistoryItems = [
        {
          params: { steps: 20 },
          result: { durationMs: 5000 },
        },
        {
          params: { steps: 20 },
          result: { durationMs: 6000 },
        },
      ]

      vi.mocked(useHistoryStore.getState).mockReturnValue({
        addItem: vi.fn(),
        getItemsByPipeline: vi.fn(() => mockHistoryItems),
      } as unknown as ReturnType<typeof useHistoryStore.getState>)

      // Set form values to have similar steps
      useGenerationStore.getState().setFormValue('zimage', 'steps', 20)

      const estimate = useGenerationStore.getState().getTimeEstimate('zimage')

      // Should estimate around 5-6 seconds (weighted average)
      expect(estimate.estimatedSeconds).toBeGreaterThanOrEqual(5)
      expect(estimate.estimatedSeconds).toBeLessThanOrEqual(6)
      expect(estimate.basedOn).toBe('history')
      expect(estimate.confidence).toBe('medium')
    })

    it('calculates estimate with high confidence when 5+ items', () => {
      const mockHistoryItems = Array.from({ length: 5 }, (_, i) => ({
        params: { steps: 20 },
        result: { durationMs: 5000 + i * 200 }, // 5000, 5200, 5400, 5600, 5800
      }))

      vi.mocked(useHistoryStore.getState).mockReturnValue({
        addItem: vi.fn(),
        getItemsByPipeline: vi.fn(() => mockHistoryItems),
      } as unknown as ReturnType<typeof useHistoryStore.getState>)

      const estimate = useGenerationStore.getState().getTimeEstimate('zimage')

      expect(estimate.basedOn).toBe('history')
      expect(estimate.confidence).toBe('high')
    })

    it('weights similar step counts more heavily', () => {
      const mockHistoryItems = [
        { params: { steps: 50 }, result: { durationMs: 10000 } }, // Different steps
        { params: { steps: 20 }, result: { durationMs: 5000 } },  // Similar steps
      ]

      vi.mocked(useHistoryStore.getState).mockReturnValue({
        addItem: vi.fn(),
        getItemsByPipeline: vi.fn(() => mockHistoryItems),
      } as unknown as ReturnType<typeof useHistoryStore.getState>)

      useGenerationStore.getState().setFormValue('zimage', 'steps', 20)

      const estimate = useGenerationStore.getState().getTimeEstimate('zimage')

      // Should be closer to 5s (similar steps) than 10s (different steps)
      expect(estimate.estimatedSeconds).toBeLessThan(8)
    })
  })

  describe('generate (non-streaming)', () => {
    it('sets generating state and clears previous result', async () => {
      // Use a promise we can control
      let resolveIt: (value: Response) => void
      const controllablePromise = new Promise<Response>((resolve) => {
        resolveIt = resolve
      })
      vi.mocked(global.fetch).mockReturnValueOnce(controllablePromise)

      // Start generation (don't await yet)
      const generatePromise = useGenerationStore
        .getState()
        .generate('zimage', '/api/generate', false)

      // Check immediate state change (synchronous)
      await new Promise((r) => setTimeout(r, 10)) // Let setState flush
      const state = useGenerationStore.getState()
      expect(state.status).toBe('generating')
      expect(state.progress).toBeNull()
      expect(state.currentResult).toBeNull()
      expect(state.error).toBeNull()
      expect(state.abortController).not.toBeNull()

      // Resolve the fetch to complete the test
      resolveIt!({
        ok: true,
        json: () => Promise.resolve({ url: 'test' }),
      } as Response)
      await generatePromise
    })

    it('completes successfully and adds to history', async () => {
      const mockAddItem = vi.fn()
      vi.mocked(useHistoryStore.getState).mockReturnValue({
        addItem: mockAddItem,
      } as ReturnType<typeof useHistoryStore.getState>)

      vi.mocked(global.fetch).mockResolvedValueOnce({
        ok: true,
        json: () =>
          Promise.resolve({
            id: 'gen-123',
            url: 'https://example.com/result.png',
            seed: 42,
          }),
      } as Response)

      useGenerationStore.getState().setFormValue('zimage', 'prompt', 'Test')
      await useGenerationStore.getState().generate('zimage', '/api/generate', false)

      const state = useGenerationStore.getState()
      expect(state.status).toBe('completed')
      expect(state.currentResult).not.toBeNull()
      expect(state.currentResult?.id).toBe('gen-123')
      expect(state.currentResult?.pipelineId).toBe('zimage')
      expect(state.abortController).toBeNull()

      expect(mockAddItem).toHaveBeenCalledTimes(1)
    })

    it('handles error response', async () => {
      vi.mocked(global.fetch).mockResolvedValueOnce({
        ok: false,
        json: () => Promise.resolve({ error: 'Model not loaded' }),
      } as Response)

      await useGenerationStore.getState().generate('zimage', '/api/generate', false)

      const state = useGenerationStore.getState()
      expect(state.status).toBe('error')
      expect(state.error?.message).toBe('Model not loaded')
      expect(state.error?.recoverable).toBe(true)
    })

    it('handles network error', async () => {
      vi.mocked(global.fetch).mockRejectedValueOnce(new Error('Network error'))

      await useGenerationStore.getState().generate('zimage', '/api/generate', false)

      const state = useGenerationStore.getState()
      expect(state.status).toBe('error')
      expect(state.error?.message).toBe('Network error')
    })

    it('sends form values in request body', async () => {
      vi.mocked(global.fetch).mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ url: 'https://example.com/result.png' }),
      } as Response)

      useGenerationStore.getState().setFormValues('zimage', {
        prompt: 'Test prompt',
        steps: 30,
      })
      await useGenerationStore.getState().generate('zimage', '/api/generate', false)

      expect(global.fetch).toHaveBeenCalledWith(
        '/api/generate',
        expect.objectContaining({
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ prompt: 'Test prompt', steps: 30 }),
        })
      )
    })
  })

  describe('cancelGeneration', () => {
    it('aborts in-progress generation', async () => {
      // Mock fetch to throw AbortError when aborted
      vi.mocked(global.fetch).mockImplementationOnce((_url, options) => {
        return new Promise<Response>((_resolve, reject) => {
          // Listen for abort signal
          const signal = options?.signal
          if (signal) {
            signal.addEventListener('abort', () => {
              const error = new Error('Aborted')
              error.name = 'AbortError'
              reject(error)
            })
          }
        })
      })

      // Start generation
      const generatePromise = useGenerationStore
        .getState()
        .generate('zimage', '/api/generate', false)

      // Wait for fetch to be called and state to update
      await new Promise((r) => setTimeout(r, 10))
      expect(useGenerationStore.getState().status).toBe('generating')

      // Cancel
      useGenerationStore.getState().cancelGeneration()

      // Wait for the promise to settle
      await generatePromise

      const state = useGenerationStore.getState()
      expect(state.status).toBe('cancelled')
      expect(state.abortController).toBeNull()
    })

    it('does nothing if no generation in progress', () => {
      // Should not throw
      expect(() => {
        useGenerationStore.getState().cancelGeneration()
      }).not.toThrow()
    })
  })

  describe('generate (streaming)', () => {
    it('parses SSE progress events', async () => {
      const mockReader = {
        read: vi.fn()
          .mockResolvedValueOnce({
            done: false,
            value: new TextEncoder().encode(
              'data: {"type":"progress","step":5,"total_steps":20}\n'
            ),
          })
          .mockResolvedValueOnce({
            done: false,
            value: new TextEncoder().encode(
              'data: {"type":"complete","id":"gen-sse","pipeline_id":"ltx2","url":"test.mp4","seed":123}\n'
            ),
          })
          .mockResolvedValueOnce({ done: true, value: undefined }),
      }

      vi.mocked(global.fetch).mockResolvedValueOnce({
        ok: true,
        body: { getReader: () => mockReader },
      } as unknown as Response)

      await useGenerationStore.getState().generate('ltx2', '/api/generate', true)

      // Check final state
      const state = useGenerationStore.getState()
      expect(state.status).toBe('completed')
      expect(state.currentResult).not.toBeNull()
      expect(state.currentResult?.id).toBe('gen-sse')
    })

    it('handles streaming error', async () => {
      vi.mocked(global.fetch).mockResolvedValueOnce({
        ok: false,
        json: () => Promise.resolve({ error: 'Streaming failed' }),
      } as Response)

      await useGenerationStore.getState().generate('ltx2', '/api/generate', true)

      const state = useGenerationStore.getState()
      expect(state.status).toBe('error')
      expect(state.error?.message).toBe('Streaming failed')
    })
  })
})
