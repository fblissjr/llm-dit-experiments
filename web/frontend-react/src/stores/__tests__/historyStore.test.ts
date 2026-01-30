/**
 * History Store Tests
 *
 * Tests for history management, comparison mode, and localStorage persistence.
 */

import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest'
import { useHistoryStore } from '../historyStore'
import { usePipelineStore } from '../pipelineStore'
import type { GenerationResult, HistoryItem } from '@/types'
import type { PipelineSchema } from '@/types/pipeline'

// Mock pipelineStore
vi.mock('../pipelineStore', () => ({
  usePipelineStore: {
    getState: vi.fn(),
  },
}))

// Mock pipeline data
const mockZImagePipeline: PipelineSchema = {
  id: 'zimage',
  name: 'Z-Image',
  description: 'Fast text-to-image',
  output_type: 'image',
  color: 'blue',
  params: [],
  supports_history: true,
  supports_img2img: false,
  supports_reference_images: false,
  supports_streaming: false,
  endpoint: '/api/generate',
  category: 'image',
}

const mockLtx2Pipeline: PipelineSchema = {
  id: 'ltx2',
  name: 'LTX-2',
  description: 'Text-to-video',
  output_type: 'video',
  color: 'purple',
  params: [],
  supports_history: true,
  supports_img2img: true,
  supports_reference_images: true,
  supports_streaming: true,
  endpoint: '/api/generate',
  category: 'video',
}

// Create mock generation result
function createMockResult(overrides?: Partial<GenerationResult>): GenerationResult {
  return {
    id: overrides?.id ?? `gen-${Date.now()}`,
    pipelineId: overrides?.pipelineId ?? 'zimage',
    outputType: overrides?.outputType ?? 'image',
    urls: overrides?.urls ?? ['https://example.com/image.png'],
    thumbnailUrl: overrides?.thumbnailUrl ?? 'https://example.com/thumb.png',
    params: overrides?.params ?? {
      prompt: 'A beautiful sunset',
      steps: 30,
      guidance_scale: 3.0,
      width: 1024,
      height: 1024,
    },
    seed: overrides?.seed ?? 12345,
    durationMs: overrides?.durationMs ?? 5000,
    timestamp: overrides?.timestamp ?? Date.now(),
  }
}

describe('historyStore', () => {
  beforeEach(() => {
    // Reset store state
    useHistoryStore.setState({
      items: [],
      selectedForCompare: [],
      isCompareMode: false,
    })

    // Mock pipelineStore.getState to return valid pipelines
    vi.mocked(usePipelineStore.getState).mockReturnValue({
      pipelines: {
        zimage: mockZImagePipeline,
        ltx2: mockLtx2Pipeline,
      },
      selectedPipelineId: 'zimage',
      isLoading: false,
      error: null,
      selectedPipeline: mockZImagePipeline,
      pipelinesByCategory: {},
      fetchPipelines: vi.fn(),
      selectPipeline: vi.fn(),
      getPipeline: (id: string) => {
        const pipelines: Record<string, PipelineSchema> = {
          zimage: mockZImagePipeline,
          ltx2: mockLtx2Pipeline,
        }
        return pipelines[id]
      },
    })

    vi.clearAllMocks()
  })

  afterEach(() => {
    vi.clearAllTimers()
  })

  describe('initial state', () => {
    it('starts with empty history', () => {
      const state = useHistoryStore.getState()

      expect(state.items).toEqual([])
      expect(state.selectedForCompare).toEqual([])
      expect(state.isCompareMode).toBe(false)
    })
  })

  describe('addItem', () => {
    it('adds item to front of history', () => {
      const result1 = createMockResult({ id: 'gen-1' })
      const result2 = createMockResult({ id: 'gen-2' })

      useHistoryStore.getState().addItem(result1)
      useHistoryStore.getState().addItem(result2)

      const state = useHistoryStore.getState()
      expect(state.items).toHaveLength(2)
      expect(state.items[0].id).toBe('gen-2') // Most recent first
      expect(state.items[1].id).toBe('gen-1')
    })

    it('creates history item with pipeline metadata', () => {
      const result = createMockResult({
        id: 'gen-1',
        pipelineId: 'zimage',
        params: { prompt: 'Test prompt' },
      })

      useHistoryStore.getState().addItem(result)

      const item = useHistoryStore.getState().items[0]
      expect(item.pipelineName).toBe('Z-Image')
      expect(item.pipelineColor).toBe('blue')
    })

    it('truncates long prompts', () => {
      const longPrompt = 'A'.repeat(100)
      const result = createMockResult({
        params: { prompt: longPrompt },
      })

      useHistoryStore.getState().addItem(result)

      const item = useHistoryStore.getState().items[0]
      expect(item.prompt).toBe(longPrompt)
      expect(item.shortPrompt.length).toBeLessThanOrEqual(50)
      expect(item.shortPrompt).toContain('...')
    })

    it('extracts key params for display', () => {
      const result = createMockResult({
        params: {
          prompt: 'Test',
          steps: 30,
          guidance_scale: 7.0,
          width: 512,
          height: 512,
        },
      })

      useHistoryStore.getState().addItem(result)

      const item = useHistoryStore.getState().items[0]
      expect(item.keyParams).toContain('30 steps')
      expect(item.keyParams).toContain('CFG 7')
      expect(item.keyParams).toContain('512×512')
    })

    it('limits history to MAX_HISTORY_ITEMS', () => {
      // Add 105 items (more than the 100 limit)
      for (let i = 0; i < 105; i++) {
        const result = createMockResult({ id: `gen-${i}` })
        useHistoryStore.getState().addItem(result)
      }

      const state = useHistoryStore.getState()
      expect(state.items.length).toBeLessThanOrEqual(100)
      // Most recent should be preserved
      expect(state.items[0].id).toBe('gen-104')
    })
  })

  describe('removeItem', () => {
    it('removes item by id', () => {
      const result1 = createMockResult({ id: 'gen-1' })
      const result2 = createMockResult({ id: 'gen-2' })

      useHistoryStore.getState().addItem(result1)
      useHistoryStore.getState().addItem(result2)
      useHistoryStore.getState().removeItem('gen-1')

      const state = useHistoryStore.getState()
      expect(state.items).toHaveLength(1)
      expect(state.items[0].id).toBe('gen-2')
    })

    it('removes item from compare selection', () => {
      const result = createMockResult({ id: 'gen-1' })

      useHistoryStore.getState().addItem(result)
      useHistoryStore.setState({ selectedForCompare: ['gen-1'] })
      useHistoryStore.getState().removeItem('gen-1')

      const state = useHistoryStore.getState()
      expect(state.selectedForCompare).not.toContain('gen-1')
    })
  })

  describe('clearHistory', () => {
    it('clears all items and resets compare state', () => {
      const result = createMockResult({ id: 'gen-1' })

      useHistoryStore.getState().addItem(result)
      useHistoryStore.setState({
        selectedForCompare: ['gen-1'],
        isCompareMode: true,
      })
      useHistoryStore.getState().clearHistory()

      const state = useHistoryStore.getState()
      expect(state.items).toEqual([])
      expect(state.selectedForCompare).toEqual([])
      expect(state.isCompareMode).toBe(false)
    })
  })

  describe('comparison mode', () => {
    beforeEach(() => {
      // Pre-populate with items
      useHistoryStore.getState().addItem(createMockResult({ id: 'gen-1' }))
      useHistoryStore.getState().addItem(createMockResult({ id: 'gen-2' }))
      useHistoryStore.getState().addItem(createMockResult({ id: 'gen-3' }))
    })

    it('toggleCompareMode toggles state', () => {
      useHistoryStore.getState().toggleCompareMode()
      expect(useHistoryStore.getState().isCompareMode).toBe(true)

      useHistoryStore.getState().toggleCompareMode()
      expect(useHistoryStore.getState().isCompareMode).toBe(false)
    })

    it('toggleCompareMode clears selection when turning off', () => {
      useHistoryStore.setState({
        isCompareMode: true,
        selectedForCompare: ['gen-1', 'gen-2'],
      })

      useHistoryStore.getState().toggleCompareMode()

      expect(useHistoryStore.getState().selectedForCompare).toEqual([])
    })

    it('selectForCompare adds item id', () => {
      useHistoryStore.getState().selectForCompare('gen-1')

      expect(useHistoryStore.getState().selectedForCompare).toContain('gen-1')
    })

    it('selectForCompare limits to 2 items', () => {
      useHistoryStore.getState().selectForCompare('gen-1')
      useHistoryStore.getState().selectForCompare('gen-2')
      useHistoryStore.getState().selectForCompare('gen-3')

      expect(useHistoryStore.getState().selectedForCompare).toHaveLength(2)
      expect(useHistoryStore.getState().selectedForCompare).not.toContain('gen-3')
    })

    it('selectForCompare prevents duplicates', () => {
      useHistoryStore.getState().selectForCompare('gen-1')
      useHistoryStore.getState().selectForCompare('gen-1')

      expect(useHistoryStore.getState().selectedForCompare).toEqual(['gen-1'])
    })

    it('deselectForCompare removes item id', () => {
      useHistoryStore.setState({ selectedForCompare: ['gen-1', 'gen-2'] })
      useHistoryStore.getState().deselectForCompare('gen-1')

      expect(useHistoryStore.getState().selectedForCompare).toEqual(['gen-2'])
    })

    it('clearCompareSelection clears selection without affecting mode', () => {
      useHistoryStore.setState({
        isCompareMode: true,
        selectedForCompare: ['gen-1', 'gen-2'],
      })
      useHistoryStore.getState().clearCompareSelection()

      const state = useHistoryStore.getState()
      expect(state.selectedForCompare).toEqual([])
      expect(state.isCompareMode).toBe(true)
    })
  })

  describe('getComparisonDiff', () => {
    it('returns empty array when less than 2 items selected', () => {
      useHistoryStore.setState({ selectedForCompare: ['gen-1'] })

      const diff = useHistoryStore.getState().getComparisonDiff()
      expect(diff).toEqual([])
    })

    it('returns parameter differences between two items', () => {
      const result1 = createMockResult({
        id: 'gen-1',
        params: { prompt: 'Test', steps: 30, guidance_scale: 3.0 },
      })
      const result2 = createMockResult({
        id: 'gen-2',
        params: { prompt: 'Test', steps: 50, guidance_scale: 7.0 },
      })

      useHistoryStore.getState().addItem(result1)
      useHistoryStore.getState().addItem(result2)
      useHistoryStore.setState({ selectedForCompare: ['gen-1', 'gen-2'] })

      const diff = useHistoryStore.getState().getComparisonDiff()

      // Prompt is same, should not be in diff
      expect(diff.find((d) => d.key === 'prompt')).toBeUndefined()

      // Steps and guidance_scale differ
      const stepsDiff = diff.find((d) => d.key === 'steps')
      expect(stepsDiff).toBeDefined()
      expect(stepsDiff?.valueA).toBe(30)
      expect(stepsDiff?.valueB).toBe(50)

      const guidanceDiff = diff.find((d) => d.key === 'guidance_scale')
      expect(guidanceDiff).toBeDefined()
      expect(guidanceDiff?.valueA).toBe(3.0)
      expect(guidanceDiff?.valueB).toBe(7.0)
    })

    it('formats diff labels from snake_case', () => {
      const result1 = createMockResult({
        id: 'gen-1',
        params: { guidance_scale: 3.0 },
      })
      const result2 = createMockResult({
        id: 'gen-2',
        params: { guidance_scale: 7.0 },
      })

      useHistoryStore.getState().addItem(result1)
      useHistoryStore.getState().addItem(result2)
      useHistoryStore.setState({ selectedForCompare: ['gen-1', 'gen-2'] })

      const diff = useHistoryStore.getState().getComparisonDiff()
      const guidanceDiff = diff.find((d) => d.key === 'guidance_scale')

      expect(guidanceDiff?.label).toBe('Guidance Scale')
    })
  })

  describe('query methods', () => {
    beforeEach(() => {
      useHistoryStore.getState().addItem(
        createMockResult({ id: 'gen-1', pipelineId: 'zimage' })
      )
      useHistoryStore.getState().addItem(
        createMockResult({ id: 'gen-2', pipelineId: 'ltx2' })
      )
      useHistoryStore.getState().addItem(
        createMockResult({ id: 'gen-3', pipelineId: 'zimage' })
      )
    })

    it('getItem returns item by id', () => {
      const item = useHistoryStore.getState().getItem('gen-2')

      expect(item).toBeDefined()
      expect(item?.id).toBe('gen-2')
    })

    it('getItem returns undefined for unknown id', () => {
      const item = useHistoryStore.getState().getItem('nonexistent')

      expect(item).toBeUndefined()
    })

    it('getItemsByPipeline filters by pipeline id', () => {
      const zimageItems = useHistoryStore.getState().getItemsByPipeline('zimage')
      const ltx2Items = useHistoryStore.getState().getItemsByPipeline('ltx2')

      expect(zimageItems).toHaveLength(2)
      expect(ltx2Items).toHaveLength(1)
    })
  })

  describe('useAsInput', () => {
    beforeEach(() => {
      useHistoryStore.getState().addItem(
        createMockResult({
          id: 'gen-1',
          thumbnailUrl: 'https://example.com/result.png',
        })
      )
    })

    it('returns thumbnail URL for pipeline that supports img2img', () => {
      const url = useHistoryStore.getState().useAsInput('gen-1', 'ltx2')

      expect(url).toBe('https://example.com/result.png')
    })

    it('returns null for pipeline that does not support img2img', () => {
      const url = useHistoryStore.getState().useAsInput('gen-1', 'zimage')

      expect(url).toBeNull()
    })

    it('returns null for nonexistent item', () => {
      const url = useHistoryStore.getState().useAsInput('nonexistent', 'ltx2')

      expect(url).toBeNull()
    })
  })
})
