import { render, RenderOptions } from '@testing-library/react'
import { ReactElement, ReactNode } from 'react'

/**
 * Custom render function that wraps components with providers.
 * Currently minimal, but ready for adding providers like theme/router.
 */
function AllProviders({ children }: { children: ReactNode }) {
  return <>{children}</>
}

function customRender(
  ui: ReactElement,
  options?: Omit<RenderOptions, 'wrapper'>
) {
  return render(ui, { wrapper: AllProviders, ...options })
}

// Re-export everything from testing-library
export * from '@testing-library/react'
export { customRender as render }

/**
 * Helper to create a mock fetch response
 */
export function mockFetchResponse<T>(data: T, options?: { status?: number; ok?: boolean }) {
  const response = {
    ok: options?.ok ?? true,
    status: options?.status ?? 200,
    json: () => Promise.resolve(data),
    text: () => Promise.resolve(JSON.stringify(data)),
  }
  return response as Response
}

/**
 * Helper to wait for async state updates
 */
export async function waitForStateUpdate() {
  await new Promise(resolve => setTimeout(resolve, 0))
}

/**
 * Helper to create a mock pipeline schema
 */
export function createMockPipelineSchema(overrides?: Partial<{
  id: string
  display_name: string
  category: string
  params: Array<{
    name: string
    display_name: string
    type: string
    default: unknown
  }>
}>) {
  return {
    id: overrides?.id ?? 'test-pipeline',
    display_name: overrides?.display_name ?? 'Test Pipeline',
    category: overrides?.category ?? 'image',
    params: overrides?.params ?? [
      {
        name: 'prompt',
        display_name: 'Prompt',
        type: 'textarea',
        default: '',
      },
      {
        name: 'steps',
        display_name: 'Steps',
        type: 'slider',
        default: 20,
        min: 1,
        max: 100,
        step: 1,
      },
    ],
  }
}
