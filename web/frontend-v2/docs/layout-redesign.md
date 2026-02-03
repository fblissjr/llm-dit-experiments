# frontend layout redesign - left navigation with model manager

*last updated: 2026-02-03*

## overview

The frontend has been redesigned to use a left navigation sidebar instead of horizontal tabs. The new layout includes an integrated model manager that gives users explicit control over which models are loaded/unloaded in VRAM.

## key improvements

### 1. vertical space efficiency
- **Before**: Horizontal tabs at the top consumed vertical space
- **After**: Collapsible left sidebar maximizes form area
- **Result**: More room for parameters and generated results

### 2. explicit model management
- **Before**: Models loaded implicitly on first generation
- **After**: Users can preload/unload models before generating
- **Benefit**: Full transparency and control over VRAM usage

### 3. mobile-first responsive design
- **Desktop**: Left sidebar with collapsible sections
- **Mobile**: Bottom navigation bar with sheet modals
- **Touch targets**: Minimum 44px for mobile usability

## architecture

### component structure

```
AppShell
├── LeftNav (desktop only)
│   ├── ModelManager
│   │   ├── VRAM Status Bar
│   │   ├── Image Models
│   │   │   └── ModelCard (per pipeline)
│   │   └── Video Models
│   │       └── ModelCard (per pipeline)
│   ├── Category Switcher (Image/Video)
│   └── Pipeline List
├── Main Content Area
│   ├── PipelineForm
│   ├── ProgressDisplay
│   ├── GenerateButton
│   └── ResultDisplay
├── Sidebar (history, right side)
└── MobileNav (mobile only)
    ├── Bottom Navigation Bar
    └── Model Manager Sheet
```

### state management

**appStore additions**:
```typescript
interface AppState {
  // Model state
  modelStatus: Record<string, ModelStatusResponse>;

  // UI state
  isLeftNavOpen: boolean;

  // Actions
  refreshModelStatus(pipelineId: string): Promise<void>;
  refreshAllModelStatus(): Promise<void>;
  loadPipelineModel(pipelineId: string): Promise<void>;
  unloadPipelineModel(pipelineId: string): Promise<void>;
  toggleLeftNav(): void;
}
```

**ModelStatusResponse**:
```typescript
interface ModelStatusResponse {
  status: 'loaded' | 'unloaded' | 'loading' | 'error';
  vramMB?: number;           // Actual VRAM if loaded
  estimatedVramMB?: number;  // Estimated VRAM if unloaded
  error?: string;            // Error message if status is 'error'
}
```

### api integration

**Backend endpoints** (already exist in `web/server.py`):

```python
GET /api/models/{pipeline_id}/status
  Returns: {"status": "loaded", "vram_mb": 12288, ...}

POST /api/models/{pipeline_id}/load
  Loads model into VRAM
  Returns: {"status": "loaded", "vram_mb": 12288}

POST /api/models/{pipeline_id}/unload
  Unloads model from VRAM
  Returns: {"status": "unloaded"}

GET /api/vram/status
  Returns overall VRAM usage and breakdown
```

**Auto-refresh**:
- Model status: Every 10 seconds
- VRAM status: Every 30 seconds

## progressive disclosure

The left nav has three collapsible sections:

### 1. load models
- VRAM usage bar (color-coded: green < 50%, yellow < 75%, orange < 90%, red >= 90%)
- Image models section
- Video models section
- Each model shows: name, status dot, VRAM usage, load/unload button

### 2. category
- Image/Video tab switcher
- Changes which pipelines are shown in pipeline list

### 3. pipelines
- List of pipelines in current category
- Color-coded by pipeline type
- Shows: name, description, selected indicator

Each section can be collapsed independently to reduce visual clutter.

## responsive behavior

### desktop (>= 768px)
- Left sidebar: 288px wide, collapsible
- Main content: Adjusts margin-left when sidebar open
- History sidebar: 320px wide on right (separate toggle)
- Keyboard shortcut: `Ctrl+B` toggles left nav

### mobile (< 768px)
- Left sidebar: Hidden
- Bottom navigation: Fixed at bottom, 64px height
- Three tabs: Image, Video, Models
- Model manager: Opens as bottom sheet modal
- Main content: Full width with bottom padding

### transitions
- Sidebar: Smooth slide in/out
- Bottom sheet: Slide up from bottom with backdrop
- Content area: Smooth margin adjustment

## keyboard shortcuts

### new shortcut
- `Ctrl+B` - Toggle left navigation (desktop only)

### existing shortcuts
- `Ctrl+Enter` - Generate
- `Ctrl+H` - Toggle history
- `Ctrl+Shift+R` - Reset form

All shortcuts work when not focused on input fields.

## accessibility

### keyboard navigation
- All buttons focusable with Tab
- Enter/Space activates buttons
- Escape closes modals/sheets

### screen readers
- Semantic HTML (`<nav>`, `<aside>`, `<main>`)
- ARIA labels on icon buttons
- Status announcements for model load/unload

### visual indicators
- Focus rings on all interactive elements
- Color-coded status dots (with text labels)
- Loading spinners with accessible animation

## implementation files

### new components
| File | Purpose | Lines |
|------|---------|-------|
| `src/components/models/ModelCard.tsx` | Individual model with load/unload | ~100 |
| `src/components/models/ModelManager.tsx` | Container for all models + VRAM | ~110 |
| `src/components/layout/LeftNav.tsx` | Left sidebar navigation | ~220 |
| `src/components/layout/MobileNav.tsx` | Bottom nav + model sheet | ~120 |

### modified components
| File | Changes |
|------|---------|
| `src/components/layout/AppShell.tsx` | Integrated LeftNav and MobileNav |
| `src/components/layout/Sidebar.tsx` | Adjusted positioning |
| `src/App.tsx` | Removed PipelineSelector, added shortcuts |

### api layer
| File | Changes |
|------|---------|
| `src/api/types.ts` | Added ModelStatus types |
| `src/api/client.ts` | Added model management functions |

### state management
| File | Changes |
|------|---------|
| `src/stores/appStore.ts` | Added model state and actions |
| `src/hooks/useKeyboardShortcuts.ts` | Added Ctrl+B shortcut |

## testing checklist

### desktop flow
- [ ] Left nav opens/closes with collapse button
- [ ] Left nav toggles with `Ctrl+B`
- [ ] Model status shows correctly (loaded/unloaded)
- [ ] Load button triggers loading state + spinner
- [ ] VRAM bar updates after load/unload
- [ ] Pipeline selection works from left nav
- [ ] Category switcher changes pipeline list

### mobile flow
- [ ] Bottom nav shows on mobile
- [ ] Tab switching works (Image/Video/Models)
- [ ] Models tab opens bottom sheet
- [ ] Sheet closes with X button or backdrop tap
- [ ] Model load/unload works from sheet
- [ ] Pipeline header shows on mobile

### error handling
- [ ] Failed model load shows error status
- [ ] Error message displays in ModelCard
- [ ] Retry after error works
- [ ] API timeout doesn't crash UI

### auto-refresh
- [ ] Model status refreshes every 10s
- [ ] VRAM status refreshes every 30s
- [ ] Refresh doesn't cause UI flicker

## future enhancements

### short term
- Model load progress bar (if backend supports)
- Estimated load time display
- Batch load/unload operations

### medium term
- Model presets (common combinations)
- VRAM prediction before loading
- Model dependency warnings

### long term
- Model cache management
- Automatic model unloading (LRU)
- Multi-GPU support in UI

## migration notes

### for users
- **No breaking changes**: All existing functionality preserved
- **New feature**: Model manager is optional - generation still auto-loads if needed
- **Mobile users**: Now have dedicated bottom navigation

### for developers
- `PipelineSelector` component no longer used in main view (only mobile header)
- `TabBar` component deprecated (functionality moved to LeftNav)
- Model state now tracked in appStore (check `modelStatus` field)

## design rationale

### why left sidebar over tabs?

**Vertical space**: Horizontal tabs consumed ~56px of vertical space. On laptop screens (1080p), every pixel matters for form controls and results.

**Scalability**: As pipeline count grows, horizontal tabs become cramped. Left sidebar scales better with scrolling.

**Industry patterns**: Most modern tools (VSCode, Figma, Notion) use left navigation. Users expect it.

**Progressive disclosure**: Sidebar allows collapsible sections. Tabs are always visible.

### why explicit model loading?

**Transparency**: Users should know what's consuming their VRAM.

**Control**: Power users want to preload models before batch generation.

**Education**: Seeing VRAM usage helps users understand model sizes and system limits.

**Debugging**: When generation fails, model status is immediately visible.

### why bottom nav for mobile?

**Thumb zone**: Bottom of screen is easier to reach on large phones.

**Content priority**: Top of screen reserved for content (form/results).

**Sheet modals**: iOS/Android users are familiar with bottom sheets for secondary actions.

**Touch targets**: Bottom nav provides ample space for 44px+ tap areas.

## related documentation

- [Design System](/home/fbliss/workspace/llm-dit-experiments/internal/web/docs/design_system.md) - Colors, spacing, components
- [Web Patterns](/home/fbliss/workspace/llm-dit-experiments/internal/web/docs/web_patterns.md) - State management patterns
- [Server API](/home/fbliss/workspace/llm-dit-experiments/web/server.py) - Backend endpoints reference
