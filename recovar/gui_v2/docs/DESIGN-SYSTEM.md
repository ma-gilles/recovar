# Design System

Concrete rules for the GUI's visual language. Everything here is implemented via Tailwind CSS utility classes and Shadcn/ui component variants. No custom CSS files.

---

## Color Palette

Based on Tailwind's zinc scale (dark theme). All colors are CSS variables in `tailwind.config.ts` so they can be overridden.

### Surfaces
| Token | Tailwind | Hex | Use |
|-------|----------|-----|-----|
| `bg-app` | `zinc-950` | `#09090b` | App background |
| `bg-surface` | `zinc-900` | `#18181b` | Cards, panels, sidebar |
| `bg-surface-hover` | `zinc-800` | `#27272a` | Hover state on surfaces |
| `bg-surface-active` | `zinc-700` | `#3f3f46` | Active/selected state |
| `bg-input` | `zinc-800` | `#27272a` | Form input backgrounds |

### Text
| Token | Tailwind | Use |
|-------|----------|-----|
| `text-primary` | `zinc-50` | Headings, important labels |
| `text-secondary` | `zinc-400` | Body text, descriptions |
| `text-muted` | `zinc-500` | Placeholders, disabled text |
| `text-inverse` | `zinc-950` | Text on light backgrounds (badges) |

### Status Colors
| Status | Color | Tailwind | Icon |
|--------|-------|----------|------|
| Running | Blue | `blue-500` | Animated spinner |
| Completed | Green | `emerald-500` | Checkmark |
| Failed | Red | `red-500` | X circle |
| Queued | Yellow | `amber-500` | Clock |
| Cancelled | Gray | `zinc-500` | Minus circle |
| Warning | Orange | `orange-500` | Triangle exclamation |

These 6 colors are used consistently everywhere: sidebar icons, status badges, log highlighting, progress indicators.

### Accent
| Token | Tailwind | Use |
|-------|----------|-----|
| `accent` | `blue-600` | Primary buttons, links, active tab underline |
| `accent-hover` | `blue-500` | Hover on primary elements |
| `accent-relion` | `violet-600` | RELION plugin accent (future) |

### Volume Viewer Colors
Multi-volume overlay palette (high contrast on dark background):
1. `sky-400` — first volume (default)
2. `rose-400` — second volume
3. `emerald-400` — third volume
4. `amber-400` — fourth volume

### Scatter Plot Colors
K-means cluster palette: use Plotly's `D3` categorical scheme (10 colors), mapped to WebGL-compatible values for regl-scatterplot.

---

## Typography

Single font stack. No web font loading (reduces latency, works offline).

| Element | Size | Weight | Font |
|---------|------|--------|------|
| Page title | `text-xl` (20px) | `font-semibold` | System sans-serif (`font-sans`) |
| Section heading | `text-lg` (18px) | `font-medium` | System sans-serif |
| Card title | `text-base` (16px) | `font-medium` | System sans-serif |
| Body text | `text-sm` (14px) | `font-normal` | System sans-serif |
| Small / caption | `text-xs` (12px) | `font-normal` | System sans-serif |
| Code / paths / CLI | `text-sm` (14px) | `font-normal` | System monospace (`font-mono`) |
| Log output | `text-xs` (12px) | `font-normal` | System monospace |

System sans-serif = `-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif`

---

## Spacing

Tailwind's default 4px scale. Consistent spacing rules:

| Context | Spacing |
|---------|---------|
| Between sections | `space-y-6` (24px) |
| Between cards/panels | `gap-4` (16px) |
| Card internal padding | `p-4` (16px) |
| Between form fields | `space-y-3` (12px) |
| Between label and input | `space-y-1` (4px) |
| Sidebar item padding | `px-3 py-2` (12px, 8px) |
| Button padding | `px-4 py-2` (16px, 8px) |
| Inline icon-to-text gap | `gap-2` (8px) |

---

## Component Patterns

### Buttons

| Variant | Use | Example |
|---------|-----|---------|
| `default` (filled blue) | Primary action | "Submit Job", "Export Subset" |
| `secondary` (zinc-800 fill) | Secondary action | "Clone Job", "Cancel" |
| `outline` (border only) | Tertiary action | "Show CLI Command", "Advanced" |
| `destructive` (red fill) | Destructive action | "Delete Job" |
| `ghost` (no border) | Inline/toolbar action | Sidebar items, icon buttons |

**Button states:**
| State | Appearance |
|-------|-----------|
| Default | As described above |
| Hover | Lighter shade (`accent-hover` / `zinc-700` / border highlight) |
| Active/pressed | Slightly darker than hover, 1px inset shadow |
| Disabled | `opacity-50 cursor-not-allowed`. No hover effect. Tooltip explains why disabled. |
| Loading | Spinner icon replaces left icon (or appears solo). Text changes to action gerund ("Submitting..."). Button is disabled during loading. `aria-busy="true"`. |

**Focus-within on composite components** (file browser row with nested buttons, job card with action icons): The container gets `ring-2 ring-blue-500/50` when any child element has `focus-visible`. This provides a clear boundary for keyboard navigation.

Destructive actions always require confirmation (modal with the action name typed or a two-click pattern).

### Forms

```
[Label]  [?]          ← label + tooltip icon
[Input field       ]  ← zinc-800 background, zinc-600 border, focus:blue-500 ring
[Helper text]         ← text-xs text-muted, only when non-obvious
[Error text]          ← text-xs text-red-500, only on validation failure
```

Advanced section:
```
[v Advanced]          ← collapsible, chevron icon, zinc-700 text
  [field]             ← same styling, slightly indented
  [field]
```

### Loading States

Every view that fetches data must show a loading state. Three patterns:

1. **Skeleton:** For known-shape content (job list, parameter table). Gray pulsing rectangles matching the expected layout.
2. **Spinner:** For unknown-duration operations (submitting a job, computing a volume). Centered blue spinner with text below: "Submitting job..." / "Computing volume..."
3. **Progress bar:** For operations with known progress (file upload, import scan). Blue bar with percentage.

Never show a blank white/black area while loading. Never show a spinner for more than 30 seconds without an explanatory message.

### Error States

Two patterns:

1. **Inline error (form validation, minor issues):** Red text below the affected element. Icon: red circle-X. Does not replace the content — appears alongside it.

2. **Full error (page-level, API failure, crash):**
   ```
   [X icon]
   Something went wrong
   [Technical error message in monospace]
   [Retry] [Report Issue]
   ```
   Red accent. Centered in the main panel. Always shows a retry action. "Report Issue" links to the GitHub issues page.

### Empty States

Every list/grid view has an empty state:

```
[Icon]
No {items} yet
[Description of how to create the first one]
[+ Create {item}]
```

Examples:
- "No jobs yet. Submit your first pipeline job to get started." [+ New Job]
- "No subsets yet. Select particles in the latent space explorer to create a subset."
- "No masks in this project. Create a mask in the mask editor or import a .mrc file."

### Toasts / Notifications

| Variant | Border class | Icon | Dismiss |
|---------|-------------|------|---------|
| Success | `border-l-4 border-emerald-500` | Checkmark (`emerald-500`) | Auto 5s |
| Error | `border-l-4 border-red-500` | X circle (`red-500`) | Persistent. "Details" expandable. |
| Warning | `border-l-4 border-orange-500` | Triangle (`orange-500`) | Auto 8s |
| Info | `border-l-4 border-blue-500` | Info circle (`blue-500`) | Auto 5s |

Container: `bg-zinc-900 rounded-lg shadow-lg p-4 min-w-[320px]`. Position: bottom-right corner, stacked. Maximum 3 visible. `aria-live="polite"` for screen readers.

### Destructive Action Confirmation

For delete, cancel, or any irreversible action:

```
Modal:
  "Delete job P003?"
  "This will permanently delete the job and all its output files ({size}).
   This action cannot be undone."
  [Cancel]  [Delete]  ← Delete button is red, on the right
```

The destructive button is always:
- Red (`destructive` variant)
- On the right side of the modal
- Requires explicit click (no keyboard shortcut to confirm)
- For high-impact actions (delete project): require typing the project name

---

## Layout Rules

### Sidebar
- Width: 240px default, collapsible to 48px (icon-only)
- Separator between sections (1px zinc-800 line)
- Active item: `bg-surface-active` + `text-primary` + left border accent
- Hover: `bg-surface-hover`
- Scroll: sidebar scrolls independently of main panel

### Main Panel
- Max width: none (fills remaining space)
- Content max width: 1400px, centered with `mx-auto` for very wide screens
- Padding: `p-6`

### Responsive Behavior
- Minimum supported width: 1024px (tablet landscape / small laptop)
- Below 1024px: sidebar collapses to icon-only by default
- Below 768px: not officially supported (show warning banner). The GUI is designed for desktop/laptop use.

---

## Accessibility Targets

- **Keyboard navigation:** All interactive elements reachable via Tab. Radix primitives handle this by default.
- **Focus rings:** Visible `ring-2 ring-blue-500 ring-offset-2 ring-offset-zinc-950` on focus-visible.
- **Color contrast:** All text meets WCAG 2.1 AA (4.5:1 for normal text, 3:1 for large text). The zinc-400 on zinc-950 combination is 5.9:1.
- **Screen readers:** Semantic HTML elements, ARIA labels on icon-only buttons, status announcements via `aria-live` regions.
- **Reduced motion:** Respect `prefers-reduced-motion`. Disable spinner animations, use instant transitions instead of fade/slide.
