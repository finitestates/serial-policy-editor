# Choice pane geometry proof

This is the pre-implementation row plan for the live Choice view. Coordinates
are zero-based; rectangles are written as `(x, y, width, height)`. Every region
spans the terminal width. Zero-height feedback or candidate-label regions are
omitted on the compact layout.

## Sizing rules

- The context height is its wrapped line count, capped at
  `min(8, max(1, terminal_height // 4))`. A non-empty context keeps at least one
  row when space permits. Overflow scrolls inside that rectangle.
- The context/candidate divider is one row and has its own rectangle.
- Preview and feedback keep their current compact/expanded heights. The
  candidate table has a two-row minimum and receives all remaining flexible
  rows after the content-sized context and reserved controls.
- Minimum priority is command, candidate table, preview, hint, heading/context,
  feedback, and candidate label. This keeps the existing compact-screen
  controls in their current priority order.
- The examples use a short two-row context or a long context exceeding the
  cap. Expanded examples include a three-row preview and two-row feedback.
  The compact example uses a one-row preview and no feedback.

## 40x12, short context

| Region | Rectangle |
| --- | --- |
| Heading | `(0, 0, 40, 1)` |
| Context | `(0, 1, 40, 2)` |
| Divider | `(0, 3, 40, 1)` |
| Preview | `(0, 4, 40, 1)` |
| Feedback | `(0, 5, 40, 0)` |
| Candidate label | `(0, 5, 40, 0)` |
| Candidate table | `(0, 5, 40, 5)` |
| Command | `(0, 10, 40, 1)` |
| Hint | `(0, 11, 40, 1)` |

The five candidate-table rows are its header plus four candidate rows. No spare
rows are assigned to context.

## 80x24, long context

| Region | Rectangle |
| --- | --- |
| Heading | `(0, 0, 80, 1)` |
| Context | `(0, 1, 80, 6)` |
| Divider | `(0, 7, 80, 1)` |
| Preview | `(0, 8, 80, 3)` |
| Feedback | `(0, 11, 80, 2)` |
| Candidate label | `(0, 13, 80, 1)` |
| Candidate table | `(0, 14, 80, 7)` |
| Command | `(0, 21, 80, 1)` |
| Hint | `(0, 22, 80, 2)` |

Scrolling changes only the context's visible line window. The divider, preview,
feedback, candidate label/table, command, and hint keep these same rectangles.

## 120x40, long context

| Region | Rectangle |
| --- | --- |
| Heading | `(0, 0, 120, 1)` |
| Context | `(0, 1, 120, 8)` |
| Divider | `(0, 9, 120, 1)` |
| Preview | `(0, 10, 120, 3)` |
| Feedback | `(0, 13, 120, 2)` |
| Candidate label | `(0, 15, 120, 1)` |
| Candidate table | `(0, 16, 120, 21)` |
| Command | `(0, 37, 120, 1)` |
| Hint | `(0, 38, 120, 2)` |

The context scroll state still uses `(0, 1, 120, 8)`; only its top line changes.
All declared regions and the divider exactly cover each parent rectangle,
without overlaps or unassigned rows.
