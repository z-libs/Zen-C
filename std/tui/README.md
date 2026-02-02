# Zen-C TUI Library

A terminal UI library for Zen-C . Build rich, interactive terminal applications with double-buffered rendering, constraint-based layouts, styled text, and a collection of ready-to-use widgets.

## Architecture

```
std/tui/
├── terminal.zc        Raw terminal control (alternate screen, raw mode, cursor)
├── style.zc           Color, Style structs and ANSI SGR generation
├── text.zc            Span, Line, Text structs for styled text
├── buffer.zc          Cell grid with double-buffering and diff rendering
├── layout.zc          Constraint-based rect splitting
├── event.zc           Keyboard event polling (non-blocking)
├── widget.zc          Frame struct and tui_run() app loop
└── widgets/
    ├── block.zc       Borders + title container
    ├── paragraph.zc   Multi-line styled text with wrapping
    ├── list.zc        Scrollable selectable list
    ├── table.zc       Column-aligned table with header
    ├── tabs.zc        Tab bar
    ├── gauge.zc       Progress bar
    ├── sparkline.zc   Inline sparkline chart
    ├── barchart.zc    Vertical bar chart
    └── input.zc       Single-line text input with cursor
```

## Quick Start

A minimal TUI application:

```zc
import "std/tui/style.zc"
import "std/tui/terminal.zc"
import "std/tui/buffer.zc"
import "std/tui/layout.zc"
import "std/tui/event.zc"
import "std/tui/widget.zc"
import "std/tui/widgets/block.zc"
import "std/tui/widgets/paragraph.zc"
import "std/tui/text.zc"

fn draw(f: Frame*) {
    let area = f.size();
    let text = Text::raw("Hello from Zen-C TUI!\nPress q to quit.");
    let para = Paragraph::new(text)
        .set_block(Block::bordered().set_title("Hello"))
        .set_wrap(true);
    para.render(area, f.buffer);
}

fn handle_event(evt: Event) -> int {
    if (evt.kind == EVT_KEY) {
        if (evt.key.code == KEY_CHAR && evt.key.ch == 'q') return 0;
    }
    return 1;
}

fn main() {
    let draw_fn = fn(f: Frame*) { draw(f); };
    let event_fn = fn(e: Event) -> int { return handle_event(e); };
    tui_run(draw_fn, event_fn);
}
```

Build and run:

```bash
./zc run examples/tui/dashboard.zc
```

## Core Concepts

### The Render Loop (`tui_run`)

`tui_run` manages the application lifecycle:

1. Enters raw mode and switches to the alternate screen
2. Each frame: clears back buffer, calls your draw function, diff-renders only changed cells
3. Polls stdin for keyboard events and calls your event handler
4. Returns 0 from the event handler to quit
5. Restores the terminal on exit

The two callback parameters must be passed as typed lambdas:

```zc
let draw_fn = fn(f: Frame*) { draw(f); };
let event_fn = fn(e: Event) -> int { return handle_event(e); };
tui_run(draw_fn, event_fn);
```

### Layout System

Layouts split a `Rect` into sub-areas using constraints. This is the same model as Ratatui.

```zc
let constraints: Constraint[3];
constraints[0] = Constraint::length(3);      // exactly 3 rows
constraints[1] = Constraint::min(5);         // at least 5 rows, expands to fill
constraints[2] = Constraint::length(1);      // exactly 1 row

let layout = Layout::vertical().set_constraints(constraints, 3);
let rects = layout.split(area);
// rects[0] = top bar,  rects[1] = main content,  rects[2] = status line
```

**Constraint types:**

| Constraint | Description |
|---|---|
| `Constraint::length(n)` | Exactly `n` cells |
| `Constraint::percentage(p)` | `p`% of total space |
| `Constraint::min(n)` | At least `n` cells, grows with remaining space |
| `Constraint::max(n)` | At most `n` cells |
| `Constraint::ratio(a, b)` | Fraction `a/b` of total space |

**Directions:** `Layout::vertical()` splits top-to-bottom, `Layout::horizontal()` splits left-to-right.

The returned `Rect*` array is `malloc`'d -- call `free(rects)` when done.

### Styling

Build styles with colors and modifiers:

```zc
let s1 = Style::fg(Color::cyan());                           // foreground color
let s2 = Style::fg(Color::white()).set_bg(Color::blue());    // fg + bg
let s3 = Style::bold().set_fg(Color::yellow());              // bold + color
let s4 = Style::new().add_modifier(MOD_ITALIC | MOD_UNDERLINE);

// RGB colors
let custom = Style::fg(Color::rgb(255, 128, 0));
```

**Available colors:** `Color::black()`, `red()`, `green()`, `yellow()`, `blue()`, `magenta()`, `cyan()`, `white()`, `default_color()`, `rgb(r, g, b)`.

**Modifier flags:** `MOD_BOLD`, `MOD_DIM`, `MOD_ITALIC`, `MOD_UNDERLINE`, `MOD_BLINK`, `MOD_REVERSE`, `MOD_HIDDEN`, `MOD_STRIKETHROUGH`. Combine with `|`.

### Styled Text

Three levels of text abstraction:

```zc
// Span: a single styled string
let span = Span::styled("hello", Style::fg(Color::red()));

// Line: a sequence of spans on one line
let spans: Span[2];
spans[0] = Span::styled("Status: ", Style::bold());
spans[1] = Span::styled("OK", Style::fg(Color::green()));
let line = Line::from_spans(spans, 2);

// Text: multiple lines (splits on \n)
let text = Text::raw("Line one\nLine two\nLine three");
```

### Event Handling

Events are polled non-blocking with ANSI escape sequence parsing:

```zc
fn handle_event(evt: Event) -> int {
    if (evt.kind == EVT_KEY) {
        // Regular character
        if (evt.key.code == KEY_CHAR && evt.key.ch == 'q') return 0;

        // Arrow keys
        if (evt.key.code == KEY_UP)   { /* ... */ }
        if (evt.key.code == KEY_DOWN) { /* ... */ }

        // Ctrl+C
        if (evt.key.code == KEY_CHAR && evt.key.ch == 'c' && evt.key.modifier == KMOD_CTRL) {
            return 0;
        }

        // Tab, Enter, Escape, Backspace
        if (evt.key.code == KEY_TAB)       { /* ... */ }
        if (evt.key.code == KEY_ENTER)     { /* ... */ }
        if (evt.key.code == KEY_ESC)       { /* ... */ }
        if (evt.key.code == KEY_BACKSPACE) { /* ... */ }
    }
    return 1; // keep running
}
```

**Key codes:** `KEY_CHAR`, `KEY_ENTER`, `KEY_ESC`, `KEY_TAB`, `KEY_BACKSPACE`, `KEY_UP`, `KEY_DOWN`, `KEY_LEFT`, `KEY_RIGHT`, `KEY_DELETE`, `KEY_HOME`, `KEY_END`, `KEY_PAGEUP`, `KEY_PAGEDOWN`, `KEY_F1`..`KEY_F12`.

**Modifiers:** `KMOD_NONE`, `KMOD_CTRL`, `KMOD_ALT`, `KMOD_SHIFT`.

## Widgets

### Block

Container widget that draws borders and an optional title. Used by most other widgets.

```zc
let block = Block::bordered()                        // all four borders
    .set_title("My Panel")
    .set_title_style(Style::bold())
    .set_border_style(Style::fg(Color::cyan()))
    .set_border_type(BORDER_ROUNDED);                // ╭╮╰╯ corners
block.render(area, f.buffer);

// Get inner area (area minus borders) for child widgets
let inner = block.inner(area);
```

**Border types:** `BORDER_PLAIN` (default), `BORDER_ROUNDED`, `BORDER_DOUBLE`, `BORDER_THICK`.

**Border sides:** `BORDER_ALL`, `BORDER_TOP`, `BORDER_BOTTOM`, `BORDER_LEFT`, `BORDER_RIGHT`. Combine with `|`.

### Paragraph

Multi-line styled text with optional word wrapping and scrolling.

```zc
let text = Text::raw("Long text content here...");
let para = Paragraph::new(text)
    .set_block(Block::bordered().set_title("Content"))
    .set_wrap(true)
    .set_scroll(0)                    // scroll offset in lines
    .set_style(Style::fg(Color::white()));
para.render(area, f.buffer);
```

### List

Scrollable, selectable list. Uses `ListState` to track selection.

```zc
let items: ListItem[3];
items[0] = ListItem::new("First item");
items[1] = ListItem::styled("Warning item", Style::fg(Color::yellow()));
items[2] = ListItem::new("Third item");

let list = List::new(items, 3)
    .set_block(Block::bordered().set_title("Items"))
    .set_highlight_style(Style::fg(Color::black()).set_bg(Color::cyan()))
    .set_highlight_symbol("> ");

let state = ListState::new();
list.render_stateful(area, f.buffer, &state);

// In event handler:
state.next(3);      // select next (wraps)
state.previous(3);  // select previous (wraps)
```

### Table

Column-aligned table with a header row and selectable rows.

```zc
let header_cells: char*[3];
header_cells[0] = "Name"; header_cells[1] = "Size"; header_cells[2] = "Type";
let header = Row::new(header_cells, 3);

let r0: char*[3]; r0[0] = "main.zc"; r0[1] = "4.2K"; r0[2] = "file";
let r1: char*[3]; r1[0] = "src";     r1[1] = "128B"; r1[2] = "dir";
let rows: Row[2];
rows[0] = Row::new(r0, 3);
rows[1] = Row::new(r1, 3);

let widths: int[3];
widths[0] = 20; widths[1] = 10; widths[2] = 8;

let table = Table::new(rows, 2)
    .set_header(header)
    .set_widths(widths, 3)
    .set_block(Block::bordered().set_title("Files"))
    .set_highlight_style(Style::fg(Color::black()).set_bg(Color::green()));

let state = TableState::new();
table.render_stateful(area, f.buffer, &state);
```

**Important:** Declare all `char*[]` cell arrays at function scope (not inside loops) so `Row` pointers remain valid through rendering.

### Tabs

Horizontal tab bar for switching views.

```zc
let titles: char*[3];
titles[0] = "Tab 1"; titles[1] = "Tab 2"; titles[2] = "Tab 3";

let tabs = Tabs::new(titles, 3)
    .set_selected(current_tab)
    .set_block(Block::bordered().set_title("Navigation"))
    .set_highlight_style(Style::fg(Color::yellow()).add_modifier(MOD_BOLD))
    .set_divider(" | ");
tabs.render(area, f.buffer);
```

### Gauge

Progress bar with a centered label.

```zc
let gauge = Gauge::new()
    .set_percent(75)                   // or .set_ratio(0.75)
    .set_label("75% complete")         // NULL = auto "75%"
    .set_block(Block::bordered().set_title("Progress"))
    .set_gauge_style(Style::fg(Color::green()));
gauge.render(area, f.buffer);
```

### Sparkline

Inline chart using Unicode block characters (&#x2581;&#x2582;&#x2583;&#x2584;&#x2585;&#x2586;&#x2587;&#x2588;). Shows the most recent data points that fit the width.

```zc
let data: int[30];
// ... fill with values ...

let spark = Sparkline::new(data, 30)
    .set_max(100)                      // clamp scale
    .set_block(Block::bordered().set_title("CPU"))
    .set_style(Style::fg(Color::cyan()));
spark.render(area, f.buffer);
```

### BarChart

Vertical bar chart with labels and values.

```zc
let bars: BarGroup[3];
bars[0] = BarGroup { label: "/",     value: 65 };
bars[1] = BarGroup { label: "/home", value: 45 };
bars[2] = BarGroup { label: "/tmp",  value: 20 };

let chart = BarChart::new(bars, 3)
    .set_bar_width(5)
    .set_bar_gap(2)
    .set_bar_style(Style::fg(Color::yellow()))
    .set_block(Block::bordered().set_title("Disk Usage"));
chart.render(area, f.buffer);
```

### Input

Single-line text input with cursor movement, insertion, and deletion.

```zc
let input = Input::new();
input.block = Block::bordered().set_title("Search");
input.has_block = true;
input.render(area, f.buffer);

// In event handler:
if (evt.key.code == KEY_ENTER) {
    let value = input.value();   // get current text
}
input.handle_key(evt.key);       // handles typing, backspace, arrows, home/end
input.clear();                   // reset
input.set_text("/some/path");    // set programmatically
```

## Examples

### Dashboard (`examples/tui/dashboard.zc`)

A multi-tab dashboard demonstrating most widgets:

- **Overview tab:** CPU sparkline, memory gauge, disk bar chart, info paragraph
- **Processes tab:** Table with selectable rows
- **Logs tab:** Styled list with colored log levels

Controls: `q` quit, `Tab` switch tabs, `Up/Down` navigate lists and tables.

```
┌─Dashboard──────────────────────────────────────────────┐
│ Overview | Processes | Logs                             │
├────────────────────────────┬───────────────────────────┤
│ ╭─CPU Usage──────────────╮ │ ╭─Info────────────────────╮│
│ │ ▂▃▅▇▆▄▃▅▇█▇▅▃▂▁      │ │ │ System Dashboard       ││
│ ╰────────────────────────╯ │ │                         ││
│ ┌─Memory─────────────────┐ │ │ This is a TUI demo...  ││
│ │████████████░░░░░ 68%   │ │ ╰─────────────────────────╯│
│ └────────────────────────┘ │                             │
│ ┌─Disk Usage %───────────┐ │                             │
│ │ ██  ██  ██  ██         │ │                             │
│ │ 65  45  20  80         │ │                             │
│ │ /  /home /tmp /var     │ │                             │
│ └────────────────────────┘ │                             │
├────────────────────────────┴───────────────────────────┤
│ q: Quit | Tab: Switch | Up/Down: Navigate              │
└────────────────────────────────────────────────────────┘
```

### File Browser (`examples/tui/filebrowser.zc`)

A two-pane file browser with live preview:

- **Left pane:** Directory listing (directories highlighted in blue)
- **Right pane:** File content preview
- **Bottom:** Path input bar (press `/` to activate)

Controls: `q` quit, `Up/Down` navigate, `Enter` open directory, `/` go-to path.

```
┌─Path──────────────────────────────────────────────────┐
│ /home/user/projects                                    │
├───────────────────┬────────────────────────────────────┤
│ ╭─Files──────────╮│ ╭─Preview: main.zc────────────────╮│
│ │ > ..           ││ │ fn main() {                     ││
│ │   src/         ││ │     println "Hello!";           ││
│ │   README.md    ││ │ }                               ││
│ │   main.zc      ││ │                                 ││
│ ╰────────────────╯│ ╰─────────────────────────────────╯│
├───────────────────┴────────────────────────────────────┤
│ ┌─Go to path (press /)───────────────────────────────┐ │
│ │                                                     │ │
│ └─────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────┘
```

## Design Notes

- **No external dependencies.** Pure POSIX terminal I/O via `raw {}` blocks wrapping `termios`, `ioctl`, and `select()`.
- **Double buffering.** Each frame renders into a back buffer, then only changed cells are written to the terminal. This eliminates flicker.
- **UTF-8 box drawing.** Borders use Unicode characters (`─│┌┐└┘╭╮╰╯═║` etc.) encoded as raw UTF-8 byte sequences.
- **Copy semantics.** Core structs (`Rect`, `Style`, `Color`, `Constraint`, `Event`, `Block`, etc.) implement `Copy` so they can be passed by value freely.
- **Closure callbacks.** `tui_run` takes `fn()` closure parameters. Pass named functions via typed lambda wrappers: `fn(f: Frame*) { draw(f); }`.
- **Caller-owned layout arrays.** `Layout::split()` returns a `malloc`'d `Rect*` array. The caller must `free()` it after use.
