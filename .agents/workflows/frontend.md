# Frontend workflow

Apply this workflow to new or changed UI, including fixes to an existing screen.

## Delegation

Spawn a dedicated frontend implementation subagent. It owns implementation, interaction testing, responsive verification, commit, push, and PR handoff. An exploration/research subagent may provide context, but never substitutes for this required implementation subagent; the primary agent must not perform the implementation itself. Before spawning, verify that the delegated environment can access the intended worktree and required repository tools. If that check fails, do not start the implementation run. If a delegated worker stalls, re-dispatch it or report the blocker; the primary may review and deliver, but must not edit implementation files unless the user explicitly waives delegation.
- Before spawning the implementation subagent, verify that its environment can access the worktree and required repository tools. If that preflight fails, fix the environment or report the blocker before implementation.
- If the subagent stalls, re-dispatch it or report the blocker. The primary agent may review and deliver, but must not edit implementation files unless the user explicitly waives delegation.

## Implementation

- Reproduce supplied designs closely and use meaningful demo data when real data is unavailable.
- Treat supplied HTML, screenshots, and mockups as immutable visual reference material. Preserve HTML in the task description verbatim, and translate its observable requirements into a checklist before coding: content, images, colors, typography, spacing, responsive layout, and interaction states.
- When the reference contains example content, the no-real-data/demo state must render that content faithfully. Do not silently replace required images, cards, rows, or labels with empty states, placeholders, or a different data shape.
- Implement all interactions represented by the reference or existing page.
- Do not leave clickable-looking buttons, links, tabs, date controls, menus, modal actions, filters, or exports as no-ops. Implement them, remove them, or deliberately disable them with an explanation.
- Make state-changing controls update an observable UI or data outcome, not only their visual styling.
- Convert the product brief into explicit visual acceptance criteria before coding. Treat named page models, navigation structures, content types, typography, responsive behavior, and interaction states as acceptance criteria.
- Brand inspiration must not override explicit product requirements from the brief or project contract.

## Acceptance

- Add regression coverage at the correct seam for each new or repaired interaction.
- Verify the rendered application interactively in a browser at desktop and mobile widths. A screenshot alone is insufficient, but visual review of screenshots or the live rendered page is required for reference-matching work; DOM structure, CSS class names, or a successful page load are not evidence of visual fidelity.
- Compare the rendered result against the supplied reference before committing. Check the reference checklist explicitly, including image loading, color and typography, all specified rows/cards, and desktop/mobile layout. Fix differences before asking the user for more design direction when the reference is already specific.
- Exercise every relevant control and assert its expected visible/data result; include open/close, navigation, filtering/date changes, editing/deleting, exports, and error/empty states when present.
- Record focused test results and identify pre-existing failures separately. Only call a command successful when it exited zero; a build or lint failure elsewhere in the repository must be reported as a blocker, not as a passing verification result.
- Before staging and handoff, inspect `git status --short`; do not commit generated screenshots, browser artifacts, temporary logs, or unrelated user files.
- Explicitly review readability, typography scale, alignment, page hierarchy, and whether every summary block communicates a meaningful product state—not only DOM presence and overflow metrics.

## Handoff

Stage only intended files, commit, push, and open or update the PR. Link its URL to the related work item when applicable.
