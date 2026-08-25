# buddy — brand overview deck brief

Paste-ready brief for Claude Design. Build a slide deck that introduces the buddy brand: who it is, the design philosophy, the visual system (type, color, texture), and what it can do today and next. Audience: people seeing buddy for the first time (collaborators, early users, potential teammates). Tone of the deck itself should FOLLOW the brand rules below: the deck is the first proof of the design system.

---

## 1. The product in one breath

**buddy** (always lowercase) is a sandboxed AI agent for the long boring tasks. It runs your browser and computer work in the background: it drafts the reply, files the form, triages the board, then waits for your ok. Nothing leaves the desk without approval.

- Product name: **buddy**, never capitalized, never "Buddy".
- The app surface is called **buddy's desk**.
- Logo mark: a single lowercase **'b'** glyph (the old "ark"/"a" mark is retired; "ark"/ARKOS survives only as internal infra naming and should not appear in the deck).
- Tagline options (use verbatim, lowercase): "a sandboxed ai agent for the long boring tasks" · "your computer work, handled in the background" · "quiet, sandboxed, and always waiting on your ok".

## 2. Brand philosophy

- **Not a chatbot.** buddy integrates into existing workflows (mail, boards, calendar, browser). Conversation is how you spec the work; the handoff is the product. "Talk to spec, hand off to get work done."
- **Talk is free, work is the unit.** Chatting with buddy is unmetered; a "run" of real work is the unit that counts. Generosity toward unattended work, never scarcity on conversation.
- **Consent is the ritual.** Every outbound action (an email, a closed issue, a submitted form) passes through an approval card. The approve/decline moment is the signature interaction of the product.
- **Quiet by default.** buddy watches the inbox, the board, the calendar, and only surfaces what needs you.
- **It learns how you work.** Corrections become memory; you stop repeating yourself.

## 3. Design tenets (the visual philosophy)

The house style in three phrases, verbatim from the design system: **refined monospace · calm editorial · near-monochrome**.

1. **One typeface, everywhere.** Everything is set in monospace. Hierarchy comes from size, weight, spacing, and ink strength, not from typeface changes.
2. **Warm paper, not white screens.** Surfaces are warm off-whites (dark mode: warm near-blacks). The feel is a well-kept desk: paper, hairline rules, soft shadows.
3. **Near-monochrome with one restrained accent.** Ink tones do almost all the work. A single muted forest green appears sparingly for the live/affirmative moments (approve, connected, active). Status colors are desaturated and share lightness so nothing screams.
4. **Small, precise, lowercase.** Labels are small caps-free lowercase mono with slight letterspacing. Copy is plainspoken, lowercase, no hype words, no exclamation marks.
5. **Calm motion.** Subtle transitions only, standard ease (cubic-bezier(.4,0,.2,1)). Nothing bounces.
6. **Restraint equals trust.** The visual quietness mirrors the product promise: it acts only with your ok. Dense-but-calm layouts, generous margins, thin 1px rules instead of boxes and fills.

## 4. Type

- Primary (and only) family: **JetBrains Mono**, fallback stack: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace.
- Usage pattern from the app: body ~13-14px, secondary text 11-12px, micro-labels 10-11px with letter-spacing 0.02-0.06em, line-height 1.6-1.75. For slides, scale up proportionally but keep the mono texture and the small-label habit.
- Headlines: same mono family, larger and heavier; lowercase.

## 5. Color tokens

Light theme (warm paper):

- ink `#1c1b18` · ink-soft `#54524b` · ink-mute `#97948a` · ink-faint `#c4c1b6`
- paper `#f7f6f1` · surface `#fdfcf9` · surface-2 `#f1efe8`
- line `#e6e3da` · line-soft `#efece4`
- accent (muted forest green) `oklch(0.55 0.075 150)` ≈ `#4e7d5b`; accent-soft = accent at 12% alpha; accent-line = accent at 35% alpha
- status: live `oklch(0.58 0.08 150)` (green) · work `oklch(0.66 0.085 75)` (muted amber) · stop `oklch(0.58 0.10 28)` (muted brick)

Dark theme (warm near-black):

- ink `#ecebe6` · ink-soft `#b6b3aa` · ink-mute `#797668` · ink-faint `#46443d`
- paper `#121210` · surface `#1a1815` · surface-2 `#201f1b`
- line `#2a2823` · line-soft `#221f1b`
- accent `oklch(0.74 0.105 150)` ≈ `#7fb98f`; same soft/line alpha treatment
- status: live `oklch(0.74 0.105 150)` · work `oklch(0.78 0.10 78)` · stop `oklch(0.70 0.12 28)`

Shape and depth: corner radius 7px; shadows are barely-there paper lifts (e.g. `0 1px 2px rgba(28,27,24,.03), 0 12px 32px -20px rgba(28,27,24,.18)`), never hard drop shadows.

For the deck: default to the light warm-paper theme; a dark closing slide is welcome.

## 6. Capabilities — current

- **Browses and computes.** Reads pages, fills forms, and runs real multi-step tasks on its own sandboxed computer and browser. Long tasks run unattended in the background.
- **Drafts, then waits for your ok.** Writes the reply, the plan, the summary; a pending-approval card shows exactly what will happen (e.g. "reply to priya re: q3 roadmap review" with the drafted text) with decline / approve.
- **Plans visibly.** Proposes a plan, keeps a live todo of steps, reports results; you can inspect any run.
- **Connected services** through one gateway: Gmail, GitHub, Linear, Microsoft Outlook Mail, Notion, Google Calendar. Per-user consent from a settings panel; each connection is granted, not assumed.
- **Memory.** Saves, searches, and updates what it learns about how you like things done, and applies it to the next task.
- **Projects and sessions.** Work is organized into projects with browsable sessions and files.

## 7. Capabilities — future (roadmap, in rough order)

- **Live attention stream.** Approvals and things-that-need-you surface in real time on a per-user channel; the desk becomes a live feed of pending decisions.
- **Watching as a first-class mode.** Standing watches on inbox, board, and calendar that can wake an idle buddy when something needs handling (event/webhook wake-ups).
- **Public sign-up.** Google OAuth onboarding, on-brand auth emails, the full ark-to-buddy rebrand shipped end to end.
- **Runs-based plans.** Pricing built on completed runs of real work; conversation stays free, unattended work gets more generous, never metered chat.
- **Scale-out.** Modernized frontend, hardened hosting, multi-worker serving for many concurrent buddies.

## 8. Suggested slide flow (adapt freely)

1. Title: 'b' mark, "buddy", tagline.
2. The problem: the long boring computer tasks.
3. What buddy is: sandboxed agent, drafts then waits for your ok.
4. The desk: approvals + active tasks + watching (recreate the landing's desk mock in-system).
5. Philosophy: not a chatbot; talk is free, work is the unit; consent is the ritual.
6. Design tenets: refined monospace · calm editorial · near-monochrome.
7. Type and color: token board (swatches + JetBrains Mono specimen), light and dark.
8. Capabilities today.
9. What's next (roadmap).
10. Close: dark slide, 'b' mark, "buddy. a sandboxed ai agent."

Rules for the deck itself: all lowercase where the brand voice calls for it, JetBrains Mono throughout, warm paper backgrounds, hairline rules, the green accent only where something is live or affirmed. No stock imagery, no gradients, no emoji.
