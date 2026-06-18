---
name: streamlit-professional-ui
description: Use this skill whenever building, editing, or styling the Streamlit UI for the restaurant rating/review prediction app. Applies to layout, CSS, components, color, typography, and overall visual design decisions. Triggers on requests like "buatkan tampilan", "perbaiki UI", "style this page", "buat dashboard", or any work inside app.py / pages/*.py that touches st.markdown CSS, st.columns, st.container, or component layout. Goal: produce a UI that looks like a real internal tool at a data/NLP company — not an AI hackathon demo.
---

# Streamlit Professional UI — Restaurant Review Rating Predictor

## Context

This app takes a restaurant review (free text) as input and predicts a rating
(e.g. 1–5 stars / positive-negative-neutral / numeric score) using an ML/NLP
model. The primary user is someone evaluating reviews — a business analyst,
restaurant owner, or QA reviewer checking model output. Design for that
person, not for a design portfolio.

**Default visual posture: quiet, dense-information, professional. The kind of
internal tool an analytics or data team ships and actually keeps using for a
year.**

## Hard rules — things to actively avoid

These are banned because they are the most common visual tells of
"AI-generated demo," not because they're inherently wrong:

- ❌ Glassmorphism / `backdrop-filter: blur()` / translucent panels
- ❌ Transparent or "glass" buttons
- ❌ Loud or rainbow gradients (especially on buttons, headers, or backgrounds)
- ❌ Neon glow, drop-shadows with color, glowing borders
- ❌ Heavy box-shadows (`shadow-xl`, multi-layer shadows, anything beyond a
  faint 1px–2px lift)
- ❌ Distracting animation: bouncing, pulsing, confetti, fade-in cascades on
  every element, animated gradients
- ❌ High-saturation or "futuristic" palettes (electric purple/blue, neon
  green, cyberpunk schemes)
- ❌ Emoji used as primary icons in headers/buttons (occasional emoji in body
  copy is fine if it matches a human tone, not as UI iconography)
- ❌ Centered hero sections with giant bold claims ("✨ Predict Your
  Restaurant's Success Instantly! 🚀")
- ❌ Default `st.balloons()` / `st.snow()` for normal predictions

If you catch yourself about to add any of the above "to make it look more
polished," that instinct is wrong for this brief — remove it instead.

## Design principles to follow

Priority order: **Clarity → Readability → Professionalism → Consistency → Delight.**

1. **Clean, structured layout.** Use `st.columns`, `st.container`, and
   `st.tabs` deliberately to group related things (input → result →
   explanation), not just to fill space.
2. **Generous whitespace.** Don't cram. Let sections breathe with consistent
   vertical rhythm (use consistent `margin`/`padding` multiples — e.g. 8px
   scale: 8/16/24/32).
3. **Clear typography hierarchy.** One font family (system font stack or a
   single clean sans like Inter/IBM Plex Sans/Source Sans). 3–4 weight/size
   steps max: page title, section label, body text, caption/meta text.
   Don't introduce a second display font for "flair."
4. **Neutral, restrained color.** A near-white or very light gray background
   (`#FAFAFA`–`#F7F7F8`), dark neutral text (`#1A1A1A`–`#222`), one muted
   primary accent color used sparingly (buttons, active states, key
   numbers), and semantic colors for rating feedback (see below). No more
   than 1 accent + semantic colors total.
5. **Simple cards with hairline borders**, not shadows-as-default. Use
   `border: 1px solid #E5E5E5` (or similar light neutral) with `border-radius`
   in the 6–10px range. Avoid heavy rounding (20px+ "bubble" look).
6. **Visual hierarchy through size/weight/spacing, not color noise.** The
   predicted rating should be the most visually prominent element on the
   result view — bigger/bolder, not brighter.
7. **Natural microcopy.** Write labels and messages the way a real product
   would: "Enter a review" not "✍️ Drop Your Review Here!", "Prediction
   failed — check input and try again" not "Oops! 😅 Something went wrong!"

## Color palette (default — adjust to brand if user specifies one)

Use this as a sensible default token set unless the user gives their own brand colors:

| Token | Hex | Use |
|---|---|---|
| `bg` | `#FAFAFA` | App background |
| `surface` | `#FFFFFF` | Cards, containers |
| `border` | `#E5E5E5` | Card borders, dividers |
| `text-primary` | `#1A1A1A` | Headings, key text |
| `text-secondary` | `#5F6368` | Body/secondary text |
| `text-muted` | `#9AA0A6` | Captions, placeholders |
| `accent` | `#2F5D50` or `#3B5BDB` | Primary buttons, links, active tab (pick ONE, muted not neon) |
| `success` | `#2E7D32` | Positive rating / good review |
| `warning` | `#B8860B` | Neutral / mixed rating |
| `danger` | `#C0392B` | Negative rating / low score |

Use `success`/`warning`/`danger` only for the semantic meaning of the
*prediction result* (e.g. rating tier), not decoratively elsewhere.

## Typography

```css
font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Inter", Roboto, sans-serif;
```

| Element | Size | Weight |
|---|---|---|
| Page title | 24–28px | 600 |
| Section header | 16–18px | 600 |
| Body text | 14–15px | 400 |
| Caption / meta | 12–13px | 400, `text-muted` color |
| Predicted rating (hero number) | 36–48px | 700 |

## Layout pattern for this specific app

A predictable, sensible structure — don't reinvent this without reason:

1. **Header row** — app name/title (left), optional short subtitle/description
   (small, muted, one line). No giant hero banner.
2. **Input section** (card or plain container)
   - `st.text_area` for the review text, clear label ("Restaurant review"),
     placeholder text that's a realistic example, not lorem-ipsum-style filler.
   - Optional: character/word count as small muted caption below the box.
   - Primary action button: solid fill using the one accent color, clear
     label ("Predict rating" / "Analyze review"), not full-width unless the
     layout calls for it.
3. **Result section** — only appears after prediction, sits directly below
   input (no separate page jump):
   - Predicted rating/stars as the visual focal point.
   - Confidence/probability shown as a clean horizontal bar chart or simple
     progress indicator — not a flashy gauge/speedometer unless that's
     genuinely the clearest way to show it.
   - Optionally: short model explanation (e.g. top contributing
     words/phrases) in a secondary card, de-emphasized vs. the main result.
4. **Footer/meta** (optional) — model name/version, small muted text, bottom
   of page. No marketing copy.

ASCII reference:

```
┌──────────────────────────────────────────────┐
│ Restaurant Review Rating Predictor            │
│ Predict a star rating from review text         │
├──────────────────────────────────────────────┤
│  Restaurant review                             │
│  ┌────────────────────────────────────────┐   │
│  │ [text area]                             │   │
│  └────────────────────────────────────────┘   │
│  12 words                      [ Predict ▸ ]  │
├──────────────────────────────────────────────┤
│  Predicted rating                              │
│       ★★★★☆  4.2 / 5                          │
│  Confidence ████████░░ 82%                     │
│                                                 │
│  Top contributing phrases                      │
│  "great service"  "a bit slow"  "would return"  │
└──────────────────────────────────────────────┘
```

## Streamlit-specific implementation notes

- Set page config early and minimally:
  ```python
  st.set_page_config(page_title="Restaurant Rating Predictor", layout="centered")
  ```
  Use `layout="centered"` for a focused single-task tool unless the user
  wants a wider analytics-style dashboard (`"wide"`), in which case constrain
  content width via columns rather than letting text stretch edge-to-edge.

- Hide Streamlit's default chrome that reads as "demo app":
  ```python
  st.markdown("""
      <style>
      #MainMenu {visibility: hidden;}
      footer {visibility: hidden;}
      header {visibility: hidden;}
      </style>
  """, unsafe_allow_html=True)
  ```

- Custom CSS should be **subtle deltas** from Streamlit defaults, not a full
  visual overhaul fighting the framework. Target: card borders, button
  color/radius, font import, spacing tweaks. Example starting block:

  ```python
  st.markdown("""
      <style>
      html, body, [class*="css"] {
          font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
      }
      .stButton button {
          background-color: #2F5D50;
          color: #FFFFFF;
          border-radius: 6px;
          border: none;
          padding: 0.5rem 1.25rem;
          font-weight: 600;
      }
      .stButton button:hover {
          background-color: #264A40;
      }
      div[data-testid="stTextArea"] textarea {
          border-radius: 6px;
          border: 1px solid #E5E5E5;
      }
      .result-card {
          background-color: #FFFFFF;
          border: 1px solid #E5E5E5;
          border-radius: 8px;
          padding: 1.5rem;
      }
      </style>
  """, unsafe_allow_html=True)
  ```

- Use `st.container(border=True)` (modern Streamlit) for simple card
  grouping before reaching for custom HTML/CSS — it already matches the
  "simple card, subtle border" requirement natively.

- For the rating display, prefer plain text/markdown with large font size
  over custom SVG gauges or animated meters, unless explicitly asked for a
  richer visualization. A clean `st.metric()` is often enough:
  ```python
  st.metric(label="Predicted Rating", value="4.2 / 5", delta=None)
  ```

- For confidence/probability bars, `st.progress()` or a simple horizontal
  `st.bar_chart` is preferable to a custom animated gauge.

- Avoid `st.balloons()`, `st.snow()`, `st.toast()` with emoji spam for normal
  prediction results. A toast is fine for things like "Model loaded" if
  understated.

## Self-check before finishing any UI pass

Before presenting styling work, verify:

- [ ] No glassmorphism, glow, or heavy shadow anywhere
- [ ] No more than one accent color outside semantic rating colors
- [ ] All copy reads like a real product, not a demo/pitch
- [ ] Predicted rating is the clearest visual focal point on the result view
- [ ] Spacing is consistent (not cramped, not randomly large)
- [ ] Page would not look out of place as an internal tool at a real company