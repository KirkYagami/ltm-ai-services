# Putting the Concepts Together

A realistic Streamlit application combines all of the preceding concepts simultaneously, because they address different layers of the same underlying execution model.

## How the concepts interact

- **Widgets** (`st.button`, `st.radio`, `st.slider`, etc.) are the source of interactions. Every time a widget's value changes, Streamlit triggers a rerun of the whole script — this is the execution model described earlier.
- **The rerun** means that any logic the app needs to run in response to that interaction is expressed as ordinary Python control flow (`if` statements checking widget values), not as a separate callback function.
- **`st.session_state`** is what allows the app to track progress across those reruns — for example, which step of a multi-step process the user is on, an accumulated score, or a running total — none of which could be represented with plain variables, since those are wiped out on every rerun.
- **Caching** (`st.cache_data` / `st.cache_resource`) ensures that any expensive setup work the app depends on — loading a dataset, initializing questions, connecting to a resource — happens once and is then reused across all the reruns that follow, rather than repeating on every interaction.

## The general pattern

Most non-trivial Streamlit apps follow the same structure:

1. Load or prepare any data/resources needed, wrapped in `@st.cache_data` or `@st.cache_resource` so this only happens once.
2. Initialize any values that need to persist in `st.session_state`, guarded by `if key not in st.session_state`.
3. Render widgets and layout based on the current state.
4. Update `st.session_state` in response to widget interactions (button presses, form submissions).
5. Let the rerun mechanism redraw the UI reflecting the new state — no manual DOM updates or explicit "refresh" calls are needed, since the next rerun naturally reflects whatever is currently in `session_state`.

Understanding this structure is sufficient to build most interactive Streamlit applications: the rest is largely a matter of which widgets and layout elements are used to present the data.

