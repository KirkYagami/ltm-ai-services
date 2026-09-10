# The Execution Model

Streamlit does not use event-driven callbacks the way traditional GUI or web frameworks do. There is no equivalent of "attach a listener to this button that runs only this function when clicked." Instead, Streamlit follows a single rule:

> Whenever the user interacts with any widget, or the page is loaded/reloaded, Streamlit re-executes the entire Python script from top to bottom.

This is often called the "rerun model."

## Why Streamlit works this way

Streamlit's design goal is to let a data script — which normally runs once, start to finish, and produces some output — become interactive without the developer having to restructure it into callback functions. To achieve this, Streamlit treats the whole script as the single source of truth for what the page should look like at any given moment. Every time something changes (a widget value, a button press, a page load), Streamlit simply re-runs that source of truth and re-renders the result.

## Consequences of this model

1. **Nothing is "wired up."** A button doesn't have an associated function that runs in isolation. Pressing it just causes the file to run again, and the `if st.button(...)` line happens to evaluate to `True` on that particular run.
2. **All code executes on every rerun**, not just the code near the widget that changed. Imports, computations, data loading, and every other statement in the file run again from the top, regardless of which widget triggered the rerun.
3. **State does not persist automatically.** Because the script restarts from the top each time, any plain Python variable defined in the script is recreated from scratch on every rerun. Anything the script needs to "remember" between reruns must be stored somewhere that survives outside of normal variable scope — this is the role `st.session_state` plays.
4. **Expensive operations repeat unnecessarily** unless explicitly told not to. Loading a large dataset or calling an external API inside the script body would normally re-execute on every single interaction, even if the inputs to that operation haven't changed. This is the problem that caching (`st.cache_data`, `st.cache_resource`) solves.

The rerun model is the single most important concept in Streamlit, because both `session_state` and caching exist specifically as mechanisms to manage the consequences of the entire script re-running on every interaction.

