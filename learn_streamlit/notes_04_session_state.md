# st.session_state

`st.session_state` is Streamlit's mechanism for storing values that persist across reruns of a script, scoped to a single user session.

## Definition

A "session" in this context refers to one open browser tab connected to the running Streamlit app. Each session gets its own independent `st.session_state` object, which behaves like a Python dictionary (it also supports attribute-style access, e.g. `st.session_state.count` is equivalent to `st.session_state["count"]`). Unlike a normal variable declared inside the script, the contents of `st.session_state` are not reset when the script reruns — they are held in memory by Streamlit's server process for the lifetime of that session, independent of any single script execution.

## Why it is necessary

As established by the execution model, every interaction reruns the entire script, and any plain variable is recreated from scratch on each run. `st.session_state` exists specifically to give the script a place to read and write values that fall outside that lifecycle. Any data that needs to accumulate or change over multiple interactions — form progress, counters, toggles, accumulated selections, results of previous computations the user should keep seeing — belongs in `session_state`.

## Initialization pattern

Because the script reruns from the top every time, code must avoid unconditionally resetting a value that is meant to persist. The standard pattern is:

```python
if "count" not in st.session_state:
    st.session_state.count = 0
```

This sets a default value only the first time the key does not yet exist (i.e., on the very first run of a new session). On every subsequent rerun, the condition is false, so the existing value is left untouched, and later code can read or modify it safely.

## Scope

`st.session_state` is per-session, not global. Two different users — or the same user in two different browser tabs — each get their own independent `session_state`. Nothing stored there is shared between sessions. This distinguishes it from server-wide storage such as a database or `st.cache_resource`, both of which can be shared across all users of the app.

