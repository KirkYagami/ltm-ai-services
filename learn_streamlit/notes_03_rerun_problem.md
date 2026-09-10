# Why Plain Variables Don't Persist

In a normal Python script, a variable retains its value for as long as the script is running. In Streamlit, this assumption breaks down, because the entire script reruns from the top on every interaction — including the line that initializes the variable.

## The mechanism

Consider a variable initialized with `count = 0` near the top of a script, followed later by logic that increments it when a button is pressed. On the surface this looks like a normal counter. In execution order, however:

1. The script starts. `count = 0` runs, setting the value to 0.
2. The button is pressed. Streamlit reruns the entire script.
3. `count = 0` runs again — before the increment logic is reached — resetting the value back to 0.
4. The increment then runs, taking the value from 0 to 1.

Every rerun repeats this same sequence. The variable is never allowed to carry a value forward from one rerun into the next, because its initialization statement is re-executed every time, immediately before the code that would change it.

## The underlying issue

This is not a bug in the script or in Streamlit — it is a direct consequence of the execution model. A plain Python variable exists only within the scope of a single execution of the script. Once that execution ends (i.e., once the rerun finishes and Streamlit is waiting for the next interaction), the variable and its value cease to exist. The next rerun starts with a completely fresh set of local variables.

## What is actually needed

To build any UI that depends on accumulated state — a counter, a running score, a list of items added over time, the current step in a multi-step form — the application needs a place to store values that exists *outside* the normal lifetime of a single script execution, and is tied to a specific user's session rather than to the script run itself. That mechanism is `st.session_state`.

