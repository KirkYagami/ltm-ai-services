# Caching

Because Streamlit reruns the entire script on every interaction, any expensive operation placed directly in the script body — loading a large file, querying a database, calling an external API, training or loading a model — would normally re-execute on every single rerun, even when the inputs to that operation have not changed. Caching exists to prevent this.

## st.cache_data

`st.cache_data` is intended for functions that return data: DataFrames, arrays, lists, dictionaries, strings, or other serializable values. When a cached function is called, Streamlit checks whether it has already computed a result for that exact combination of function and arguments. If so, it returns a stored copy of the previous result instead of re-running the function body. If the function is called again with different arguments, the cache key changes, and the function executes fresh for that new input.

Because `st.cache_data` returns a *copy* of the cached object, mutating the returned value in one part of the app does not affect the cached original or any other place that retrieves it — this makes it safe for data that different parts of the script might modify independently.

## st.cache_resource

`st.cache_resource` is intended for objects that should be shared as a single instance rather than copied — most commonly database connections, ML models loaded into memory, or other resources that are expensive to create and are not meant to be duplicated. Unlike `st.cache_data`, it does not return a copy; every call retrieves the same underlying object. This matters for things like a database connection, where creating a new connection object per rerun (or per user) would be wasteful or incorrect.

## Cache keys

Both decorators cache based on the function's arguments (and its code). Calling the same cached function with the same arguments returns the cached result; calling it with different arguments triggers a fresh computation, which is then cached separately. This is what allows an app to, for example, cache results per selected date range or per uploaded file, rather than caching a single fixed result regardless of input.

## Relationship to the execution model

Caching does not change the fact that the whole script reruns on every interaction — the surrounding code (widget definitions, layout, conditionals) still executes every time. What caching avoids is redoing the *expensive work inside a specific function* when its inputs are unchanged, which is what makes a script that reruns constantly still perform acceptably.

