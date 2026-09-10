# Why Streamlit

Streamlit is a Python framework for building web applications without writing HTML, CSS, or JavaScript. A Streamlit application is a regular Python script. Each function call in that script — `st.title()`, `st.text_input()`, `st.button()`, and so on — corresponds directly to a UI element rendered in the browser.

## The traditional web stack vs. Streamlit

Building a web page conventionally requires at least three separate concerns:

- **HTML** for structure (what elements exist on the page)
- **CSS** for presentation (how those elements look)
- **JavaScript** for behavior (how the page responds to user actions)

A typical web app also needs a backend to serve HTML, handle form submissions, and return updated data — usually via a framework like Flask or Django, plus a templating engine, plus routing.

Streamlit collapses all of this into one layer. The Python script itself is simultaneously the backend logic and the description of the UI. There is no separate template file, no routing configuration, and no client-side JavaScript to write. Streamlit's internal engine (a Tornado web server communicating with the browser over WebSockets) handles the translation from Python objects to rendered HTML/CSS/JS automatically.

## What this means in practice

Every `st.*` function is a declarative statement: "put this element on the page, in this position, with this content." The developer does not manipulate the DOM, does not write event handlers, and does not manage HTTP requests manually. Layout is expressed through Python control flow (if statements, loops, columns) rather than markup.

This is why Streamlit is widely used for data science and machine learning tooling: it lets someone who knows Python, but not frontend development, turn a script, model, or analysis into a usable interactive application.

## Trade-off

The simplicity comes from Streamlit taking over control of how the page is built and updated. This leads directly to Streamlit's core execution model — the entire script re-running on every interaction — which is a deliberate design choice, not a limitation. Understanding that model is the foundation for everything else in Streamlit.

