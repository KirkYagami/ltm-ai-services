import streamlit as st
from datetime import datetime
import time

print(f"SCRIPT RUN at {datetime.now().strftime('%H:%M:%S.%f')}")

st.set_page_config(page_title="Execution Model", page_icon="🔁")

st.title("Streamlit's Execution Model 🔁")
st.write(f"This script last ran at: `{datetime.now().strftime('%H:%M:%S.%f')}`")

with st.spinner("Working..."):
    time.sleep(0.3)

st.divider()

value = st.slider("Move me", 0, 100, 50)
st.write(f"Slider value: **{value}**")

if st.button("Click me"):
    st.write("Button clicked.")
