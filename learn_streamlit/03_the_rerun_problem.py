import streamlit as st

st.set_page_config(page_title="The Rerun Problem", page_icon="❌")

st.title("Click Counter — v1 ❌")

count = 0

if st.button("Click to increment"):
    count += 1

st.write(f"### Count: {count}")
