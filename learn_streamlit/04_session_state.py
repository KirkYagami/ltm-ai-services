import streamlit as st

st.set_page_config(page_title="Session State", page_icon="✅")

st.title("Click Counter — v2 ✅")

st.text_input("Kuch bhii")

if "count" not in st.session_state:
    st.session_state.count = 0

col1, col2 = st.columns(2)

with col1:
    if st.button("➕ Increment"):
        st.session_state.count += 1

with col2:
    if st.button("🔄 Reset"):
        st.session_state.count = 0

st.write(f"### Count: {st.session_state.count}")

with st.expander("session_state contents"):
    st.json(dict(st.session_state))


def reset_age():
    st.session_state.age = 0

st.text_input(
    "Name",
    key="name",
    on_change=reset_age
)

st.number_input(
    "Age",
    key="age"
)