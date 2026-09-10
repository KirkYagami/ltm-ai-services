import streamlit as st

st.set_page_config(page_title="Why Streamlit?", page_icon="🎈", layout="centered")

st.title("Welcome to Streamlit 🎈")
st.subheader("Built entirely with Python")

name = st.text_input("What's your name?")
experience = st.slider("Years of programming experience", 0, 20, 1)
language = st.selectbox("Favourite programming language", ["Python", "Java", "C++", "JavaScript", "Other"])
likes_streamlit = st.checkbox("I think Streamlit looks useful so far")

if st.button("Say hello"):
    st.success(f"Hello {name or 'there'}! {experience} years of experience, and you like {language}.")
    if likes_streamlit:
        st.balloons()
