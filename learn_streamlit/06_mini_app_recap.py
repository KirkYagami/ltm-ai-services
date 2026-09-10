import streamlit as st
import time

st.set_page_config(page_title="Mini Quiz App", page_icon="🧠")

st.title("Mini Quiz App 🧠")


@st.cache_data
def load_questions():
    time.sleep(1.5)
    return [
        {"q": "What does HTML stand for?",
         "options": ["Hyper Trainer Markup Language", "Hyper Text Markup Language", "High Text Markup Language"],
         "answer": "Hyper Text Markup Language"},
        {"q": "Streamlit apps are written in which language?",
         "options": ["JavaScript", "Python", "PHP"],
         "answer": "Python"},
        {"q": "What re-runs on every widget interaction in Streamlit?",
         "options": ["Only the widget's code", "The entire script", "Nothing"],
         "answer": "The entire script"},
        {"q": "What keeps a value alive across reruns for one user?",
         "options": ["A normal variable", "st.cache_data", "st.session_state"],
         "answer": "st.session_state"},
    ]


questions = load_questions()

if "q_index" not in st.session_state:
    st.session_state.q_index = 0
if "score" not in st.session_state:
    st.session_state.score = 0
if "finished" not in st.session_state:
    st.session_state.finished = False

if st.session_state.finished:
    st.success(f"Score: {st.session_state.score} / {len(questions)}")
    if st.button("Restart"):
        st.session_state.q_index = 0
        st.session_state.score = 0
        st.session_state.finished = False
        st.rerun()
else:
    current = questions[st.session_state.q_index]
    st.write(f"**Question {st.session_state.q_index + 1} of {len(questions)}**")
    st.write(current["q"])

    choice = st.radio("Choose one:", current["options"], key=f"choice_{st.session_state.q_index}")

    if st.button("Submit"):
        if choice == current["answer"]:
            st.session_state.score += 1
            st.success("Correct!")
        else:
            st.error(f"Incorrect. Answer: {current['answer']}")

        if st.session_state.q_index + 1 < len(questions):
            st.session_state.q_index += 1
        else:
            st.session_state.finished = True

st.caption(f"Score: {st.session_state.score}")
