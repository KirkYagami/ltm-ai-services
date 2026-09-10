import streamlit as st
import pandas as pd
import time

st.set_page_config(page_title="Caching", page_icon="⚡")

st.title("Caching ⚡")


@st.cache_data
def load_slow_data(n_rows: int) -> pd.DataFrame:
    time.sleep(2)
    return pd.DataFrame({"id": range(n_rows), "value": [i * 2 for i in range(n_rows)]})


@st.cache_resource
def get_shared_connection():
    time.sleep(1)
    return {"connection_id": id(object()), "status": "connected"}


n_rows = st.slider("Number of rows", 5, 50, 10)

start = time.time()
df = load_slow_data(n_rows)
st.write(f"Loaded {n_rows} rows in **{time.time() - start:.2f}s**")
st.dataframe(df, height=200)

st.divider()

conn = get_shared_connection()
st.json(conn)

st.divider()

unrelated = st.slider("Unrelated slider", 0, 10, 5)
st.write(f"Value: {unrelated}")
