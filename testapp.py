import streamlit as st

@st.cache_resource
def test_import():
    import sklearn
    return "OK"

st.write(test_import())