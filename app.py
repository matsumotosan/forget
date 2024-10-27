import streamlit as st

pg = st.navigation([
    st.Page("app_mnist.py", title="Digit Recognition"),
    st.Page("app_face.py", title="Face Recognition"),
])

pg.run()
