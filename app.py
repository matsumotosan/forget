import streamlit as st


pg = st.navigation(
    [
        st.Page("app_home.py", title="Machine Unlearning"),
        st.Page("app_cifar.py", title="CIFAR-10 Classifiction"),
        st.Page("app_face.py", title="Face Recognition"),
    ]
)

pg.run()
