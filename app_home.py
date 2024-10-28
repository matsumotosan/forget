import streamlit as st


st.title("Machine Unlearning")

st.write(
    'Machine unlearning arises from concerns surrounding privacy, security, and the "right to be forgotten" '
    "- a particularly critical concern for applications that handle sensitive information. "
    "A naive retraining of models on datasets modified to exclude the data to be forgotten or unlearned is infeasible for large models. "
    "Machine unlearning is not limited to applications concerned with privacy and security as the "
    "same techniques may be applied to address the effects of outliers or outdated samples in the original training data. "
    "With these concerns, many machine unlearning methods have been "
    "proposed in the recent years. However, in part due to the lack of a clear understanding and "
    'definition of what constitutes "unlearning", it is unclear if models are unlearning what is asked '
    "of them or if they are simply learning to conceal them. We are interested in investigating this last question in our final project."
)
