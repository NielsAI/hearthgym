import streamlit as st
import pandas as pd

st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load the Streamlit navigation module
pages = {
    "General": [
        st.Page("navigation/0_General_Analysis.py", title="General Analysis"),
    ],
    "Action Analysis": [
        st.Page("navigation/1_Action_Counts.py", title="Action Counts"),
        st.Page("navigation/2_Action_Sequences.py", title="Action Sequences"),
        st.Page("navigation/3_Action_Networks.py", title="Action Networks"),
    ],
    "Tensorboard": [
        st.Page("navigation/4_Tensorboard.py", title="Tensorboard"),
    ],
}
           
# Create the navigation bar and run the Dashboard
pg = st.navigation(pages)
pg.run()