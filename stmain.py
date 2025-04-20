#--------------------             
# Author : Serge Zaugg
# Description : Main streamlit entry point
# run locally : streamlit run stmain.py
#--------------------

import streamlit as st
from streamlit import session_state as ss

st.set_page_config(layout="wide")

# initial value of session state
if 'color_a' not in ss:
    ss["color_a"] = '#FF00AA'
if 'color_b' not in ss:
    ss.color_b = '#6AFF00'
if 'decision_thld' not in ss:
    ss.decision_thld = 0.5
if 'upar' not in ss:
    ss.upar = {
        "N_1" : 1000,
        "mu_1" : 0.20,
        "sigma_1" : 0.20,

        }



# make navigation
p0 = st.Page("st_page_00.py", title="Summary")
p1 = st.Page("st_page_01.py", title="Interactive")
pg = st.navigation([p1, p0])
pg.run()

with st.sidebar:
    st.markdown(":violet[**Explain and understand machine learning performance metrics**]") 
    st.text("v0.9.2")
    st.title(""); st.title(""); st.title(""); st.title(""); st.title(""); st.title(""); st.title("")
    st.title(""); st.title(""); st.title(""); st.title("") 
    st.markdown(''':gray[RELATED TOPICS]''')
    st.page_link("https://purenoisefeatures.streamlit.app/", label=":gray[pure-noise-features]")
    st.page_link("https://featureimportance.streamlit.app/", label=":gray[feature-importance]")