#--------------------             
# Author : Serge Zaugg
# Description : Main streamlit entry point
# run locally : streamlit run stmain.py
#--------------------

import streamlit as st
from streamlit import session_state as ss

st.set_page_config(layout="wide")

# initial value of session state
if 'upar' not in ss:
    ss["upar"] = {
        "col_a" : '#FF00FF',
        "col_b" : '#6AFF00',
        "dth" : 0.5,
        "N_1" : 1000,
        "mu_1" : 0.20,
        "sigma_1" : 0.20,
        "N_2" : 1000,
        "mu_2" : 0.80,
        "sigma_2" : 0.20,
        }

# make navigation
p0 = st.Page("st_page_00.py", title="Summary")
p1 = st.Page("st_page_01.py", title="Interactive")
pg = st.navigation([p1, p0])
pg.run()

with st.sidebar:
    st.markdown(":violet[**Explain and understand machine learning performance metrics**]") 
    st.text("v1.1.1")
    st.title(""); 
    st.markdown("**QUICK GUIDE**") 
    st.markdown("**(1)** Finetune score distribution") 
    st.markdown("**(2)** Set decision threshold") 
    st.markdown("**(3)** Check confusion matrix") 
    st.markdown("**(4)** See effect on metrics") 
    st.title(""); st.title(""); st.title(""); st.title("")
    st.markdown(''':gray[RELATED TOPICS]''')
    st.page_link("https://purenoisefeatures.streamlit.app/", label=":gray[pure-noise-features]")
    st.page_link("https://scikit-learn.org/stable/api/sklearn.metrics.html", label=":gray[scikit-learn metrics]")

    