#--------------------             
# Author : Serge Zaugg
# Description : Main streamlit entry point
# run locally : streamlit run stmain.py
#--------------------

import streamlit as st
from streamlit import session_state as ss

st.set_page_config(layout="wide", initial_sidebar_state="expanded")

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
    st.markdown(":violet[**Illustration of machine learning performance metrics and diagnostic tests**]") 
    # st.title(""); st.title(""); st.title(""); st.title(""); st.title("")
    st.title("")
    with st.container(border=True): 
        st.markdown("""
        ## Terminology:              
        **Positives** = items to be detected  
        **Negatives** = items not of interest  
        **TP** = True Positives  
        **TN** = True Negatives  
        **FP** = False Positives  
        **FN** = False Negatives  
        **PPV** = Positive Predictive Value  
        **NPV** = Negative Predictive Value  
               
        """)        
    # logos an links
    st.header("");st.header("");st.header("")
    st.header("");st.header("");st.header("")
    c1,c2=st.columns([80,200])
    c1.text("")
    c1.image(image='pics/z_logo_violet.png', width=65)
    c2.markdown(''':primary[v1.1.4]  
    :primary[Created by]
    :primary[[Serge Zaugg](https://www.linkedin.com/in/dkifh34rtn345eb5fhrthdbgf45)   ]
    :primary[[Pollito-ML](https://github.com/sergezaugg)]
    ''')
    st.logo(image='pics/z_logo_violet.png', size="large", link="https://github.com/sergezaugg")

       