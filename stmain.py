#--------------------             
# Author : Serge Zaugg
# Description : Main streamlit entry point
# run locally : streamlit run stmain.py
#--------------------

import streamlit as st
from streamlit import session_state as ss
from utils import update_ss
import numpy as np

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

    with st.container(border=True):
        st.text("Simulate score distribution")
        col_x1, col_x2, = st.columns([0.50, 0.50])
        with col_x1: 
            st.text('Negatives')
            ss.upar['N_1'] = st.number_input("N", min_value=1, max_value=10000, value=ss.upar['N_1'], step=10, key = "Class_A_001", on_change=update_ss, args=["Class_A_001", "N_1"])
            ss.upar['mu_1'] = st.slider("Mean", min_value = 0.03, max_value=0.97, value=ss.upar['mu_1'], label_visibility = "visible", key = "Class_A_002", on_change = update_ss, args=["Class_A_002", "mu_1"])
            # dynamically compute feasible upper std 
            upper_lim = 0.90*np.sqrt(ss.upar['mu_1']*(1-ss.upar['mu_1'])) 
            ss.upar['sigma_1'] = st.slider("Standard Deviation", min_value = 0.03, max_value=upper_lim, value=min(upper_lim, ss.upar['sigma_1']),  
                                        label_visibility = "visible", key = "Class_A_003", on_change = update_ss, args=["Class_A_003", "sigma_1"])
            ss["upar"]["col_a"] = st.color_picker("Color", ss["upar"]["col_a"])   
        with col_x2: 
            st.text('Positives')
            ss.upar['N_2'] = st.number_input("N", min_value=1, max_value=10000, value=ss.upar['N_2'], step=10, key = "Class_B_001", on_change=update_ss, args=["Class_B_001", "N_2"])
            ss.upar['mu_2']    = st.slider("Mean", min_value = 0.03, max_value=0.97, value=ss.upar['mu_2'], label_visibility = "visible", key = "Class_B_002", on_change=update_ss, args=["Class_B_002", "mu_2"])
            # dynamically compute feasible upper std 
            upper_lim = 0.90*np.sqrt(ss.upar['mu_2']*(1-ss.upar['mu_2'])) 
            ss.upar['sigma_2'] = st.slider("Standard Deviation", min_value = 0.03, max_value=upper_lim, value=min(upper_lim, ss.upar['sigma_2']),  
                                        label_visibility = "visible", key = "Class_B_003", on_change = update_ss, args=["Class_B_003", "sigma_2"])
            ss["upar"]["col_b"] = st.color_picker("Color", ss["upar"]["col_b"])



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
    c1, c2 = st.columns([55,200])
    c1.image(image='pics/z_logo_violet.png', width=65)
    c2.markdown('''
    :primary[v1.2.0]  
    :primary[Created by]
    :primary[[Serge Zaugg](https://www.linkedin.com/in/dkifh34rtn345eb5fhrthdbgf45)]  
    :primary[[Pollito-ML](https://github.com/sergezaugg)]
    ''')
    st.logo(image='pics/z_logo_violet.png', size="large", link="https://github.com/sergezaugg")

       