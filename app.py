#--------------------             
# Author : Serge Zaugg
# Description : Main streamlit entry point
# run locally : streamlit run stmain.py
#--------------------

import streamlit as st
from streamlit import session_state as ss
from utils import update_ss
import numpy as np
from resources.styles import apply_global_styles

st.set_page_config(layout="wide", initial_sidebar_state="expanded", page_title="ML metrics", page_icon='pics/z_logo_violet.png')
apply_global_styles() # apply custom CSS styles 

# change color of navigation bar background       
st.markdown("""
    <style>
    .stAppHeader {background-color: #000000;}
    </style>
""", unsafe_allow_html=True)

# add main title in top bar via CSS 
st.markdown("""
<style>
    .app-title {
        position: fixed;
        top: +13px;  /* decrease to move up */
        font-size: 27px;
        font-weight: 600;
        color: #a064ff;
        z-index: 999999;
        pointer-events: none;
    }
</style>
<div class="app-title">Interactive Guide to Machine Learning Performance Metrics</div>
""", unsafe_allow_html=True)

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
    # st.markdown(":violet[**Guide to Machine Learning Performance Metrics**]") 
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

    # logos an links
    c1, c2 = st.columns([55,200])
    st.logo(image='pics/z_logo_violet.png', size="large", link="https://github.com/sergezaugg")

       