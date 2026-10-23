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
p0 = st.Page("pages/st_page_00.py", title="ML Summary")
p1 = st.Page("pages/st_page_01.py", title="Interactive Guide")
p2 = st.Page("pages/st_page_about.py", title="About")

pg = st.navigation([p1, p0, p2])
pg.run()

with st.sidebar:
    # logos an links
    c1, c2 = st.columns([55,200])
    st.logo(image='pics/z_logo_violet.png', size="large", link="https://github.com/sergezaugg")

       