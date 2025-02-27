#--------------------             
# Author : Serge Zaugg
# Description : Main streamlit entry point
#--------------------

import streamlit as st

st.set_page_config(layout="wide")

p0 = st.Page("st_page_00.py", title="Summary")
p1 = st.Page("st_page_01.py", title="Dashboard")

pg = st.navigation([p1, p0])

pg.run()

# run locally
# streamlit run stmain.py