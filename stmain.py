#--------------------             
# Author : Serge Zaugg
# Description : Main streamlit entry point
# run locally : streamlit run stmain.py
#--------------------

import streamlit as st

st.set_page_config(layout="wide")

p0 = st.Page("st_page_00.py", title="Summary")
p1 = st.Page("st_page_01.py", title="Dashboard")

pg = st.navigation([p0, p1])

pg.run()

with st.sidebar:
    st.text("Under devel. (v0.9.0)")
    st.title(""); st.title(""); st.title(""); st.title(""); st.title(""); st.title(""); st.title("")
    st.title(""); st.title(""); st.title(""); st.title("") 
    st.markdown(''':gray[RELATED TOPICS]''')
    st.page_link("https://purenoisefeatures.streamlit.app/", label=":gray[ml-performance-metrics]")
    st.page_link("https://featureimportance.streamlit.app/", label=":gray[feature-importance:red]")