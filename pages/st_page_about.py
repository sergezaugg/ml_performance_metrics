#--------------------             
# Author : Serge Zaugg
# Description : Some infos for user
#--------------------

import streamlit as st
from streamlit import session_state as ss

col_aa, col_bb, = st.columns([0.50, 0.40])
with col_aa: 
    with st.container(border=True, ):

        st.markdown("""
        ML performance metrics are fundamental for any ML and Deep Learning application.  
        Here you can get an intuitive understanding of several fundamental metrics with a simple interactive app.
        """)
    
        st.markdown(''' 
        Created by: :primary[[Serge Zaugg](https://www.linkedin.com/in/dkifh34rtn345eb5fhrthdbgf45/)]     
        GitHub: :primary[[Pollito-ML](https://github.com/sergezaugg)]
        ''')


