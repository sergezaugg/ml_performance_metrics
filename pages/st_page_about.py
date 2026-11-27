#--------------------             
# Author : Serge Zaugg
# Description : Some infos for user
#--------------------

import streamlit as st
from streamlit import session_state as ss

col_aa, col_bb, = st.columns([1.00, 0.20])
with col_aa: 
    with st.container(border=True, ):
        st.markdown(''' 
        Created by: :primary[Serge Zaugg]    
        GitHub: :primary[[Pollito-ML](https://github.com/sergezaugg)]
        ''')


