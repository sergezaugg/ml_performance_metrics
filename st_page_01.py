
#--------------------             
# Author : Serge Zaugg
# Description : A Streamlit dashboard to illustrate ML performance metrics 
#--------------------

import numpy as np
import streamlit as st
import numpy as np
import streamlit as st
from streamlit import session_state as ss
from utils import make_df,make_fig, make_fig, get_performance_metrics, get_safe_params, frag_show_plot

#-----------------------
# 1st line 
col_a1, col_a2, col_space011,= st.columns([0.20, 0.80, 0.10])

# get user input
with col_a1: 
    with st.container(height=475, border=True, key='conta_01'):
        st.text("Simulate score distribution *")
        col_x1, col_x2, = st.columns([0.50, 0.50])
        with col_x1: 
            st.text('Class A')
            N_1, mu_1, sigma_1 = get_safe_params(k = "aa", init_mu = 0.20)
        with col_x2: 
            st.text('Class B °')
            N_2, mu_2, sigma_2 = get_safe_params(k = "bb", init_mu = 0.80)
    with st.container(height=None, border=True, key='conta_01b'):
        ss.decision_thld = st.slider("Decision threshold", min_value= 0.0, max_value=1.0, value=0.50,  label_visibility = "visible",key="slide_07")
    with st.container(height=None, border=True, key='conta_01c'):
        c1, c2 = st.columns([0.20, 0.20])
        with c1:
            ss.color_a = st.color_picker("Class A Color", ss.color_a) 
        with c2:
            ss.color_b = st.color_picker("Class B Color", ss.color_b)

# compute data, get perf metrics, and make plot 
df = make_df(N_1, N_2, mu_1, mu_2, sigma_1, sigma_2)
df_perf_metrics = get_performance_metrics(df = df, thld = ss.decision_thld)
fig00 = make_fig(df = df, dot_colors = [ss.color_a, ss.color_b])
fig00.add_vline(x=ss.decision_thld)

# display plot and perf metrics 
with col_a2:
    frag_show_plot(fig00, df_perf_metrics)
      
st.text("""
        * A Beta distribution parametrized with mean and standard deviation (S.D.) is used for each class. Note that some combinations of mean and S.D. are not feasible for the Beta distribution. 
        ° Class B represents the positive class, i.e. the one to be detected.
        """)


