
#--------------------             
# Author : Serge Zaugg
# Description : A Streamlit dashboard to illustrate ML performance metrics 
#--------------------

import numpy as np
import streamlit as st
import numpy as np
import streamlit as st
from streamlit import session_state as ss
from utils import make_df, make_fig, get_performance_metrics, get_metrics_thld_free, show_metrics, update_ss
from utils import show_tf_metrics, show_confusion_matrix

#-----------------------
# 1st line 

# compute data, get perf metrics, and make plot 
df = make_df(ss.upar['N_1'], ss.upar['N_2'], ss.upar['mu_1'], ss.upar['mu_2'], ss.upar['sigma_1'], ss.upar['sigma_2'])
df_metrics_thld = get_performance_metrics(df = df, thld = ss["upar"]["dth"])
df_metrics_free = get_metrics_thld_free(df = df)
fig00 = make_fig(df = df, dot_colors = [ss["upar"]["col_a"], ss["upar"]["col_b"]])
fig00.add_vline(x=ss["upar"]["dth"])

# display plot and perf metrics 
with st.container(border=True):

    _, c2, _ = st.columns([0.010, 1.00, 0.015])
    with c2:
        ss["upar"]["dth"] = st.slider(label ="Decision threshold", min_value= 0.0, max_value=1.0, value=ss["upar"]["dth"], 
            key="slide_07", on_change=update_ss, args=["slide_07", "dth"], label_visibility = "visible")
    st.plotly_chart(fig00, use_container_width=True, config={"displayModeBar": False})    



 
#-----------------------
# 2nd line 
col_b0, col_b1, col_b2,= st.columns([0.28, 0.43, 0.80])

with col_b0:
    with st.container(border=True): 
        show_tf_metrics(df_free = df_metrics_free) 

with col_b1:
    with st.container(border=True): 
        show_confusion_matrix(df_thld = df_metrics_thld) 

with col_b2: 
    with st.container(border=True): 
        show_metrics(df_thld = df_metrics_thld)
    


