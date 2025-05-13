
#--------------------             
# Author : Serge Zaugg
# Description : A Streamlit dashboard to illustrate ML performance metrics 
#--------------------

import numpy as np
import streamlit as st
import numpy as np
import streamlit as st
from streamlit import session_state as ss
from utils import make_df, make_fig, get_performance_metrics, get_metrics_thld_free, show_metrics, update_ss, show_confusion_matrix

#-----------------------
# 1st line 
col_a1, col_a2, col_a3,= st.columns([0.25, 0.60, 0.25])

# get user input
with col_a1: 
    with st.container(height=480, border=True):
        st.text("(1) Simulate score distribution")
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

# compute data, get perf metrics, and make plot 
df = make_df(ss.upar['N_1'], ss.upar['N_2'], ss.upar['mu_1'], ss.upar['mu_2'], ss.upar['sigma_1'], ss.upar['sigma_2'])
df_metrics_thld = get_performance_metrics(df = df, thld = ss["upar"]["dth"])
df_metrics_free = get_metrics_thld_free(df = df)
fig00 = make_fig(df = df, dot_colors = [ss["upar"]["col_a"], ss["upar"]["col_b"]])
fig00.add_vline(x=ss["upar"]["dth"])

# display plot and perf metrics 
with col_a2:
    with st.container(height=480, border=True):
   
        _, c2, _ = st.columns([0.01, 1.00, 0.015])
        with c2:
            st.text("(2) Decision threshold")
            ss["upar"]["dth"] = st.slider("(2) Decision threshold", min_value= 0.0, max_value=1.0, value=ss["upar"]["dth"], 
                                        key="slide_07", on_change=update_ss, args=["slide_07", "dth"], label_visibility= "hidden")
        st.plotly_chart(fig00, use_container_width=True)    
 
with col_a3:  
    with st.container(height=480, border=True):
        show_confusion_matrix(df_thld = df_metrics_thld)

#-----------------------
# 2nd line 
col_b1, col_b2,= st.columns([0.25, 0.85])

with col_b1:
    with st.container(height=400, border=True): 
        st.markdown("""
        **Positives** = items to be detected
                    
        **Negatives** = items not of interest 
                    
        **TP** = True Positives
                    
        **TN** = True Negatives
                    
        **FP** = False Positives 
                    
        **FN** = False Negatives 
                    
        **PPV** = Positive Predictive Value  
                    
        **NPV** = Negative Predictive Value
                    
        """)        

with col_b2: 
    with st.container(height=400,border=True): 
        show_metrics(df_thld = df_metrics_thld, df_free = df_metrics_free)
    


