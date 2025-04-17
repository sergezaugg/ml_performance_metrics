
#--------------------             
# Author : Serge Zaugg
# Description : A Streamlit dashboard to illustrate ML performance metrics 
#--------------------

import numpy as np
import streamlit as st
import numpy as np
# import pandas as pd
import streamlit as st
# import plotly.express as px
# from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, accuracy_score
from streamlit import session_state as ss
from utils import make_df,make_fig, make_fig, get_performance_metrics


# initial value of session state
if 'color_a' not in ss:
    ss.color_a = '#FF00AA'
if 'color_b' not in ss:
    ss.color_b = '#6AFF00'
if 'decision_thld' not in ss:
    ss.decision_thld = 0.5


#-----------------------
# 1st line 
col_a1, col_a2, col_space011,= st.columns([0.20, 0.80, 0.10])

with col_a1: 
    
    with st.container(height=475, border=True, key='conta_01'):

        st.text("Simulate score distribution*")

        col_x1, col_x2, = st.columns([0.50, 0.50])

        with col_x1: 
            st.text('Class A')
            N_1     = st.slider("N",     min_value =  10, max_value=5000,  value=1000, label_visibility = "visible", key="slide_01")
            mu_1    = st.slider("Mean",  min_value = 0.03, max_value=0.97,  value=0.20,  label_visibility = "visible",key="slide_02")
            # dynamically compute feasible upper std 
            upper_lim_1 = 0.90*np.sqrt(mu_1*(1-mu_1)) # to be checked!
            sigma_1 = st.slider("S.D.", min_value = 0.03, max_value=upper_lim_1, value=0.20,  label_visibility = "visible",key="slide_03")
        
        with col_x2: 
            st.text('Class B')
            N_2     = st.slider("N",     min_value=  10, max_value=5000, value=1000, label_visibility = "visible",key="slide_04")
            mu_2    = st.slider("Mean",  min_value= 0.03, max_value=0.97, value=0.80, label_visibility = "visible",key="slide_05")
            # dynamically compute feasible upper std 
            upper_lim_2 = 0.90*np.sqrt(mu_2*(1-mu_2)) # to be checked!
            sigma_2 = st.slider("S.D.", min_value= 0.03, max_value=upper_lim_2, value=0.20, label_visibility = "visible",key="slide_06")

    with st.container(height=None, border=True, key='conta_01b'):
        ss.decision_thld = st.slider("Decision threshold", min_value= 0.0, max_value=1.0, value=0.50,  label_visibility = "visible",key="slide_07")

    with st.container(height=None, border=True, key='conta_01c'):
        c1, c2 = st.columns([0.20, 0.20])
        with c1:
            ss.color_a = st.color_picker("Class A Color", ss.color_a) 
        with c2:
            ss.color_b = st.color_picker("Class B Color", ss.color_b)




df = make_df(N_1, N_2, mu_1, mu_2, sigma_1, sigma_2)

fig00 = make_fig(df = df, dot_colors = [ss.color_a, ss.color_b])

fig00.add_vline(x=ss.decision_thld)

df_perf_metrics = get_performance_metrics(df = df, thld = ss.decision_thld)






with col_a2:
    with st.container(height=475, border=True, key='conta_02'):
        col1, col2, _ = st.columns((0.8, 0.5, 0.2))
        col1.text("Visualize distribution of score")
        st.plotly_chart(fig00, use_container_width=True)

    with st.container(height=None, border=False, key='conta_03'):
        col1, col2 = st.columns([0.4, 0.6])
        col1.subheader("Threshold-free metrics")
        col2.subheader("Threshold dependent metrics")
        col1, col2, col3, col4, col5, = st.columns([0.2, 0.2, 0.2, 0.2, 0.2])
        col1.metric("ROC-AUC", df_perf_metrics["ROC-AUC"], border=True)
        col2.metric("Average Precision (AP)", df_perf_metrics["Average Precision"], border=True)
        col3.metric("Precision", df_perf_metrics['Precision'], border=True)
        col4.metric("Recall", df_perf_metrics['Recall'], border=True)
        col5.metric("Accuracy", df_perf_metrics['Accuracy'], border=True)
       
st.text("""
        * A Beta distribution parametrized with mean and standard deviation (S.D.) is used for each class. Note that some combinations of mean and S.D. are not feasible for the Beta distribution. 
        'Class B' represents the "positive class", i.e. the one to be detected.
        """)


