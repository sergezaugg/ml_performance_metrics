
#--------------------             
# Author : Serge Zaugg
# Description : A Streamlit dashboard to illustrate ML performance metrics 
#--------------------

import numpy as np
import streamlit as st
import numpy as np
import streamlit as st
from streamlit import session_state as ss
from utils import make_df,make_fig, make_fig, get_performance_metrics, frag_show_plot

# un.elegant solution to make params more permanent in session
def store_value():
    ss["upar"]["dth"] = ss["slide_07"]

def store_mu1():
    ss["upar"]['mu_1'] = ss["Class_A_002"]    

def store_N1():
    ss["upar"]['N_1'] = ss["Class_A_001"]    

def store_sigma1():
    ss["upar"]['sigma_1'] = ss["Class_A_003"]    

def store_mu2():
    ss["upar"]['mu_2'] = ss["Class_B_002"]    

def store_N2():
    ss["upar"]['N_2'] = ss["Class_B_001"]    

def store_sigma2():
    ss["upar"]['sigma_2'] = ss["Class_B_003"]    


#-----------------------
# 1st line 
col_a1, col_a2, col_space011,= st.columns([0.20, 0.80, 0.10])

# get user input
with col_a1: 
    with st.container(height=475, border=True, key='conta_01'):
        st.text("Simulate score distribution *")
        col_x1, col_x2, = st.columns([0.50, 0.50])
        with col_x1: 
            st.text('Negative')
            st.text('Class')
            ss.upar['N_1']     = st.slider("N",     min_value =  10, max_value=5000,   value=ss.upar['N_1'], label_visibility = "visible", key = "Class_A_001", on_change = store_N1)
            ss.upar['mu_1']    = st.slider("Mean",  min_value = 0.03, max_value=0.97,  value=ss.upar['mu_1'],  label_visibility = "visible", key = "Class_A_002", on_change = store_mu1)
            # dynamically compute feasible upper std 
            upper_lim = 0.90*np.sqrt(ss.upar['mu_1']*(1-ss.upar['mu_1'])) 
            ss.upar['sigma_1'] = st.slider("S.D.", min_value = 0.03, max_value=upper_lim, value=min(upper_lim, ss.upar['sigma_1']),  
                                           label_visibility = "visible", key = "Class_A_003", on_change = store_sigma1)
        with col_x2: 
            st.text('Positive °')
            st.text('Class')
            ss.upar['N_2']     = st.slider("N",     min_value =  10, max_value=5000,   value=ss.upar['N_2'], label_visibility = "visible", key = "Class_B_001",on_change = store_N2)
            ss.upar['mu_2']    = st.slider("Mean",  min_value = 0.03, max_value=0.97,  value=ss.upar['mu_2'],  label_visibility = "visible", key = "Class_B_002", on_change = store_mu2)
            # dynamically compute feasible upper std 
            upper_lim = 0.90*np.sqrt(ss.upar['mu_2']*(1-ss.upar['mu_2'])) 
            ss.upar['sigma_2'] = st.slider("S.D.", min_value = 0.03, max_value=upper_lim, value=min(upper_lim, ss.upar['sigma_2']),  
                                           label_visibility = "visible", key = "Class_B_003", on_change = store_sigma2)

    with st.container(height=None, border=True, key='conta_01b'):
        ss["upar"]["dth"] = st.slider("Decision threshold", min_value= 0.0, max_value=1.0, value=ss["upar"]["dth"],  label_visibility = "visible", key="slide_07", on_change=store_value)

    with st.container(height=None, border=True, key='conta_01c'):
        c1, c2 = st.columns([0.20, 0.20])
        with c1:
            ss["upar"]["col_a"] = st.color_picker("Negative", ss["upar"]["col_a"]) 
        with c2:
            ss["upar"]["col_b"] = st.color_picker("Positive", ss["upar"]["col_b"])
            
    st.text("""
            * A Beta distribution parametrized with mean and standard deviation (S.D.) is used for each class. Note that some combinations of mean and S.D. are not feasible for the Beta distribution. 
            ° Positive class is the class to be detected.
            """)        


# compute data, get perf metrics, and make plot 
df = make_df(ss.upar['N_1'], ss.upar['N_2'], ss.upar['mu_1'], ss.upar['mu_2'], ss.upar['sigma_1'], ss.upar['sigma_2'])
df_perf_metrics = get_performance_metrics(df = df, thld = ss["upar"]["dth"])
fig00 = make_fig(df = df, dot_colors = [ss["upar"]["col_a"], ss["upar"]["col_b"]])
fig00.add_vline(x=ss["upar"]["dth"])

# display plot and perf metrics 
with col_a2:
    frag_show_plot(fig00, df_perf_metrics)

  



  
# st.dataframe(df_perf_metrics["Confusion matrix"], hide_index = False)

# st.markdown('''
#     :red[Streamlit] :orange[can] :green[write] :blue[text] :violet[in]
            
#     :gray[pretty] :rainbow[colors] and :blue-background[highlight] text.''')



