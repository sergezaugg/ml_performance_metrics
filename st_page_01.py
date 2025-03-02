
#--------------------             
# Author : Serge Zaugg
# Description : A Streamlit dashboard to illustrate ML performance metrics 
#--------------------

import numpy as np
import streamlit as st
import plotly.express as px
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, accuracy_score
from streamlit import session_state as ss

# initial value of session state
if 'color_a' not in ss:
    ss.color_a = '#FF00AA'
if 'color_b' not in ss:
    ss.color_b = '#6AFF00'
if 'decision_thld' not in ss:
    ss.decision_thld = 0.5

#-----------------------
# define

@st.cache_data
def make_one_class_data(N, mu, sigma, class_name):
    sigma2 = sigma**2
    var_max = mu*(1-mu)
    sdt_max = np.sqrt(var_max)
    assert (sigma2 <= var_max) , "sigma must be <= " + str(sdt_max)
    assert (sigma2 > 0 ) , "sigma must be > 0" 
    a = mu*(mu*(1-mu)/sigma2 - 1)
    b = a*(1-mu)/mu
    vals = np.random.beta(a = a, b = b, size = N) 
    df = pd.DataFrame({"proba_score":vals})
    df['class'] = class_name
    return(df)

# # test 
# mu_1 = 0.5
# sigma_1 = 0.44
# aaa = make_one_class_data(5000, mu_1, sigma_1, "aaa")
# [aaa['proba_score'].mean().round(2), mu_1]
# [aaa['proba_score'].std().round(2), sigma_1]


@st.cache_data
def make_df(N_1, N_2, mu_1, mu_2, sigma_1, sigma_2):
    class_name_1 = "Class A"
    class_name_2 = "Class B"
    df = pd.concat([
        make_one_class_data(N_1, mu_1, sigma_1, class_name_1) ,
        make_one_class_data(N_2, mu_2, sigma_2, class_name_2) 
        ])
    df['jitter'] = np.random.uniform(size=N_1+N_2)
    df['jitter'][df['class'] == class_name_1] += 1
    return(df)

@st.cache_data
def make_fig(df, dot_colors):
    fig00 = px.scatter(
        data_frame = df,
        x = 'proba_score',
        y = 'jitter',
        color = 'class',
        color_discrete_sequence = dot_colors,
        template='plotly_dark',
        width = 900,
        height = 370,
         labels={"proba_score": "Score", "jitter": "Random jitter"},
        )
    _ = fig00.update_xaxes(showline = True, linecolor = 'white', linewidth = 2, row = 1, col = 1, mirror = True)
    _ = fig00.update_yaxes(showline = True, linecolor = 'white', linewidth = 2, row = 1, col = 1, mirror = True)
    _ = fig00.update_traces(marker=dict(size=4))
    _ = fig00.update_layout(xaxis=dict(showgrid=False), yaxis=dict(showgrid=False))
    _ = fig00.update_layout(xaxis_range=[-0.01, +1.01])
    _ = fig00.update_layout(paper_bgcolor="#350030",)
    return(fig00)
    # fig00.show()

@st.cache_data
def get_performance_metrics(df, thld):
    rauc_val = roc_auc_score(y_true = df['class'], y_score = df['proba_score'])
    avep_val = average_precision_score(y_true = df['class'], y_score = df['proba_score'], pos_label='Class B')
    # precision and recall
    y_tru = df['class']=='Class B'
    y_pre = df['proba_score'] > thld 
    precis_val = precision_score(y_true = y_tru, y_pred = y_pre)
    recall_val = recall_score(y_true = y_tru, y_pred = y_pre) 
    accura_val = accuracy_score(y_true = y_tru, y_pred = y_pre)
    # convert to nicely formatted string
    rauc_val = "{:.2f}".format(rauc_val.round(2)) 
    avep_val = "{:.2f}".format(avep_val.round(2))
    precis_val = "{:.2f}".format(np.round(precis_val,2)) 
    recall_val = "{:.2f}".format(np.round(recall_val,2))
    accuracy_val = "{:.2f}".format(np.round(accura_val,2))
    # combine
    resu = {"ROC-AUC" : rauc_val,  "Average Precision" : avep_val ,  "Precision" : precis_val , "Recall" : recall_val, "Accuracy" : accuracy_val }
    return(resu)                         


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
            mu_1    = st.slider("Mean",  min_value = 0.01, max_value=0.99,  value=0.20,  label_visibility = "visible",key="slide_02")
            # dynamically compute feasible upper std 
            upper_lim_1 = 0.98*np.sqrt(mu_1*(1-mu_1))
            sigma_1 = st.slider("S.D.", min_value = 0.01, max_value=upper_lim_1, value=0.20,  label_visibility = "visible",key="slide_03")
        
        with col_x2: 
            st.text('Class B')
            N_2     = st.slider("N",     min_value=  10, max_value=5000, value=1000, label_visibility = "visible",key="slide_04")
            mu_2    = st.slider("Mean",  min_value= 0.01, max_value=0.99, value=0.80, label_visibility = "visible",key="slide_05")
            # dynamically compute feasible upper std 
            upper_lim_2 = 0.98*np.sqrt(mu_2*(1-mu_2))
            sigma_2 = st.slider("S.D.", min_value= 0.01, max_value=upper_lim_2, value=0.20, label_visibility = "visible",key="slide_06")

    with st.container(height=None, border=True, key='conta_01b'):
        ss.decision_thld = st.slider("Decision threshold", min_value= 0.0, max_value=1.0, value=0.50,  label_visibility = "visible",key="slide_07")

    with st.container(height=None, border=True, key='conta_01c'):
        c1, c2, c3 = st.columns([0.20, 0.20, 0.40])
        with c1:
            ss.color_a = st.color_picker("Class A Color", ss.color_a) 
        with c2:
            ss.color_b = st.color_picker("Class B Color", ss.color_b)
        with c3:
            st.text("")
            st.text("")
            st.button("Confirm colors")
    
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
       
st.divider()

st.text("""
        * A Beta distribution parametrized with mean and standard deviation (S.D) is used for each class. 
        Note that some combinations of mean and S.D are not feasible for the Beta distribution. 

        Note that 'Class B' represents the "positive class", i.e. the one to be detected.
        """)


st.page_link("st_page_00.py", label="LINK : Summary with context and explanations", icon = "💜")

