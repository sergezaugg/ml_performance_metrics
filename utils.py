#--------------------             
# Author : Serge Zaugg
# Description : Streamlit function and fragments are here
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


@st.cache_data
def make_one_class_data(N, mu, sigma, class_name):
    sigma2 = sigma**2
    a = mu*(mu*(1-mu)/sigma2 - 1)
    b = a*(1-mu)/mu
    vals = np.random.beta(a = a, b = b, size = N) 
    df = pd.DataFrame({"proba_score":vals})
    df['class'] = class_name
    return(df)


@st.cache_data
def make_df(N_1, N_2, mu_1, mu_2, sigma_1, sigma_2):
    class_name_1 = "Background"
    class_name_2 = "Target"
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


@st.cache_data
def get_performance_metrics(df, thld):
    rauc_val = roc_auc_score(y_true = df['class'], y_score = df['proba_score'])
    avep_val = average_precision_score(y_true = df['class'], y_score = df['proba_score'], pos_label='Target')
    # precision and recall
    y_tru = df['class']=='Target'
    y_pre = df['proba_score'] > thld 
    precis_val = precision_score(y_true = y_tru, y_pred = y_pre)
    recall_val = recall_score(y_true = y_tru, y_pred = y_pre) 
    accura_val = accuracy_score(y_true = y_tru, y_pred = y_pre)

    specif_val = recall_score(y_true = np.logical_not(y_tru), y_pred = np.logical_not(y_pre)) 

    # convert to nicely formatted string
    rauc_val = "{:.2f}".format(rauc_val.round(2)) 
    avep_val = "{:.2f}".format(avep_val.round(2))
    precis_val = "{:.2f}".format(np.round(precis_val,2)) 
    recall_val = "{:.2f}".format(np.round(recall_val,2))
    accuracy_val = "{:.2f}".format(np.round(accura_val,2))
    specificity_val = "{:.2f}".format(np.round(specif_val,2))
    # combine
    resu = {"ROC-AUC" : rauc_val,  "Average Precision" : avep_val ,  "Precision" : precis_val , "Recall" : recall_val, "Accuracy" : accuracy_val , "Specificity" : specificity_val}
    return(resu)                         


@st.fragment
def frag_show_plot(fig, df_perf_metrics):
    """
    
    """
    with st.container(height=475, border=True, key='conta_02'):
        col1, col2, _ = st.columns((0.8, 0.5, 0.2))
        col1.text("Visualize score distribution")
        st.plotly_chart(fig, use_container_width=True)
    with st.container(height=None, border=False, key='conta_03'):
        col1, col2 = st.columns([0.4, 0.8])
        col1.subheader("Threshold-free metrics")
        col2.subheader("Threshold-dependent metrics")
        col1, col2, col3, col4, col5, col6, = st.columns([0.2, 0.2, 0.2, 0.2, 0.2, 0.2])
        col1.metric("ROC-AUC", df_perf_metrics["ROC-AUC"], border=True)
        col2.metric("Average Precision", df_perf_metrics["Average Precision"], border=True)
        col3.metric("Precision", df_perf_metrics['Precision'], border=True)
        col4.metric("Recall (Sensitivity)", df_perf_metrics['Recall'], border=True)
        col5.metric("Specificity", df_perf_metrics['Specificity'], border=True)  
        col6.metric("Accuracy", df_perf_metrics['Accuracy'], border=True)     



if __name__ == '__main__':
    # test 
    mu_1 = 0.03
    sigma_1 = 0.20
    aaa = make_one_class_data(5000, mu_1, sigma_1, "aaa")
    [aaa['proba_score'].mean().round(2), mu_1]
    [aaa['proba_score'].std().round(2), sigma_1]

