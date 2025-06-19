#--------------------             
# Author : Serge Zaugg
# Description : Streamlit function and fragments are here
#--------------------

import streamlit as st
import plotly.express as px
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, accuracy_score, confusion_matrix
from streamlit import session_state as ss


def update_ss(kname, ssname):
    """
    description : helper callback fun to implement statefull apps
    kname : key name of widget
    ssname : key name of variable in session state (ss)
    """
    ss["upar"][ssname] = ss[kname]      


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
    class_name_1 = "Negative"
    class_name_2 = "Positive"
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
        height = 300,
        labels={"proba_score": "Score", "jitter": ""},
        title = "",
        )
    _ = fig00.update_xaxes(showline = True, linecolor = 'white', linewidth = 2, row = 1, col = 1, mirror = True)
    _ = fig00.update_yaxes(showline = True, linecolor = 'white', linewidth = 2, row = 1, col = 1, mirror = True)
    _ = fig00.update_traces(marker=dict(size=4))
    _ = fig00.update_layout(xaxis=dict(showgrid=False, zeroline=False), yaxis=dict(showgrid=False, zeroline=False))
    _ = fig00.update_layout(xaxis_range=[-0.00001, +1.00001])
    _ = fig00.update_layout(paper_bgcolor="#000000") # "#350030"
    _ = fig00.update_yaxes(showticklabels=False)
    # text font sizes 
    # _ = fig00.update_layout(title_font_size=25)
    _ = fig00.update_layout(xaxis_title_font_size=25)
    _ = fig00.update_layout(yaxis_title_font_size=25)
    _ = fig00.update_layout(xaxis_tickfont_size=25)
    _ = fig00.update_layout(legend_font_size=20)
    # _ = fig00.update_layout(title_y=0.96)
    _ = fig00.update_layout(showlegend=False)
    _ = fig00.update_layout(yaxis_title=None)
    _ = fig00.update_layout(margin=dict(t=10, b=10, l=15, r=15))
    # _ = fig00.update_layout(xaxis={'side': 'top'}) # , yaxis={'side': 'right'}  )
    # 
    return(fig00)


@st.cache_data
def get_metrics_thld_free(df):
    rauc_val = roc_auc_score(y_true = df['class'], y_score = df['proba_score'])
    avep_val = average_precision_score(y_true = df['class'], y_score = df['proba_score'], pos_label='Positive')
    # convert to nicely formatted string
    rauc_val = "{:.2f}".format(round(rauc_val,2)) 
    avep_val = "{:.2f}".format(round(avep_val,2))
    # combine
    resu = {"ROC-AUC" : rauc_val,  "Average Precision" : avep_val}
    return(resu)                         


@st.cache_data
def get_performance_metrics(df, thld):
    # precision and recall
    y_tru = df['class']=='Positive'
    y_pre = df['proba_score'] > thld 
    precis_val = precision_score(y_true = y_tru, y_pred = y_pre)
    npv_val = precision_score(y_true = np.logical_not(y_tru), y_pred = np.logical_not(y_pre))
    recall_val = recall_score(y_true = y_tru, y_pred = y_pre) 
    accura_val = accuracy_score(y_true = y_tru, y_pred = y_pre)
    specif_val = recall_score(y_true = np.logical_not(y_tru), y_pred = np.logical_not(y_pre)) 
    confmat_val = confusion_matrix(y_tru, y_pre)
    # convert to nicely formatted string
    precis_val = "{:.2f}".format(np.round(precis_val,2)) 
    npv_val = "{:.2f}".format(np.round(npv_val,2)) 
    recall_val = "{:.2f}".format(np.round(recall_val,2))
    accuracy_val = "{:.2f}".format(np.round(accura_val,2))
    specificity_val = "{:.2f}".format(np.round(specif_val,2))
    # combine
    resu = {"Precision" : precis_val , "NPV" : npv_val, "Recall" : recall_val, "Accuracy" : accuracy_val , "Specificity" : specificity_val, "Confusion matrix" : confmat_val}
    return(resu)                 


@st.cache_data
def show_metrics(df_thld, df_free):
    """
    """  
    col1, col2, col3 = st.columns([0.9, 0.9, 0.4])
    col1.text("Per-rows metrics")
    col2.text("Per-columns metrics")
    col3.text("Accuracy")
    
    col1, col2, _, col22, col3, _ , col4, = st.columns([0.4, 0.4, 0.1, 0.4, 0.4, 0.1, 0.4])
    col1.metric("Specificity", df_thld['Specificity'], border=True, help = "TN / (TN+FP)") 
    col2.metric("Sensitivity (Recall)", df_thld['Recall'], border=True, help = "TP / (TP+FN)") 
    col22.metric("NPV", df_thld['NPV'], border=True, help = "TN / (TN+FN)")
    col3.metric("PPV (Precision)", df_thld['Precision'], border=True, help = "TP / (TP+FP)")
    col4.metric("Accuracy", df_thld['Accuracy'], border=True, help = "(TP+TN) / (TP+TN+FP+FN)") 

    st.text("Threshold-free metrics")
    col1, col2, _, col22, col3, _ , col4, = st.columns([0.4, 0.4, 0.1, 0.4, 0.4, 0.1, 0.4])
    col1.metric("ROC-AUC", df_free["ROC-AUC"], border=True)
    col2.metric("Average Precision", df_free["Average Precision"], border=True)

   
@st.cache_data
def show_confusion_matrix(df_thld):
    """
    """  
    tn_val = df_thld["Confusion matrix"][0,0]
    fp_val = df_thld["Confusion matrix"][0,1]
    fn_val = df_thld["Confusion matrix"][1,0]
    tp_val = df_thld["Confusion matrix"][1,1]
    st.text("Confusion matrix") 
    st.text("  ") 
    st.text("  ") 
    st.text("  ") 
    st.text("  ") 
    st.text("  ") 
    col1, col2, = st.columns([0.2, 0.2])
    col1.metric("TN", tn_val, border=True, help = "Negatives below threshold")
    col1.metric("FN", fn_val, border=True, help = "Positives below threshold")
    col2.metric("FP", fp_val, border=True, help = "Negatives above threshold")
    col2.metric("TP", tp_val, border=True, help = "Positives above threshold") 

if __name__ == '__main__':
    # test 
    mu_1 = 0.03
    sigma_1 = 0.20
    aaa = make_one_class_data(5000, mu_1, sigma_1, "aaa")
    [aaa['proba_score'].mean().round(2), mu_1]
    [aaa['proba_score'].std().round(2), sigma_1]




