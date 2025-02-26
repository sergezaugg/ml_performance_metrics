
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
    ss.color_a = '#D100FF'
if 'color_b' not in ss:
    ss.color_b = '#33AAFF'
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


def make_fig(dot_colors):
    fig00 = px.scatter(
        data_frame = df,
        x = 'proba_score',
        y = 'jitter',
        color = 'class',
        color_discrete_sequence = dot_colors,
        template='plotly_dark',
        width = 900,
        height = 500,
         labels={"proba_score": "Score", "jitter": "Random jitter"},
        )
    _ = fig00.update_xaxes(showline = True, linecolor = 'white', linewidth = 2, row = 1, col = 1, mirror = True)
    _ = fig00.update_yaxes(showline = True, linecolor = 'white', linewidth = 2, row = 1, col = 1, mirror = True)
    _ = fig00.update_traces(marker=dict(size=4))
    _ = fig00.update_layout(xaxis=dict(showgrid=False), yaxis=dict(showgrid=False))
    _ = fig00.update_layout(xaxis_range=[-0.01, +1.01])
    _ = fig00.update_layout(paper_bgcolor="#444444",)
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
col_a1, col_space01, col_a2, col_space011,= st.columns([0.20, 0.05, 0.80, 0.10])

with col_a1: 
    st.subheader("Distribution params")

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

    st.subheader("Decision threshold")
    ss.decision_thld = st.slider("thld", min_value= 0.0, max_value=1.0, value=0.50,  label_visibility = "hidden",key="slide_07")
    
df = make_df(N_1, N_2, mu_1, mu_2, sigma_1, sigma_2)

fig00 = make_fig([ss.color_a, ss.color_b])

fig00.add_vline(x=ss.decision_thld)

df_perf_metrics = get_performance_metrics(df = df, thld = ss.decision_thld)

with col_a2:
    col1, col2, _ = st.columns((0.8, 0.5, 0.2))
    col1.subheader("Visualize distribution of score")
    col2.page_link("st_page_00.py", label="LINK : Summary with context and explanations", icon = "💜")
    st.plotly_chart(fig00, use_container_width=True)


#-----------------------
# 2nd line 
col_a2, col_space02, col_b2, col_space021 = st.columns([0.20, 0.05, 0.80, 0.10])

with col_a2:
    st.subheader("")
    c1, c2, _ = st.columns([0.20, 0.20, 0.40])
    st.button("Confirm")
    with c1:
        ss.color_a = st.color_picker("Class A Color", ss.color_a) 
    with c2:
        ss.color_b = st.color_picker("Class B Color", ss.color_b)
        
with col_b2:
    col1, col2 = st.columns([0.4, 0.6])
    col1.subheader("Threshold free metrics")
    col2.subheader("Threshold dependent metrics")
    # st.subheader("Performance metrics")
    col1, col2, col3, col4, col5, = st.columns([0.2, 0.2, 0.2, 0.2, 0.2])
    col1.metric("ROC-AUC", df_perf_metrics["ROC-AUC"], border=True)
    col2.metric("Average Precision", df_perf_metrics["Average Precision"], border=True)
    col3.metric("Precision", df_perf_metrics['Precision'], border=True)
    col4.metric("Recall", df_perf_metrics['Recall'], border=True)
    col5.metric("Accuracy", df_perf_metrics['Accuracy'], border=True)








# slider_default_values = [1000, 0.2, 0.2, 1000, 0.8, 0.2, 0.5]

# if st.button("Foo"):
#     print("haha")
#     ss.clear()
#     st.rerun()
#     ss["slide_01"] = slider_default_values[0]
#     ss["slide_02"] = slider_default_values[1]
#     ss["slide_03"] = slider_default_values[2]
#     ss["slide_04"] = slider_default_values[3]
#     ss["slide_05"] = slider_default_values[4]
#     ss["slide_06"] = slider_default_values[5]
#     ss["slide_07"] = slider_default_values[6]
#     st.rerun()

# # st.text(ss["slide_01"])

