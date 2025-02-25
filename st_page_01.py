
#--------------------             
# Author : Serge Zaugg
# Description : 
#--------------------

import numpy as np
import streamlit as st
import plotly.express as px
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.metrics import roc_auc_score, average_precision_score
from streamlit import session_state as ss


if 'color_a' not in ss:
    ss.color_a = '#ee33ff'
if 'color_b' not in ss:
    ss.color_b = '#33aaff'



#-----------------------
# 1 define

@st.cache_data
def make_one_class_data(N, mu, sigma, class_name):
    sigma2 = sigma**2
    var_max = mu*(1-mu)
    sdt_max = np.sqrt(var_max)
    assert (sigma2 < var_max) , "sigma must be < " + str(sdt_max)
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
        make_one_class_data(N_1, mu_1, sigma_1, class_name_1),
        make_one_class_data(N_2, mu_2, sigma_2, class_name_2)
        ])
    df['jitter'] = np.random.uniform(size=N_1+N_2)
    return(df)


#-----------------------
# 








#-----------------------
# 1 st line 
col_a1, col_space01, col_a2, = st.columns([0.20, 0.05, 0.80])

with col_a1: 
    st.subheader("Set params")

    col_x1, col_x2, = st.columns([0.50, 0.50])

    with col_x1: 
        st.text('Class A')
        N_1     = st.slider("N",     min_value=  10,   max_value=5000,  value=1000, label_visibility = "visible", key="slide_01")
        mu_1    = st.slider("mean",  min_value = 0.0,  max_value=0.99,  value=0.10,  label_visibility = "visible",key="slide_02")
        sigma_1 = st.slider("stdev", min_value = 0.0,   max_value=0.99, value=0.20,  label_visibility = "visible",key="slide_03")
    
    with col_x2: 
        st.text('Class b')
        N_2     = st.slider("N",     min_value=  10,   max_value=5000,  value=1000,  label_visibility = "visible",key="slide_04")
        mu_2    = st.slider("mean",  min_value= 0.0,   max_value=0.99,  value=0.90,  label_visibility = "visible",key="slide_05")
        sigma_2 = st.slider("stdev", min_value= 0.0,   max_value=0.99,  value=0.20,  label_visibility = "visible",key="slide_06")
    
# N_1 = 5000
# mu_1 = 0.5
# sigma_1 = 0.02
# N_2 = 500
# mu_2 = 0.8
# sigma_2 = 0.1

df = make_df(N_1, N_2, mu_1, mu_2, sigma_1, sigma_2)

rauc_val =           roc_auc_score(y_true = df['class'], y_score = df['proba_score'])
avep_val = average_precision_score(y_true = df['class'], y_score = df['proba_score'], pos_label='Class B')

fig00 = px.scatter(
    data_frame = df,
    x = 'proba_score',
    y = 'jitter',
    color = 'class',
    color_discrete_sequence=[ss.color_a, ss.color_b],
    template='plotly_dark',
    width = 900,
    height = 500,
    )

_ = fig00.update_xaxes(showline = True, linecolor = 'white', linewidth = 2, row = 1, col = 1, mirror = True)
_ = fig00.update_yaxes(showline = True, linecolor = 'white', linewidth = 2, row = 1, col = 1, mirror = True)
_ = fig00.update_traces(marker=dict(size=4))
_ = fig00.update_layout(xaxis=dict(showgrid=False), yaxis=dict(showgrid=False))
_ = fig00.update_layout(xaxis_range=[-0.01, +1.01])
_ = fig00.update_layout(paper_bgcolor="#112233",)
# fig00.show()

with col_a2:
    st.subheader("Viz")
    st.plotly_chart(fig00, use_container_width=True)

        
   

   
   

#-----------------------
# 2 nd line 
col_a2, col_b2, = st.columns([0.20, 0.80])

with col_a2:
    st.subheader("Scores ")
    st.text("ROC-AUC          : " + "{:.3f}".format(rauc_val.round(4))  )
    st.text("Average Precision: " + "{:.3f}".format(avep_val.round(4))    )
   
with col_b2:
    c1, c2, = st.columns([0.20, 0.80])
    with c1:
        ss.color_a = st.color_picker("Class A Color", '#ee33ff')
    with c2:
        ss.color_b = st.color_picker("Class B Color", '#33aaff')
  

