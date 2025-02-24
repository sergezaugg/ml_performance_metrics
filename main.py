#--------------
#
#
#--------------

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.metrics import roc_auc_score

def make_one_class_data(N, beta_a, beta_b, class_name):
    vals = np.random.beta(a = beta_a, b = beta_b, size = N)
    df = pd.DataFrame({"proba_score":vals})
    df['class'] = class_name
    return(df)



# reparamtrize 
# from https://www.johndcook.com/blog/2021/04/07/beta-given-mean-variance/
mu = np.random.uniform()
sigma2 = np.random.uniform(0, mu*(1-mu))
a = mu*(mu*(1-mu)/sigma2 - 1)
b = a*(1-mu)/mu


N_1 = 5000
beta_a_1 = a
beta_b_1 = b
class_name_1 = "Class A"

aaa = make_one_class_data(N_1, beta_a_1, beta_b_1, class_name_1)

[aaa['proba_score'].mean(), mu]
[aaa['proba_score'].var(), sigma2]





N_2 = 500
beta_a_2 = 3.0
beta_b_2 = 1.0
class_name_2 = "Class B"

df = pd.concat([
    make_one_class_data(N_1, beta_a_1, beta_b_1, class_name_1),
    make_one_class_data(N_2, beta_a_2, beta_b_2, class_name_2)
    ])

df['jitter'] = np.random.uniform(size=N_1+N_2)

fig00 = px.scatter(
    data_frame = df,
    x = 'proba_score',
    y = 'jitter',
    color = 'class',
    color_discrete_sequence=['#ee33ff', '#33aaff'],
    template='plotly_dark',
    width = 900,
    height = 400,
    )

_ = fig00.update_xaxes(showline = True, linecolor = 'white', linewidth = 1, row = 1, col = 1, mirror = True)
_ = fig00.update_yaxes(showline = True, linecolor = 'white', linewidth = 1, row = 1, col = 1, mirror = True)
_ = fig00.update_traces(marker=dict(size=4))
_ = fig00.update_layout(xaxis=dict(showgrid=False),yaxis=dict(showgrid=False))
fig00.show()


roc_auc_score(y_true = df['class'], y_score = df['proba_score'])









