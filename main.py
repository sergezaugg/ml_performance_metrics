#--------------
#
#
#--------------

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.metrics import roc_auc_score, average_precision_score



# # reparamtrize 
# # from https://www.johndcook.com/blog/2021/04/07/beta-given-mean-variance/
# mu = np.random.uniform()
# sigma2 = np.random.uniform(0, mu*(1-mu))


# a = mu*(mu*(1-mu)/sigma2 - 1)
# b = a*(1-mu)/mu

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






N_1 = 5000
mu_1 = 0.5
sigma_1 = 0.02
class_name_1 = "Class A"

N_2 = 500
mu_2 = 0.8
sigma_2 = 0.1
class_name_2 = "Class B"

df = pd.concat([
    make_one_class_data(N_1, mu_1, sigma_1, class_name_1)
    ,
    make_one_class_data(N_2, mu_2, sigma_2, class_name_2)
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
_ = fig00.update_layout(xaxis_range=[-0.1,1.1])
fig00.show()


roc_auc_score(y_true = df['class'], y_score = df['proba_score'])

average_precision_score(y_true = df['class'], y_score = df['proba_score'], pos_label='Class A')







