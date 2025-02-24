#--------------
#
#
#--------------



import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px



def make_one_class_data(N, beta_a, beta_b, class_name):
    vals = np.random.beta(a = beta_a, b = beta_b, size = N)
    df = pd.DataFrame({"proba_score":vals})
    df['class'] = class_name
    return(df)



N_1 = 100
beta_a_1 = 2.0
beta_b_1 = 4.0
class_name_1 = "Class A"


N_2 = 100
beta_a_2 = 4.0
beta_b_2 = 2.0
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
    width = 700,
    height = 400,
    )

_ = fig00.update_xaxes(showline = True, linecolor = 'white', linewidth = 1, row = 1, col = 1, mirror = True)
_ = fig00.update_yaxes(showline = True, linecolor = 'white', linewidth = 1, row = 1, col = 1, mirror = True)
# _ = fig00.update_layout(paper_bgcolor="#112233",)
fig00.update_traces(marker=dict(size=5))

fig00.show()

