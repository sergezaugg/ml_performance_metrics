#--------------------             
# Author : Serge Zaugg
# Description : some info 
#--------------------

import streamlit as st

col_aa, col_bb, = st.columns([0.60, 0.40])

with col_aa: 
    st.title('Understand ML performance metrics')
    st.markdown(
    '''    
    :violet[**SUMMARY:**]
    In applied ML projects we must measure the predictive performance of models.
    Many performance metric exist and data scientists have a clear understanding of what they mean.
    However, the interpretation of these metrics is not always clear for other domain specialists.
    Here, I present several usual metrics to measure the predictive performance of supervised classification models.
    The tool simulates the predicted score returned by a model for two classes.
    The score can be manually tuned to assess many relevant scenarios (balanced vs unbalanced classes, good vs bad separation).
    The score is graphically shown and a decision threshold can be manually set (needed for some metrics)
    ''')

    st.page_link("st_page_01.py", label="LINK : Interactive dashboard", icon = "💜")
    
    st.markdown(
    '''    
    :violet[**SCENARIOS:**]
    The metric's behavior and also classic pitfall can be easily illustrated:
    * ROC-AUC is unsensitive to class balance
    * Average Precision is sensitive to class balance
    * Precision and Recall are complementary
    * Accuracy can be misleading with unbalanced classes
    * Model with a moderate ROC-AUC can still achieve high precision at the cost of a low recall 
    * A decision threshold of 0.5 is not always the best choice
    ''')

    st.markdown(
    '''    
    :violet[**ALSO:**]
    There is much more than just metrics for the assessment of ML models.
    But this is out-of-scope for this dashboard.
    ''')
   
    


    # 😊


    # st.page_link("https://github.com/sergezaugg/ml_performance_metrics", label="DasGithub", icon="🌎")


   