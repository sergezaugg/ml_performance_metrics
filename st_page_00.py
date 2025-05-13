#--------------------             
# Author : Serge Zaugg
# Description : Some concise background info 
#--------------------

import streamlit as st

col_aa, col_bb, = st.columns([0.50, 0.50])

with col_aa: 

    with st.container(border=True, key='conta_01'):
        st.title(":violet[Explain and understand machine learning performance metrics]") 

    with st.container(border=True, key='conta_02'):
        st.markdown(
        '''    
        :violet[**SUMMARY**]
        
        In applied machine learning we must measure the predictive performance of models.
        Many performance metric exist but their practical interpretation is not always trivial.
        Here, I present several important metrics of predictive performance.
        The tool simulates the predicted score returned by a model for binary classification.
        The score's ability to separate the classes and the class balance can be manually adjusted to assess many relevant scenarios.
        The score is graphically shown and the decision threshold can be manually adjusted.
        In epidemiology we typically define healthy subjects as **Negatives** and subjects with disease as **Positives**.
        ''')
    
        st.markdown(
        '''  
        :violet[**APPLICATIONS**]

        With this dashboard, the behavior of performance metrics can be intuitively explained.
        For example:
        * Precision and Recall are complementary
        * Specificity and Sensitivity are complementary
        * Model with a moderate ROC-AUC can achieve high Precision at the cost of a low Recall
        * A decision threshold of 0.5 is not always a good choice
        * ROC-AUC is unsensitive to class balance
        * Average Precision is sensitive to class balance
        * Accuracy is misleading with unbalanced classes
        * Etc.
        ''')

        st.markdown(
        '''
        :violet[**DETAILS**]  
        
        * Beta distribution is used to simulate the scores 

        ''')
