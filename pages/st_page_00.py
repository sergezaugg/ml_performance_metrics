#--------------------             
# Author : Serge Zaugg
# Description : Some concise background info 
#--------------------

import streamlit as st

col_aa, col_bb, = st.columns([1.00, 0.20])

with col_aa: 

    with st.container(border=True, key='conta_02'):
        st.markdown(
        '''    
        :violet[**SUMMARY**]

        In applied ML, metrics such as the Precision and Recall are used to evaluate classification models, 
        but their practical meaning can be difficult to grasp. This app simulates model predictions 
        and allows users to adjust parameters and visualize how they affect these metrics.
        The same metrics are used to evaluate medical diagnostic tests, making this app a useful 
        resource for clinicians to understand and interpret these concepts as well.
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
        ''')

        st.markdown(
        '''
        :violet[**DETAILS**]  
        
        * Beta distribution is used to simulate the scores 

        ''')
