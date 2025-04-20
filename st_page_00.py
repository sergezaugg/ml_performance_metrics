#--------------------             
# Author : Serge Zaugg
# Description : Some concise background info 
#--------------------

import streamlit as st

col_aa, col_bb, = st.columns([0.60, 0.40])

with col_aa: 

    with st.container(border=True, key='conta_01'):
        st.title(":violet[Explain and understand machine learning performance metrics]") 
        st.subheader(":violet[Applied Machine Learning  ---  ML Tutorials  ---  Supervised Classification]") 

    with st.container(border=True, key='conta_02'):
        st.markdown(
        '''    
        :violet[**SUMMARY**]
        
        In applied machine learning projects we must measure the predictive performance of models.
        Many performance metric exist and data scientists have a clear understanding of what they mean.
        However, the interpretation of these metrics is not always clear for other domain specialists.
        Here, I present several usual metrics of predictive performance.
        The tool simulates the predicted score returned by a model for two classes.
        The score can be manually tuned to assess many relevant scenarios (balanced vs unbalanced classes, good vs bad separation).
        The score is graphically shown and a decision threshold can be manually adjusted.
        (Obviously, there is much more than just metrics for the assessment of ML models.
        But this is out-of-scope for this dashboard.)
        ''')
    
    with st.container(border=True, key='conta_03'):
        st.markdown(
        '''  
        :violet[**APPLICATIONS**]

        With this dashboard, the behavior of performance metrics can be intuitively explained:
        * Precision and Recall are complementary
        * Model with a moderate ROC-AUC can still achieve high precision at the cost of a low recall
        * A decision threshold of 0.5 is not always the best choice
        * ROC-AUC is unsensitive to class balance
        * Average Precision is sensitive to class balance
        * Accuracy can be misleading with unbalanced classes
        ''')





# """
# True positive rate (TPR) = recall = sensitivity 
# True negative rate (TNR) = specificity
# """ 


   