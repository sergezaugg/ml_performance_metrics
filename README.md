# Explain and understand machine learning performance metrics

**SUMMARY**

In applied machine learning projects we must measure the predictive performance of models.
Many performance metric exist and data scientists have a clear understanding of what they mean.
However, the interpretation of these metrics is not always clear for other domain specialists.
Here, I present several usual metrics of predictive performance.
The tool simulates the predicted score returned by a model for two classes.
The score can be manually tuned to assess many relevant scenarios (balanced vs unbalanced classes, good vs bad separation).
The score is graphically shown and a decision threshold can be manually adjusted.

**An interactive dashboard can be found here https://ml-performance-metrics.streamlit.app/**

**DEPENDENCIES**
* Developed under Python 3.12.8
* First make a venv, then:
* pip install -r requirements.txt

**USAGE**
* Clone the repo
* go to the repo's root dir
* To start the Streamlit dashboard do ```streamlit run stmain.py```
* To work with a script, open **main.py**. Several scenarios can be defined in the dict at the top of the script



