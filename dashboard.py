import streamlit as st
import json
import requests
import pandas as pd
import numpy as np
import math
import matplotlib
import shap
import plotly.express as px
from plotly.subplots import make_subplots
from streamlit_shap import st_shap

st.set_page_config(layout="wide")

st.title('Loan application dashboard')

number_of_features = int(requests.get(url='http://127.0.0.1:8000/get_number_of_features').json())

col1, col2, col3, col4 = st.columns(4)
id = col1.text_input('Enter client ID')
if id:
    res = json.loads(requests.get(url=f'http://127.0.0.1:8000/predict/{id}').text)
    if isinstance(res, str):
        st.subheader(res)
    else:
        if res['prediction']==1:
            st.subheader('loan refused')
            st.subheader(f"solvency score : {round(res['probability'], 2)}\nscore from 0 to 1 – Must be over 0.5 for the loan to be granted")
        elif res['prediction']==0:
            st.subheader('loan granted')
            st.subheader(f"solvency score : {round(res['probability'], 2)}\nscore from 0 to 1 – Must be over 0.5 for the loan to be granted")
        
        show_impacts = st.checkbox('Show feature impacts')
        if show_impacts:
            explanation_dict = requests.get(url=f'http://127.0.0.1:8000/explain/{id}').json()
            explanation = shap.Explanation(values=np.array(explanation_dict['values']), 
                                           base_values=explanation_dict['expected_value'], 
                                           data=explanation_dict['data'],
                                           feature_names=explanation_dict['feature_names'])
            col1, col2, col3, col4 = st.columns(4)
            n_features = col1.slider('To which nth most impactful feature get the impact on score :', 
                                     min_value=10, 
                                     max_value=number_of_features,
                                     step=1)
            st_shap(shap.plots.waterfall(explanation, max_display=n_features+1), width=1150)

            chart_explanation = st.checkbox('See chart explanation')
            if chart_explanation:
                st.write('E[f(X)] is the base value, i.e. the score the algorithm would return if it had no input data about this particular client (which score is the average score on all the training dataset).')
                st.write('f(x) is the score returned for the client being considered.')
                st.write("You can see on this chart the contribution of the client's data to the score, for each feature, starting from the base value E[f(X)] at the bottom.")
                st.write("Next to each feature's name is the corresponding client's data.")
                st.write('')

            see_descriptions = st.checkbox("See features' descriptions")
            if see_descriptions:
                descriptions = json.loads(requests.get(url=f'http://127.0.0.1:8000/descriptions').text)
                abs_values = [abs(el) for el in explanation_dict['values']]
                for i in range(n_features):
                    index = abs_values.index(sorted(abs_values, reverse=True)[i])
                    st.write(f"{explanation_dict['feature_names'][index]} : {descriptions[explanation_dict['feature_names'][index]]}")
            st.write('')

            col1, col2, col3, col4 = st.columns(4)
            n_features = col1.slider('To which nth most impactful feature get information on :', 
                                     min_value=0, 
                                     max_value=number_of_features, 
                                     value=3,
                                     step=3)
            if n_features > number_of_features:
                    n_features = number_of_features
            
            if n_features!=0:
                features = requests.get(url=f'http://127.0.0.1:8000/compare/{id}').json()
                figures = []
                subplot_titles = []
                for i in range(n_features):
                    barchart_df = pd.DataFrame(data=features[i]['barchart_dict']['data'], 
                                               columns=features[i]['barchart_dict']['columns'])
                    if list(barchart_df.columns)==['category', 'loan_status', 'count']:
                        color_discrete_map = {'granted': '#00cc96', 'refused': '#EF553B'}
                        figures.append(px.bar(barchart_df, 
                                              x='category', 
                                              y='count', 
                                              color='loan_status',
                                              color_discrete_map=color_discrete_map))
                        if features[i]['feature_impact']>=0:
                            feature_impact = f"+{round(features[i]['feature_impact'], 2)}"
                        else:
                            feature_impact = f"{round(features[i]['feature_impact'], 2)}"
                        title = f"{features[i]['feature']}  —  client value: {features[i]['client_value']}  —  impact on score: {feature_impact}"
                        subplot_titles.append(title)
                    else:
                        figures.append(px.bar(barchart_df, 
                                              x='value displayed', 
                                              y=features[i]['feature']))
                        if features[i]['feature_impact']>=0:
                            feature_impact = f"+{round(features[i]['feature_impact'], 2)}"
                        else:
                            feature_impact = f"{round(features[i]['feature_impact'], 2)}"
                        title = f"{features[i]['feature']}  —  impact on score: {feature_impact}"
                        subplot_titles.append(title)
                
                fig = make_subplots(rows=math.ceil(len(figures)/3), 
                                    cols=3, 
                                    subplot_titles=subplot_titles, 
                                    horizontal_spacing=0.1) 

                for i, figure in enumerate(figures):
                    if i % 3 == 0:
                        nth_col=1
                    elif (i-1) % 3 == 0:
                        nth_col=2
                    elif (i-2) % 3 == 0:
                        nth_col=3
                    for trace in range(len(figure["data"])):
                        fig.append_trace(figure["data"][trace], row=math.ceil((i+1)/3), col=nth_col)

                fig.update_layout(title_text=f"First {n_features} impactful features on the score", 
                                  height=600*math.ceil(len(figures)/3), 
                                  showlegend=False)

                st.plotly_chart(fig, use_container_width=True)