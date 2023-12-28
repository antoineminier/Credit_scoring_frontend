import streamlit as st
import json
import requests
import pandas as pd
import numpy as np
import math
import shap
import plotly.express as px
from plotly.subplots import make_subplots
from streamlit_shap import st_shap

st.set_page_config(layout="wide")

st.markdown("<h1 style='text-align: center;'>LOAN APPLICATION DASHBOARD</h1>", unsafe_allow_html=True)
st.text("")

number_of_features = int(requests.get(url='https://credit-scoring-backend.onrender.com/get_number_of_features').json())

col1, col2, col3, col4 = st.columns(4)
id = col1.text_input('Enter client ID')
if id:
    res = json.loads(requests.get(url=f'https://credit-scoring-backend.onrender.com/predict/{id}').text)
    if isinstance(res, str):
        st.subheader(res)
    else:
        if res['probability']>res['threshold']:
            st.subheader('loan refused')
        elif res['probability']<=res['threshold']:
            st.subheader('loan granted')
        st.subheader(f"default probability : {round(res['probability']*100, 1)} %\nmust not be over {str(round(res['threshold']*100))} % for the loan to be granted")
        
        st.text("")
        show_impacts = st.checkbox('Show feature impacts')
        if show_impacts:
            explanation_dict = requests.get(url=f'https://credit-scoring-backend.onrender.com/explain/{id}').json()
            explanation = shap.Explanation(values=np.array(explanation_dict['values']), 
                                           base_values=explanation_dict['expected_value'], 
                                           data=explanation_dict['data'],
                                           feature_names=explanation_dict['feature_names'])
            
            st.markdown("To which n<sup>th</sup> most impactful feature on the default probability get the impact :", unsafe_allow_html=True)
            col1, col2 = st.columns([0.3, 0.7])
            n_features = col1.slider("To which nth most impactful feature get the impact on the default probability :",
                                     min_value=10, 
                                     max_value=number_of_features,
                                     step=1,
                                     label_visibility="collapsed")
            col1, col2 = st.columns([0.2, 0.8])
            with col2:
                st.header("Impact of the most impactful features on the default probability")
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
                descriptions = json.loads(requests.get(url=f'https://credit-scoring-backend.onrender.com/descriptions').text)
                abs_values = [abs(el) for el in explanation_dict['values']]
                for i in range(n_features):
                    index = abs_values.index(sorted(abs_values, reverse=True)[i])
                    st.write(f"{explanation_dict['feature_names'][index]} : {descriptions[explanation_dict['feature_names'][index]]}")
            st.write('')

            st.text("")
            st.markdown(f"<h2 style='text-align: center;'>Comparison with the other loan applicants</h2>", unsafe_allow_html=True)
            st.text("")
            st.markdown("To which n<sup>th</sup> most impactful feature get information on :", unsafe_allow_html=True)
            col1, col2, col3, col4 = st.columns(4)
            n_features = col1.slider("To which nth most impactful feature get information on :", 
                                     min_value=0, 
                                     max_value=number_of_features, 
                                     value=3,
                                     step=3,
                                     label_visibility="collapsed")
            
            if n_features!=0:
                features = requests.get(url=f'https://credit-scoring-backend.onrender.com/compare/{id}').json()
                figures = []
                for i in range(n_features):
                    barchart_df = pd.DataFrame(data=features[i]['barchart_dict']['data'], 
                                               columns=features[i]['barchart_dict']['columns'])
                    if features[i]['feature_impact']>=0:
                        feature_impact = f"+{round(features[i]['feature_impact'], 2)}"
                    else:
                        feature_impact = f"{round(features[i]['feature_impact'], 2)}"
                    if list(barchart_df.columns)==['category', 'loan_status', 'count']:
                        color_discrete_map = {'granted': '#00cc96', 'refused': '#EF553B'}
                        title = f"feature : {features[i]['feature']}<br><sup>client value: {features[i]['client_value']}  —  impact on default probability: {feature_impact}</sup>"
                        barchart = px.bar(barchart_df, 
                                          x='category', 
                                          y='count', 
                                          barmode='group',
                                          color='loan_status',
                                          color_discrete_map=color_discrete_map, 
                                          title=title)
                    else:
                        title = f"feature : {features[i]['feature']}<br><sup>impact on default probability: {feature_impact}</sup>"
                        barchart = px.bar(barchart_df, 
                                          x='value displayed', 
                                          y=features[i]['feature'], 
                                          barmode='group',
                                          title=title)
                    barchart.update_layout(xaxis_title='', 
                                           yaxis_title='',
                                           title_font_size=20,
                                           title_x=0.5,
                                           title_xanchor="center", 
                                           legend=dict(
                                               yanchor="top",
                                               y=0.99,
                                               xanchor="left",
                                               x=0.8
                                           ))
                    figures.append(barchart)

                n_rows = 0
                charts_displayed = 0
                while n_rows < (n_features // 3) :
                    col1, col2, col3 = st.columns(3)
                    n_rows += 1
                    with col1:
                        st.plotly_chart(figures[charts_displayed], use_container_width=True)
                        charts_displayed += 1
                    with col2:
                        st.plotly_chart(figures[charts_displayed], use_container_width=True) 
                        charts_displayed += 1
                    with col3:
                        st.plotly_chart(figures[charts_displayed], use_container_width=True)
                        charts_displayed += 1
                
                if n_features - charts_displayed == 1 :
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.plotly_chart(figures[charts_displayed], use_container_width=True)
                if n_features - charts_displayed ==2 :
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.plotly_chart(figures[charts_displayed], use_container_width=True)
                        charts_displayed += 1
                    with col2:
                        st.plotly_chart(figures[charts_displayed], use_container_width=True)