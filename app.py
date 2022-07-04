import streamlit as st
import pandas as pd
import numpy as np
import os
import requests
import joblib
from lime import lime_tabular
import streamlit.components.v1 as components
from load_css import local_css
import plotly.express as px
import plotly.figure_factory as ff
import seaborn as sns
import matplotlib.pyplot as plt
local_css("style.css")

# Build app
title_text = 'Dashboard Credit Scoring'
subheader_text = '''Etude de solvabilité du client'''

# Title
st.markdown(f"<h2 style='text-align: center;'><b>{title_text}</b></h2>", unsafe_allow_html=True)
st.markdown(f"<h5 style='text-align: center;'>{subheader_text}</h5>", unsafe_allow_html=True)
st.text("")

# Interpretability list
cwd = os.getcwd() # Get the current working directory
name = "interpretability_list.joblib"
interpretability_list = joblib.load(os.path.join(cwd, name))

# Side Bar
with st.sidebar:
    cwd = os.getcwd() # Get the current working directory
    streamlite_image = os.path.join(cwd, "streamlite_logo.png")
    sb.image(streamlite_image, width=300)
    boite_image = os.path.join(cwd, "boite_logo.png")
    sb.image(boite_image, width=300)

    # SELECTION DU CUSTOMER_ID
    customer_id_list = np.arange(len(interpretability_list))
    customer_id = st.selectbox('Please select the customer_ID to analyse :', customer_id_list)
    st.write('You selected:', customer_id)
    print("User selected the customer_id {}".format(customer_id))


# AFFICHAGE DU CLIENT
cwd = os.getcwd()
name = "X_test_32.pickle"
df = joblib.load(os.path.join(cwd, name))

# *********************************************************************************************************************
# PREDICTION
# using model from api
if customer_id != None :
    # Catch the model deployed in PythonAnyWhere
    url = 'https://rob128.pythonanywhere.com/api'
    r = requests.post(url=url, json={"customer_ID": str(customer_id)})
    response = r.json()
    # st.write(response["probabilite"])
    print("Probability {} \n Solvability {}".format(response["probabilite"], response["solvabilite"]))

    # PIE CHART SOLVABILITY
    # Stating graphical parameters
    COLOR_BR_r = ['#00CC96', '#EF553B'] #['dodgerblue', 'indianred']
    # adapting message wether client's pos or neg
    if response["solvabilite"] == 0 :
        subheader_text = '''Successful payment probability !'''
    else:
        subheader_text = '''**Failure payment probability.**'''
    st.markdown(f"<h5 style='text-align: center;'>{subheader_text}</h5>", unsafe_allow_html=True)
    # plotting pie plot for proba, finding good h x w was a bit tough
    y_val = [response["probabilite"]*100, 100 - response["probabilite"]*100]
    fig = px.pie(values=y_val, names=[0,1], color=[0,1], color_discrete_sequence=COLOR_BR_r, width=230, height=230)
    fig.update_layout(margin=dict(l=0, r=30, t=30, b=0))
    st.plotly_chart(fig)
    
    # Response
    if response["solvabilite"] == 0:
        # st.write("The customer is solvent, with a probability of : {}".format(response["probabilite"]))
        t = "<div> <span class='highlight green'> The customer is solvent </span></div>"
        st.markdown(t, unsafe_allow_html=True)
        st.write("\n")
        st.write("With a probability of : {}%".format(response["probabilite"]*100))
    else:
        t = "<div> <span class='highlight red'> The customer is not solvent </span></div>"
        st.markdown(t, unsafe_allow_html=True)
        st.write("\n")
        st.write("With a probability of : {}%".format(response["probabilite"]*100))
    
    # AFFICHAGE DES DONNES CLIENT
    subheader_text = '''Here are the customer data'''
    st.markdown(f"<h5 style='text-align: center;'>{subheader_text}</h5>", unsafe_allow_html=True)
    st.dataframe(df.iloc[customer_id])

    # INTERPRETABILITES
    if st.button("Explain Results"):
        with st.spinner('Calculating...'):
            html = interpretability_list[customer_id].as_html()
            components.html(html, height=800)

# PLOT DES VARIABLES DOMINANTES DU CLIENT
# st.subheader("Distribution of the 2 variables with most positive contribution to the loan agreement .")
# positive_contribution_index_list = []
# negative_contribution_index_list = []
# for i in range(len(interpretability_list[customer_id].as_map()[1])):
#     if interpretability_list[customer_id].as_map()[1][i][1] > 0:
#         positive_contribution_index_list.append(interpretability_list[customer_id].as_map()[1][i][0])
#     else:
#         negative_contribution_index_list.append(interpretability_list[customer_id].as_map()[1][i][0])
# positive_feature_list = list(df.iloc[:, positive_contribution_index_list].columns)
# negative_feature_list = list(df.iloc[:, negative_contribution_index_list].columns)
# # fig 1 
# fig = px.histogram(df, x=positive_feature_list[0], title='Distribution of {}'.format(positive_feature_list[0]))
# fig.update_layout(bargap=0.2)
# st.plotly_chart(fig, use_container_width=True)
# # fig 2
# fig = px.histogram(df, x=positive_feature_list[1], title='Distribution of {}'.format(positive_feature_list[1]))
# fig.update_layout(bargap=0.2)
# st.plotly_chart(fig, use_container_width=True)     
# st.subheader("Distribution of the 2 variables with most negative contribution to the loan agreement .")
# # fig 1 
# fig = px.histogram(df, x=negative_feature_list[0], title='Distribution of {}'.format(negative_feature_list[0]))
# fig.update_layout(bargap=0.2)
# st.plotly_chart(fig, use_container_width=True)
# # fig 2
# fig = px.histogram(df, x=negative_feature_list[1], title='Distribution of {}'.format(negative_feature_list[1]))
# fig.update_layout(bargap=0.2)
# st.plotly_chart(fig, use_container_width=True)

# *********************************************************************************************************************

st.subheader("Below you can situate customer by plotting distribution.")
feature_selected = st.selectbox('Select a feature to plot', df.columns)
st.write('You selected:', feature_selected)

# PLOTTING
fig, ax = plt.subplots()
sns.histplot(data=df, x=feature_selected)
value2highlight = df.iloc[customer_id][feature_selected]
x_list = [(abs(value2highlight - p.get_x())) for p in ax.patches]
for p in ax.patches :
    if abs(value2highlight - p.get_x()) == min(x_list):
        p.set_color('crimson')
ax.set_title("Distribution {}".format(feature_selected))
st.pyplot(fig)
