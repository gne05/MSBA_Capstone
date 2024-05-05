#Import Libraries
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import hydralit_components as hc
import os
from io import StringIO
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
import math




# Set Page Icon,Title, and Layout
st.set_page_config(layout="wide",  page_title = "A Predictive Analytics and Customer Segmentation Tool Designed for Simly")

#Set up the navigation bar
menu_data = [
{'label':"Home Page", 'icon': "bi bi-house"},
{'label':"Segmentation", 'icon': "fas fa-object-group"},
{'label':"Classification", 'icon': "fas fa-toggle-on"},
{'label':"Regression", 'icon': "fas fa-chart-line"}]


over_theme = {'txc_inactive': 'white','menu_background':'#ff7f00', 'option_active':'white'}

menu_id = hc.nav_bar(
    menu_definition=menu_data,
    override_theme=over_theme,
    hide_streamlit_markers=True,
    sticky_nav=True,
    sticky_mode='sticky'
    )

# Introduction Page
if menu_id == "Home Page":
    st.markdown(" ")
    st.markdown(" ")
    st.markdown("<h1 style='text-align: center; color: black;'>Simly Analytical Tool<i class='bi bi-heart-fill' style='color: red;'></i> </h2>", unsafe_allow_html=True)
    st.markdown("<hr style='border-top: 3px solid black;'>", unsafe_allow_html=True)
    st.subheader("Welcome to this analytical, data-driven tool that will help Simly in future marketing campaigns")
    st.markdown(" ")
    col1, col2 = st.columns(2)
    with col1: st.image('simly logo.jpg', width=500,use_column_width=False)

    with col2:
        st.markdown(" ")
        st.markdown(" ")
        st.markdown(" ")
        st.markdown(" ")
        st.markdown(" ")
        st.markdown(" ")
        st.markdown(" ")

        st.image('airplane.jpg', width=500,use_column_width=False)

    st.markdown(" ")
    st.markdown(" ")

    st.subheader("Like many startup companies, Simly invests significantly in marketing and advertising campaigns to build a sizable and loyal customer base. Consequently, over the past year, Simly experienced financial losses attributed to the substantial amounts spent on these initiatives.")
    st.markdown(" ")
    st.markdown(" ")
    st.subheader("This tool will facilitate the segmentation of Simly's customer base, enabling a more nuanced understanding of customer behavior and preferences. Through the creation of distinct segments, Simly can tailor marketing strategies, improve customer targeting, and enhance overall engagement")

# Segmentation Page
if menu_id == "Segmentation":

    st.markdown(" ")
    st.markdown(" ")
    st.markdown("<h1 style='text-align: center; color: black;'>Segmentation Tool<i class='bi bi-heart-fill' style='color: red;'></i> </h2>", unsafe_allow_html=True)

    st.markdown("<hr style='border-top: 3px solid black;'>", unsafe_allow_html=True)

    st.subheader("This segmentation tool will cluster any of Simly's customers into 1 of 4 clusters based on their RFM Scores. However, on an overall basis the data on which the deployed segmentation model was trained on, looks as per the below on a 3D plot")
    st.markdown(" ")

    col1, col2 = st.columns(2)
    with col1: st.image('segmentation.png', width=500,use_column_width=False)

    with col2:
        st.markdown(" ")
        st.markdown(" ")
        st.markdown(" ")
        st.markdown(" ")
        st.markdown(" ")
        st.markdown(" ")
        st.markdown(" ")
        st.markdown(" ")

        # Define your text
        text = {
        0: 'Inactive Customer',
        1: 'Loyal but Low-Spender Customer',
        2: 'High-Spender Customer',
        3: 'Frequent Customer'
                }

# Display the justified text for each item
    for key, value in text.items():
            st.markdown(f"<p style='text-align: justify;'><b>{key}: '{value}'</b></p>", unsafe_allow_html=True)

    pickle_in = open('clustering.pkl', 'rb') 
    clustering = pickle.load(pickle_in)

    pickle_in = open('scaler.pkl', 'rb')
    scaler =  pickle.load(pickle_in) 

    def prediction(Frequency, Average_Monetary, Recency):   
 
        Frequency = np.array(Frequency).reshape(-1, 1)
        Average_Monetary = np.array(Average_Monetary).reshape(-1, 1)


        current_date = datetime.now().date()
        date_difference = (current_date - Recency).days
        date_difference = np.array(date_difference).reshape(-1, 1)


        # Scale the input features using loaded scaler
        X = np.concatenate((Frequency, Average_Monetary,date_difference), axis=1)
        X_scaled = scaler.transform(X)

    # Making predictions 
        prediction = clustering.predict(X_scaled)
     
        if prediction == 0:
            pred = 'Inactive'

        if prediction == 1:
            pred = 'Loyal/ Low-Spending' 
        if prediction == 2:
            pred = 'High-Spending'   
        if prediction ==3:
            pred = 'Frequent'
        return pred
      
  
    def main():       
      
    # following lines create boxes in which user can enter data required to make prediction 
        Frequency = st.number_input('Number of Transactions Done',  min_value =1, value=1, step=1)
        Average_Monetary = st.number_input('Average Monetary Value Spent by the Customer per Transaction',min_value=0.01, value=0.01, step=0.01) 
        Recency = st.date_input("Enter the Last Transaction Date:") 
        result =""
      
    # when 'Predict' is clicked, make the prediction and store it 
        if  st.button("Predict"): 
            result = prediction(Frequency, Average_Monetary, Recency) 
            st.success('Your cluster is {}'.format(result))
            print(clustering.labels_)
     
    if __name__=='__main__': 
        main()


if menu_id == "Classification":

    st.markdown(" ")
    st.markdown(" ")
    st.markdown("<h1 style='text-align: center; color: black;'>Churn Classification Tool<i class='bi bi-heart-fill' style='color: red;'></i> </h2>", unsafe_allow_html=True)

    st.markdown("<hr style='border-top: 3px solid black;'>", unsafe_allow_html=True)

    st.subheader("This classification tool will predict whether a Simly customer will churn based on a set of features that showed to have an importance during the model deployment and assesment phase. However, on an overall basis only 20% of Simly's customers have been considered to churn throughout the period of the data received as illustrated below")
    st.markdown(" ")
    st.markdown(" ")
    col1, col2, col3 = st.columns([1,2,1])

    with col2:
        st.image('graph.png', width=500,use_column_width=False)

    pickle_in = open('classifier3.pkl', 'rb') 
    classifier = pickle.load(pickle_in)

    # def preprocess_categorical(data):
    # # Get dummies for Card_Brand and Card_Funding
    #     data_encoded = pd.get_dummies(data, columns=['Card_Brand', 'Card_Funding'])
    #     return data_encoded

    def prediction(Acquired_Promo_Code, Frequency, Average_Monetary, Lives_Abroad, user_data):   
 
        if Acquired_Promo_Code == "Yes":
            Acquired_Promo_Code = 1
        else:
            Acquired_Promo_Code = 0

        if Lives_Abroad == "Yes":
            Lives_Abroad = 1
        else:
            Lives_Abroad = 0

        features = np.array([Acquired_Promo_Code, Frequency, Average_Monetary, Lives_Abroad] + list(user_data.iloc[0]))

        prediction = classifier.predict(features.reshape(1, -1))

        if prediction == 0:
            pred = 'Not Churned'

        if prediction == 1:
            pred = 'Churned' 
        
        return pred
      
  

    def main():       
      
    # following lines create boxes in which user can enter data required to make prediction 
        Frequency = st.number_input('Number of Transactions Done',  min_value =1, value=1, step=1)
        Average_Monetary = st.number_input('Average Monetary Value Spent by the Customer per Transaction',min_value=0.01, value=0.01, step=0.01) 
        Acquired_Promo_Code = st.selectbox('Was the User Acquired through a referral',("Yes","No"))
        Lives_Abroad = st.selectbox("Does the user live outside his country of residence",("Yes","No"))
        Card_Brand = st.selectbox("Select Card Brand of the User:", ("AmericanExpress", "DinnersClub", "Discover","JCB","MasterCard","UnionPay","Visa"))
        Card_Funding = st.selectbox("Select Card Funding of the User:", ("Credit", "Debit","Prepaid"))

        columns = [
        'Card Brand_AmericanExpress', 'Card Brand_DinersClub', 'Card Brand_Discover',
        'Card Brand_JCB', 'Card Brand_MasterCard', 'Card Brand_UnionPay', 'Card Brand_Visa',
        'Card Funding_credit', 'Card Funding_debit', 'Card Funding_prepaid'
        ]
        user_data = pd.DataFrame(0, index=[0], columns=columns)

        print(user_data)

    # Setting the correct columns based on user input
        card_brand_column = f'Card Brand_{Card_Brand}'
        card_funding_column = f'Card Funding_{Card_Funding.lower()}'

        user_data[card_brand_column] = 1
        user_data[card_funding_column] = 1


        result =""
      
    # when 'Predict' is clicked, make the prediction and store it 
        if  st.button("Predict"): 
            result = prediction(Acquired_Promo_Code, Frequency, Average_Monetary, Lives_Abroad, user_data) 
            st.success('The user is predicted to {}'.format(result))
     
    if __name__ =='__main__': 
        main()



if menu_id == "Regression":

    st.markdown(" ")
    st.markdown(" ")
    st.markdown("<h1 style='text-align: center; color: black;'>Regression Tool<i class='bi bi-heart-fill' style='color: red;'></i> </h2>", unsafe_allow_html=True)

    st.markdown("<hr style='border-top: 3px solid black;'>", unsafe_allow_html=True)

    st.subheader("This predictive tool is designed to calculate the anticipated timeframe required for a customer to yield a profitable return, taking into account Simly's average Customer Acquisition Cost (CAC) of $8. By leveraging this tool, Simly aims to optimize its CAC investment, thereby mitigating potential financial losses.")
    st.markdown(" ")
    st.markdown(" ")
    st.subheader("Based on the illustration below, it's evident that Simly's customers tend to generate profit swiftly, typically within a day of their initial transaction. Interestingly, customers residing abroad from their home country exhibit a slightly quicker turnaround time in achieving profitability.")
    
    st.markdown(" ")
    st.markdown(" ")
    st.markdown(" ")
    st.markdown(" ")

    col1, col2, col3 = st.columns([1,2,1])

    with col2:
        st.image('profit.png', width=500,use_column_width=False)

    pickle_in = open('regressor2.pkl', 'rb') 
    regressor = pickle.load(pickle_in)


    def prediction(Frequency, Lives_Abroad, user_data):   
 
        if Lives_Abroad == "Yes":
            Lives_Abroad = 1
        else:
            Lives_Abroad = 0

        features = np.array([Frequency, Lives_Abroad] + list(user_data.iloc[0]))

        prediction = regressor.predict(features.reshape(1, -1))
        
        return prediction
      
  
    def main():       
      
    # following lines create boxes in which user can enter data required to make prediction 
        Frequency = st.number_input('Number of Transactions Done',  min_value =1, value=1, step=1)
        Lives_Abroad = st.selectbox("Does the user live outside his country of residence",("Yes","No"))
        Card_Brand = st.selectbox("Select Card Brand of the User:", ("AmericanExpress", "DinnersClub", "Discover","JCB","MasterCard","UnionPay","Visa"))
        Card_Funding = st.selectbox("Select Card Funding of the User:", ("Credit", "Debit","Prepaid"))

        columns = [
        'Card Brand_AmericanExpress', 'Card Brand_DinersClub', 'Card Brand_Discover',
        'Card Brand_JCB', 'Card Brand_MasterCard', 'Card Brand_UnionPay', 'Card Brand_Visa',
        'Card Funding_credit', 'Card Funding_debit', 'Card Funding_prepaid'
        ]
        user_data = pd.DataFrame(0, index=[0], columns=columns)

        print(user_data)

    # Setting the correct columns based on user input
        card_brand_column = f'Card Brand_{Card_Brand}'
        card_funding_column = f'Card Funding_{Card_Funding.lower()}'

        user_data[card_brand_column] = 1
        user_data[card_funding_column] = 1


        result =""
      
    # when 'Predict' is clicked, make the prediction and store it 
        if  st.button("Predict"): 
            result = prediction(Frequency, Lives_Abroad, user_data) 
            st.success('The user yielded Simly a profit after {} day(s)'.format(math.ceil(result[0])))
     
    if __name__ =='__main__': 
        main()









 




   






