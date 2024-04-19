import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelBinarizer
import streamlit as st
from streamlit_option_menu import option_menu
import re

st.set_page_config(layout="wide")
st.markdown("<h1 style='text-align: center; color: #CC5533;'>Industrial Copper Modeling</h1>", unsafe_allow_html=True)
st.markdown("<hr style='border: 2px solid #CC5533;'>", unsafe_allow_html=True)

select = option_menu(
    menu_title = None,
    options = ["Home","Data Prediction"],
    icons =["house","bar-chart"],
    default_index=0,
    orientation="horizontal",
    styles={"container": {"padding": "0!important", "background-color": "#38e691","size":"cover", "width": "200"},
        "icon": {"color": "black", "font-size": "25px"},
            
        "nav-link": {"font-size": "25px", "text-align": "center", "margin": "-2px", "--hover-color": "#00A699"},
        "nav-link-selected": {"background-color": "#eba75f",  "font-family": "YourFontFamily"}})

if select == "Home":
    st.write("")
    st.write("")
    col1,col2,col3= st.columns([4,4,3])
    with col1:
        st.markdown("### Introduction :")
        st.markdown("**This project aims to develop two machine learning models for the copper industry to address the challenges of predicting selling price and lead classification. Manual predictions can be time-consuming and may not result in optimal pricing decisions or accurately capture leads. The models will utilize advanced techniques such as data normalization, outlier detection and handling, handling data in the wrong format, identifying the distribution of features, and leveraging tree-based models, specifically the decision tree algorithm, to predict the selling price and leads accurately.**")

    with col2:
        st.markdown("### Regression model details :")
        st.markdown("**The copper industry deals with less complex data related to sales and pricing. However, this data may suffer from issues such as skewness and noisy data, which can affect the accuracy of manual predictions. Dealing with these challenges manually can be time-consuming and may not result in optimal pricing decisions. A machine learning regression model can address these issues by utilizing advanced techniques such as data normalization, outlier detection and handling, handling data in wrong format, identifying the distribution of features, and leveraging tree-based models, specifically the decision tree algorithm.**")

    with col3:
        st.markdown("### Classification model details :")
        st.markdown("**Another area where the copper industry faces challenges is in capturing the leads. A lead classification model is a system for evaluating and classifying leads based on how likely they are to become a customer. You can use the STATUS variable with WON being considered as Success and LOST being considered as Failure and remove data points other than WON, LOST STATUS values.**")

    st.write("")
    st.write("")
    col4,col5,col6= st.columns([5,2,3])
    with col4:
        st.markdown("### The solution includes the following steps :")
        st.markdown("- **Exploring skewness and outliers in the dataset.**")
        st.markdown("- **Transforming the data into a suitable format and performing any necessary cleaning and pre-processing steps.**")
        st.markdown("- **Developing a machine learning regression model which predicts the continuous variable 'Selling_Price' using the decision tree regressor.**")
        st.markdown("- **Developing a machine learning classification model which predicts the Status: WON or LOST using the decision tree classifier.**")
        st.markdown("- **Creating a Streamlit page where you can insert each column value and get the Selling_Price predicted value or Status (Won/Lost).**")
        
    with col5:
        st.markdown("###  Requirements :")
        st.markdown("- **Python**")
        st.markdown("- **Pandas**")
        st.markdown("- **Streamlit**")
        st.markdown("- **Data Preprocessing**")
        st.markdown("- **Exploratory Data Analysis (EDA)**")

    with col6:
        st.markdown("###  Getting Started :")
        st.markdown("- **Clone the repository.**")
        st.markdown("- **Install the required libraries.**")
        st.markdown("- **Run the Streamlit app using the command: streamlit run app.py.**")
        st.markdown("- **Enter the values for each column to get the Selling_Price predicted value or Status (Won/Lost).**")
            
if select == "Data Prediction":
    
    tab1, tab2 = st.tabs(["**PREDICT SELLING PRICE**", "**PREDICT STATUS**"])
    with tab1:    
        
            # Define the possible values for the dropdown menus
            status_options = ['Won', 'Draft', 'To be approved', 'Lost', 'Not lost for AM', 'Wonderful', 'Revised', 'Offered', 'Offerable']
            item_type_options = ['W', 'WI', 'S', 'Others', 'PL', 'IPL', 'SLAWR']
            country_options = [28., 25., 30., 32., 38., 78., 27., 77., 113., 79., 26., 39., 40., 84., 80., 107., 89.]
            application_options = [10., 41., 28., 59., 15., 4., 38., 56., 42., 26., 27., 19., 20., 66., 29., 22., 40., 25., 67., 79., 3., 99., 2., 5., 39., 69., 70., 65., 58., 68.]
            product=['611112', '611728', '628112', '628117', '628377', '640400', '640405', '640665', 
                         '611993', '929423819', '1282007633', '1332077137', '164141591', '164336407', 
                         '164337175', '1665572032', '1665572374', '1665584320', '1665584642', '1665584662', 
                         '1668701376', '1668701698', '1668701718', '1668701725', '1670798778', '1671863738', 
                         '1671876026', '1690738206', '1690738219', '1693867550', '1693867563', '1721130331', '1722207579']
    
            # Define the widgets for user input
            with st.form("my_form"):
                col1,col2,col3=st.columns([5,2,5])
                with col1:
                    st.write(' ')
                    status = st.selectbox("Status", status_options,key=1)
                    item_type = st.selectbox("Item Type", item_type_options,key=2)
                    country = st.selectbox("Country", sorted(country_options),key=3)
                    application = st.selectbox("Application", sorted(application_options),key=4)
                    product_ref = st.selectbox("Product Reference", product,key=5)
                with col3:               
                    st.write( f'<h5 style="color:rgb(0, 153, 153,0.4);">NOTE: Min & Max given for reference, you can enter any value</h5>', unsafe_allow_html=True )
                    quantity_tons = st.text_input("Enter Quantity Tons (Min:611728 & Max:1722207579)")
                    thickness = st.text_input("Enter thickness (Min:0.18 & Max:400)")
                    width = st.text_input("Enter width (Min:1, Max:2990)")
                    customer = st.text_input("customer ID (Min:12458, Max:30408185)")
                    submit_button = st.form_submit_button(label="PREDICT SELLING PRICE")
                    st.markdown("""
                        <style>
                        div.stButton > button:first-child {
                            background-color: lightblue;
                            color: white;
                            width: 100%;
                        }
                        </style>
                    """, unsafe_allow_html=True)
        
                flag=0 
                pattern = "^(?:\d+|\d*\.\d+)$"
                for i in [quantity_tons,thickness,width,customer]:             
                    if re.match(pattern, i):
                        pass
                    else:                    
                        flag=1  
                        break
                
            if submit_button and flag==1:
                col1,col2,col3 = st.columns(3)
                with col2:
                    if len(i)==0:
                        st.warning("Please enter a valid number, space not allowed", icon="⚠️")
                    else:
                        st.warning("You have entered an invalid value: ",i ,icon="⚠️")  
                 
            if submit_button and flag==0:
                
                import pickle
                with open(r"C:/Users/Admin/Downloads/GUVI_Python/Copper_Modeling/model.pkl", 'rb') as file:
                    loaded_model = pickle.load(file)
                with open(r"C:/Users/Admin/Downloads/GUVI_Python/Copper_Modeling/scaler.pkl", 'rb') as f:
                    scaler_loaded = pickle.load(f)
    
                with open(r"C:/Users/Admin/Downloads/GUVI_Python/Copper_Modeling/t.pkl", 'rb') as f:
                    t_loaded = pickle.load(f)
    
                with open(r"C:/Users/Admin/Downloads/GUVI_Python/Copper_Modeling/s.pkl", 'rb') as f:
                    s_loaded = pickle.load(f)
    
                new_sample= np.array([[np.log(float(quantity_tons)),application,np.log(float(thickness)),float(width),country,float(customer),int(product_ref),item_type,status]])
                new_sample_ohe = t_loaded.transform(new_sample[:, [7]]).toarray()
                new_sample_be = s_loaded.transform(new_sample[:, [8]]).toarray()
                new_sample = np.concatenate((new_sample[:, [0,1,2, 3, 4, 5, 6,]], new_sample_ohe, new_sample_be), axis=1)
                new_sample1 = scaler_loaded.transform(new_sample)
                new_pred = loaded_model.predict(new_sample1)[0]
                col1,col2,col3 = st.columns([4,8,3])
                with col2:
                    st.write('## :green[Predicted selling price:] ', np.exp(new_pred))
                
    with tab2: 
        
            with st.form("my_form1"):
                col1,col2,col3=st.columns([5,1,5])
                with col1:
                    cquantity_tons = st.text_input("Enter Quantity Tons (Min:611728 & Max:1722207579)")
                    cthickness = st.text_input("Enter thickness (Min:0.18 & Max:400)")
                    cwidth = st.text_input("Enter width (Min:1, Max:2990)")
                    ccustomer = st.text_input("customer ID (Min:12458, Max:30408185)")
                    cselling = st.text_input("Selling Price (Min:1, Max:100001015)") 
                  
                with col3:    
                    st.write(' ')
                    citem_type = st.selectbox("Item Type", item_type_options,key=21)
                    ccountry = st.selectbox("Country", sorted(country_options),key=31)
                    capplication = st.selectbox("Application", sorted(application_options),key=41)  
                    cproduct_ref = st.selectbox("Product Reference", product,key=51)           
                    csubmit_button = st.form_submit_button(label="PREDICT STATUS")
        
                cflag=0 
                pattern = "^(?:\d+|\d*\.\d+)$"
                for k in [cquantity_tons,cthickness,cwidth,ccustomer,cselling]:             
                    if re.match(pattern, k):
                        pass
                    else:                    
                        cflag=1  
                        break
                
            if csubmit_button and cflag==1:
                col1,col2,col3 = st.columns(3)
                with col2:
                    if len(k)==0:
                        st.warning("Please enter a valid number, space not allowed", icon="⚠️")
                    else:
                        st.warning("You have entered an invalid value: ",k ,icon="⚠️")  
                 
            if csubmit_button and cflag==0:
                import pickle
                with open(r"C:/Users/Admin/Downloads/GUVI_Python/Copper_Modeling/cmodel.pkl", 'rb') as file:
                    cloaded_model = pickle.load(file)
    
                with open(r'C:/Users/Admin/Downloads/GUVI_Python/Copper_Modeling/cscaler.pkl', 'rb') as f:
                    cscaler_loaded = pickle.load(f)
    
                with open(r"C:/Users/Admin/Downloads/GUVI_Python/Copper_Modeling/ct.pkl", 'rb') as f:
                    ct_loaded = pickle.load(f)
    
                # Predict the status for a new sample
                # 'quantity tons_log', 'selling_price_log','application', 'thickness_log', 'width','country','customer','product_ref']].values, X_ohe
                new_sample = np.array([[np.log(float(cquantity_tons)), np.log(float(cselling)), capplication, np.log(float(cthickness)),float(cwidth),ccountry,int(ccustomer),int(product_ref),citem_type]])
                new_sample_ohe = ct_loaded.transform(new_sample[:, [8]]).toarray()
                new_sample = np.concatenate((new_sample[:, [1,0,2, 3, 4, 5, 6,7]], new_sample_ohe), axis=1)
                new_sample = cscaler_loaded.transform(new_sample)
                new_pred = cloaded_model.predict(new_sample)
                col1,col2,col3 =  st.columns([5,5,3])
                with col2:
                    if new_pred==1:
                        st.write('## :green[The Status is Won ✅ ] ')
                    else:
                        st.write('## :red[The status is Lost ❌ ] ')