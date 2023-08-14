
import pickle
from datetime import datetime, date
import sklearn
import pandas as pd
import base64
import streamlit as st
import warnings
warnings.filterwarnings('ignore')


def price_prediction(Brand, Model, CC, Age):
    pickle_file_path = "Data/Final_sale_price_model.pkl"
    # Load the pickle file
    with open(pickle_file_path, 'rb') as file:
        saved_models = pickle.load(file)
    # saved_models = pd.read_pickle("file.pkl")
    # Extract the model objects and scaler
    Ml_Model = saved_models['Model']
    x_train_cols = saved_models['training_columns']
    scaler = saved_models['scaler_Age_CC']
    BaseN_encode_Model = saved_models['Model_encoder']
    OneHotEncode_Brand = saved_models['Brand_encoding']

    test_data = pd.DataFrame({  'Brand': [Brand],
                            'Model_new': [Model] , 
                            'CC': [CC], 
                            'Age':[Age],
                            'Sale_Price': 0})
    test_data = OneHotEncode_Brand.transform(test_data)
    data_encoded1 = BaseN_encode_Model.transform(test_data['Model_new'])
    test_data = pd.concat([test_data, data_encoded1], axis=1)
    
    test_data[['Age','CC']] = scaler.transform(test_data[['Age','CC']])
    test_data = test_data[x_train_cols]
    test_data = test_data.drop('Sale_Price', axis=1)
    sale_price = Ml_Model.predict(test_data)
    return sale_price[0]

def Login():     
    data = pd.read_csv("Data/data_for_streamlit.csv")
    data['CC']  = data['CC'].astype('int')

    Brand_list = data['Brand'].unique().tolist()
    Brand_list.insert(0, 'Select here')
    # Brand_list = ["select here",'TVS', 'HERO', 'ROYAL ENFIELD', 'HONDA', 'BAJAJ', 'SUZUKI',
    #        'YAMAHA', 'HERO HONDA']
    Model_list = {}
    for _, row in data.iterrows():
        if row['Brand'] not in Model_list:
            Model_list[row['Brand']] = []
        if row['Model_new'] not in Model_list[row['Brand']]:
            Model_list[row['Brand']].append(row['Model_new'])
    CC_list = {}
    for _, row in data.iterrows():
        if row['Model_new'] not in CC_list:
            CC_list[row['Model_new']] = []
        if row['CC'] not in CC_list[row['Model_new']]:
            CC_list[row['Model_new']].append(row['CC'])

    Age_list = [0,1,2,3,4,5,6,7]
    
    st.markdown("<h1 style='text-align: right; color: #64469b;'font-size:15px; margin-top:-20px;margin-right:200px; font-family:Helvetica;'>DriveX Vehicle Sale Price Prediction</h1>", unsafe_allow_html=True)
    uniq000 = "key000"
    uniq001 = "key001"
    uniq002 = "key002"
    uniq003 = "key003"
    uniq004 = "key004"
    col1, col2,col3,col4= st.columns([0.5,0.8,0.8,0.8])
    with col3:
        Brand = st.selectbox("Brand", Brand_list,key=uniq001)
        @st.cache_data
        def update_model_options(Brand):
            return Model_list.get(Brand, [])
        Model = st.selectbox("Model", update_model_options(Brand) )
        
        def handle_brand_change():
            global Model
            Model = update_model_options(Brand)[0] if update_model_options(Brand) else None
        if Brand:
            handle_brand_change()
    cc_list = [0]
    with col4:
        if Model:
            CC = st.selectbox("CC", CC_list.get(Model, [])) 
        else:
            CC= st.selectbox("CC",cc_list)
        Age = st.selectbox('Age',Age_list )
    col3,col4 = st.columns([3,2])
    with col3:
        st.write("")
    with col4:
        button_style = """
            <style>
                .stButton button {
                    background-color: #64469b;
                    color: white;
                }
                .stButton button.clicked {
                    background-color: white;
                    color: white;
                }
                header[data-testid="stHeader"] {
                    display: none;
                }
            </style>
            """

        # Displaying the button
        st.markdown(button_style, unsafe_allow_html=True)
        if st.button("Predict Price"):
            try:
                Sale_price = price_prediction(Brand, Model, CC, Age)
                Sale_price = round(Sale_price )
                Sale_price = f"{Sale_price:,}"
            
                
                custom_css1 = """
                    <style>
                    .rounded-box {
                        background-color: #f47624;
                        padding: 0px;
                        margin: 0 auto;
                        text-align: center;
                        width: 240px;
                        height: 130px;
                        transform: translateX(-50%);
                        border-radius: 15px; /* Adjust the value to change the roundness */
                    }
                    .rounded-box h1, .rounded-box h2 {
                        color: white;
                        font-size: 20px;
                    }
                    .rounded-box h2 {
                        font-size: 40px;
                        color:white;
                    }
                    </style>
                    """

                # Apply the custom CSS style
                st.markdown(custom_css1, unsafe_allow_html=True)

                # Display the rounded corner box with dynamic sale price
                st.markdown(
                    f"<div class='rounded-box'>"
                    f"<h1>Predicted Price</h1>"
                    f"<h2>\u20B9 {Sale_price}</h2>"
                    f"</div>",
                    unsafe_allow_html=True
                )
            except Exception as e:
                st.write("Invalid Data")