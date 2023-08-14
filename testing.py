# Importing necessary library
import cv2
import streamlit as st
from ultralytics import YOLO
import tempfile
import os
import base64
from PIL import Image
from streamlit import get_option, set_option
import time
import glob
import threading
import shutil
from streamlit_option_menu import option_menu
from vef import vef_run
from auto_annotation import annotation
from Main import Login
current_directory = os.getcwd()
def add_bg_from_local(image_file):
        with open(image_file, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
        st.markdown(
            f"""
        <style>
        .stApp {{
            background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
            background-size: cover
        }} 
         header[data-testid="stHeader"] {{
        display:none;
        }}
        
        <style>         """,
            unsafe_allow_html=True
        )
# Setting page layout
set_option('deprecation.showfileUploaderEncoding', False)
st.set_page_config(
    page_title="DriveX",  # Setting page title
    page_icon="🤖",  # Setting page icon
    layout="wide",  # Setting layout to wide
    initial_sidebar_state="expanded"  # Expanding sidebar by default
)
# Creating main page heading
background_img_path = os.path.join(current_directory,"background_update.jpg")
add_bg_from_local(background_img_path)
paje_style = """<style>
.st-bc {
        display: flex;
        flex-direction: column;
        align-items: flex-start;
        margin-top: -10px;
    }
.tabs {
        width: 100%; 
    }
    .tab-content {
        width: 100%;   
    }
.stApp {
        max-width: 100%;
        margin: 0;
    }
.full-width-content {
        width: 100%;
        max-width: 100%;
    }
    </style>"""
st.markdown(paje_style, unsafe_allow_html=True)
    
# Create tabs
tab_list = ["Price Prediction", "OdoMeter Reading", "Number Plate Extraction","Auto Annotation","Vehicle Part Detection"]
#tab1, tab2, tab3,tab4,tab5 = st.tabs(tab_list)
selected = option_menu(
    menu_title=None,
    options=["Price Prediction", "Auto Annotation","Vehicle Part Detection","OdoMeter Reading", "Number Plate Extraction",],
    icons=["currency-rupee","speedometer2","badge-ar-fill","arrow-right-square","arrow-right-circle-fill"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
    styles={
        "container":{"padding":"0!important","background-color":"pink"},
        "icon":{"color":"orange","font-size":"25px"},
        "nav-link":{
            "font-size":"15px",
            "text-align":"left",
            "margin":"0px",
            "--hover-color":"#eb8334",
        },
        "nav-link-selected":{"background-color":"#8833FF"},
    }
)
if selected=="Price Prediction":
    Login()
if selected=="OdoMeter Reading":
    pass
if selected=="Number Plate Extraction":
    pass
if selected=="Auto Annotation":
    annotation()
if selected=="Vehicle Part Detection":
    vef_run()
