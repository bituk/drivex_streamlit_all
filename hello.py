import streamlit as st

# Create tabs
tab_list = ["Price Prediction", "OdoMeter Reading", "Number Plate Extraction", "Auto Annotation", "Vehicle Part Detection"]
tab1, tab2, tab3, tab4, tab5 = st.tabs(tab_list)

# Display tab content
if tab1:
    st.header("Price Prediction")
    uploaded_file = st.file_uploader("Upload a file", key="tab1_file_uploader")

    if uploaded_file is not None:
        # Your processing code here for tab 1

elif tab2:
    st.header("OdoMeter Reading")
    uploaded_file = st.file_uploader("Upload a file", key="tab2_file_uploader")

    if uploaded_file is not None:
        # Your processing code here for tab 2

elif tab3:
    st.header("Number Plate Extraction")
    uploaded_file = st.file_uploader("Upload a file", key="tab3_file_uploader")

    if uploaded_file is not None:
        # Your processing code here for tab 3

elif tab4:
    st.header("Auto Annotation")
    uploaded_file = st.file_uploader("Upload a file", key="tab4_file_uploader")

    if uploaded_file is not None:
        # Your processing code here for tab 4

elif tab5:
    st.header("Vehicle Part Detection")
    uploaded_file = st.file_uploader("Upload a file", key="tab5_file_uploader")

    if uploaded_file is not None:
        # Your processing code here for tab 5
