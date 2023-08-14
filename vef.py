def vef_run():
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
    import time
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


    def save_video(frames, video_path):
        # Create a video writer object
        frame_width, frame_height = frames[0].shape[1], frames[0].shape[0]
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

        # Write each frame to the video file
        for frame in frames:
            out.write(frame)

        # Release the video writer
        out.release()

    def get_video_download_link(file_path):
        with open(file_path, "rb") as file:
            video_data = file.read()
        encoded_video = base64.b64encode(video_data).decode()
        href = f'<a href="data:video/mp4;base64,{encoded_video}" download="predicted_video.mp4">Click here to download the predicted video</a>'
        return href

    def simulate_processing_time(seconds):
        time.sleep(seconds)
    # Replace the relative path to your weight file
    current_directory = os.getcwd()
    model_path = os.path.join(current_directory,"best1.pt")

    def Depreciation_price(VEF_code):
        print("The VEF code for this vehicle is ",VEF_code)
        position=0
        Total_price = 0
        mapped_price_item = {"dent":200,"headlight":500,"Backlight":400,"indicator":400,"seat":800,"mirror":300}
        for i in VEF_code:
            if position==0:
                print("The depreciation price for dent is",int(i)*mapped_price_item["dent"])
                Total_price+=(int(i)*mapped_price_item["dent"])
                #print(VEF_code[0])
            if (position == 1) and int(i)==0:
                print("The depreciation price for Headlight is",mapped_price_item["headlight"])
                #print(VEF_code[1])
                Total_price+=mapped_price_item["headlight"]
            if position == 2 and int(i)==0:
                print("The depreciation price for Backlight is",mapped_price_item["Backlight"])
                #print(VEF_code[2])
                Total_price+=mapped_price_item["Backlight"]
            if position == 3 and int(i)==0:
                print("The depreciation price for indiator is",mapped_price_item["indicator"])
                #print(VEF_code[3])
                Total_price+=mapped_price_item["indicator"]
            if position == 4 and int(i)==0:
                print("The depreciation price for seat is",mapped_price_item["seat"])
                #print(VEF_code[4])
                Total_price+=mapped_price_item["seat"]
            if position == 5 and int(i)==0:
                print("The depreciation price for mirror is",mapped_price_item["mirror"])
                #print(VEF_code[5])
                Total_price+=mapped_price_item["mirror"]
            position+=1
        print("The Total depreciation price for vehicle is",Total_price)
        return Total_price
    def Vef_code(file_path):
        current_directory = os.getcwd()
        videos_directory = os.path.join(current_directory,"videos")
        #creating videos directory to store videos file
        if not os.path.exists(videos_directory):
            os.makedirs(videos_directory)
        filename = os.path.basename(file_path)
        base_name, extension = os.path.splitext(filename)
        file_location = f"{videos_directory}/{base_name}.{extension}"
        annotated_video_path = os.path.join(current_directory,"annotated_videos")
        if not os.path.exists(annotated_video_path):
            os.makedirs(annotated_video_path)
        else:
            shutil.rmtree(annotated_video_path)
            os.makedirs(annotated_video_path)
        def predict_video(video_path): 
            filename = os.path.basename(file_location)
            base_name, extension = os.path.splitext(filename)
            
            # Read the input video file
            video = cv2.VideoCapture(video_path)
            
            # Get the video's frame rate, width, and height
            fps = int(video.get(cv2.CAP_PROP_FPS))
            width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Initialize the video writer with the same frame rate, width, and height as the input video
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            output_path = f"{annotated_video_path}/{base_name}mp4"
            print("output path",output_path)
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            model1_path = os.path.join(current_directory,"modelA1.pt")
            model1 = YOLO(model1_path)
            # Process the video frame by frame
            while True:
                # Read the next frame from the video
                ret, frame1 = video.read()
                
                # Break the loop if we have reached the end of the video
                if not ret:
                    break
                
                # Perform object detection on the frame using the yolov8 model
                results = model1.predict(frame1,conf=0.7,verbose=False)
                annotated_frame = results[0].plot()
                out.write(annotated_frame)
                
            # Release the video and video writer objects
            out.release()
            
            # Return the path of the output video file
            print("The annotated video is saved in ",output_path)
            return output_path
        video_path = file_path
        # Create a shared variable to store the output path
        output_path_container = []
        # Define a function to run the predict_video function in the thread
        def run_predict_video(video_path):
            output_path = predict_video(video_path)
            output_path_container.append(output_path)
        # Create Thread 1
        thread1 = threading.Thread(target=run_predict_video,args=(video_path,))
        # Start the thread
        thread1.start()
        # Loading videos
        video = cv2.VideoCapture(file_path)

        # checking video is loaded successfully or not
        if not video.isOpened():
            print("Error in video loading")
        else:
            print("Video loaded successfully")
        
        # Extracted_frames directory
        Extracted_frames_dir = os.path.join(current_directory,"extracted_frames")
        if not os.path.exists(Extracted_frames_dir):
            os.makedirs(Extracted_frames_dir)
        
        # Initialize frame count and timestamp variables
        frame_count = 0
        timestamp = 0
        while True:
            # Set the current frame position
            video.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)  # Convert to milliseconds

            # Read the next frame
            success, frame = video.read()

            # Break the loop if no frame is read
            if not success:
                break

            # Generate a frame file name
            frame_name = f"{Extracted_frames_dir}/frame_{frame_count}.jpg"

            # Save the frame as an image file
            cv2.imwrite(frame_name, frame)

            # Increment the frame count
            frame_count += 1

            # Update the timestamp to the next second
            timestamp += 1
        
        #Loading model
        model_path = os.path.join(current_directory,"best1.pt")
        model = YOLO(model_path)
        # Making empty list for storing predicted_class names
        predicted_labels = []
        # Getting predicted lables from extracted image
        for i in os.listdir(Extracted_frames_dir):
            # Image
            im = os.path.join(Extracted_frames_dir,i)
            # Inference
            results = model.predict(im,verbose=False,conf=0.80)

            for i in list(map(lambda x:int(x),results[0].boxes.cls.tolist())):
                labels = model.names[i]
                if labels not in predicted_labels:
                    predicted_labels.append(model.names[i])
        
        # For removing stored file
        for i in os.listdir(Extracted_frames_dir):
            os.remove(os.path.join(Extracted_frames_dir,i))
        print(predicted_labels)

        mapped_part_item = {"Good Headlight":1,"Good Backlight":1,"Good Mirror":1,"Good Seat":1,"Good indicator":1,"Bad Headlight":0,"Bad Backlight":0,"Bad Mirror":0,"Bad Seat":0,"Bad indicator":0}
        # Making a function for getting parts code
        def parts_code(n):
            headlight_code = 0
            backlight_code = 0
            indicator_code = 0
            mirror_code = 0
            seat_code = 0
            predicted_list = n
            
            # For headlight
            if "Good Headlight" in predicted_list:
                if "Bad Headlight" in predicted_list:
                    headlight_code = mapped_part_item["Bad Headlight"]
                else:
                    headlight_code = mapped_part_item["Good Headlight"]
                
            # For backlight
            if "Good Backlight" in predicted_list:
                if "Bad Backlight" in predicted_list:
                    backlight_code = mapped_part_item["Bad Backlight"]
                else:
                    backlight_code = mapped_part_item["Good Backlight"]

            # For indicator
            if "Good indicator" in predicted_list:
                if "Bad indicator" in predicted_list:
                    indicator_code = mapped_part_item["Bad indicator"]
                else:
                    indicator_code = mapped_part_item["Good indicator"]

            # For Seat
            if "Good Seat" in predicted_list:
                if "Bad Seat" in predicted_list:
                    seat_code = mapped_part_item["Bad Seat"]
                else:
                    seat_code = mapped_part_item["Good Seat"]

            # For mirror
            if "Good Mirror" in predicted_list:
                if "Bad Mirror" in predicted_list:
                    mirror_code = mapped_part_item["Bad Mirror"]
                else:
                    mirror_code = mapped_part_item["Good Mirror"]
                
            #result_code
            result_code = str(headlight_code)+str(backlight_code)+str(indicator_code)+str(seat_code)+str(mirror_code)
            return result_code
        # For dent detection
        # Initialize frame count and timestamp variables
        frame_count = 0
        timestamp = 0
        while True:
            # Set the current frame position
            video.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000*8)  # Convert to milliseconds and getting 1 image per 8 second

            # Read the next frame
            success, frame = video.read()

            # Break the loop if no frame is read
            if not success:
                break

            # Generate a frame file name
            frame_name = f"{Extracted_frames_dir}/frame_{frame_count}.jpg"

            # Save the frame as an image file
            cv2.imwrite(frame_name, frame)

            # Increment the frame count
            frame_count += 1

            # Update the timestamp to the next second
            timestamp += 1
        # Making empty list for storing predicted_class names
        predicted_dent_labels = []
        # Getting predicted lables from extracted image
        for i in os.listdir(Extracted_frames_dir):
            # Image
            im = os.path.join(Extracted_frames_dir,i)
            # Inference
            results = model(im,verbose=False)
            for i in list(map(lambda x:int(x),results[0].boxes.cls.tolist())):
                labels = model.names[i]
                if labels.lower()=="dents" or labels.lower()=="scratch":
                    predicted_dent_labels.append(labels)   
        Dents_count = len(predicted_dent_labels)
        # Number of dent should be less than 10
        if Dents_count >= 10:
            Dents_count = 9
        # For removing stored extracted file
        for i in os.listdir(Extracted_frames_dir):
            os.remove(os.path.join(Extracted_frames_dir,i))
        print("The Number of dents and scatch are",predicted_dent_labels.count("Dents"))
        # VEF code extracter

        def vef(predicted_class,dent_count):
            VEF_code = ""
            number_of_dents = dent_count
            n=predicted_class
            VEF_code = str(number_of_dents)+ parts_code(n)
            return VEF_code
        VEF_code= vef(predicted_labels,Dents_count)
        # Wait for the thread to complete
        thread1.join()
        # Check if the thread has completed successfully
        if not thread1.is_alive():
            print("Thread completed successfully.")
        else:
            print("Thread did not complete successfully.")

        # Access the result from the shared variable
        if output_path_container:
            annotated_path = output_path_container[0]
            print("Annotated video path:", annotated_path)
        else:
            annotated_path = ""
            print("Thread did not produce an output path.")
        #Releasing video or clsoing video
        video.release()
        
        #return {"VEF_code":VEF_code,"Predicted_video_path":annotated_path}
        return {"VEF_code":VEF_code,}
    
    col1, col2,col3,col4= st.columns([0.5,0.8,1,0.7])

    with col1:
        st.write("")
    with col3:
        st.markdown("<h2 style='text-align: right; color: #64469b;'font-size:15px; margin-right:00px; font-family:Helvetica;'>Depreciation Price Calculation</h2>", unsafe_allow_html=True)
        custom_css = """
        <style>
        .css-1gulkj5 {
            display: flex;
            -webkit-box-align: center;
            align-items: center;
            padding: 0.6rem;
            width:400px;
            background-color: #FF7B33;
            border-radius: 3rem;
            color: rgb(49, 51, 63);
            margin-left:60px;
        }
        </style>
        """
        st.markdown(custom_css, unsafe_allow_html=True)
        uploaded_file_css = """
        <style>
        .css-fis6aj {
        left: 0px;
        right: 0px;
        line-height: 1.25;
        padding-top: 0.75rem;
        padding-left: 1rem;
        padding-right: 0.2rem;
        }
        </style>"""
        st.markdown(uploaded_file_css, unsafe_allow_html=True)
        # Adding file uploader to sidebar for selecting videos
        uploaded_file = st.file_uploader(" ", type=["mp4"],key="tab1_file_uploader")
        
        # Customizing the button color
        button_style = """
            <style>
            .stButton button {
                background-color:#8833FF;
                color: #ffffff;
                padding: 0.5rem;
                width:300px;
                margin-top:50px;
                border-radius: 3rem;
                margin-left:120px;
            }
            </style>
            """
        # Displaying the button
        st.markdown(button_style, unsafe_allow_html=True)
        detect_objects = st.button("Detect Video",key="detection_key")

    if detect_objects:
        with col3:
            
            if uploaded_file is None:
                # Custom container for warning message
                warning_container = st.container()
                # Apply custom styling to the warning container
                warning_container.markdown(
                    """
                    <style>
                        .warning-container {
                            background-color: #000000;
                            padding: 0.5rem;
                            width: 300px;
                            margin-top: 50px;
                            border-radius: 3rem;
                            margin-left: 120px;
                        }
                        .warning-text {
                            color: #ffffff;
                            text-align: center; 
                        }
                    </style>
                    """,
                    unsafe_allow_html=True
                )

                # Display warning message within the custom container
                with warning_container:
                    st.markdown('<div class="warning-container"><div class="warning-text">Please upload a video to detect objects.</div></div>', unsafe_allow_html=True)
            else:
                # Custom container for info message
                info_container = st.container()
                # Apply custom styling to the info container
                info_container.markdown(
                    """
                    <style>
                        .info-container {
                            background-color: #000000;
                            padding: 0.5rem;
                            width: 300px;
                            height: 70px; /* Adjust height as needed */
                            margin-top: 50px;
                            border-radius: 1rem;
                            margin-left: 120px;
                            display: flex;
                            align-items: center;
                            justify-content: center;
                        }
                        .info-text {
                            text-align: center;
                            color: #ffffff;
                        }
                    </style>
                    """,
                    unsafe_allow_html=True
                )
                # Display info message within the custom container
                info_container = st.empty()
                with info_container:
                    st.markdown('<div class="info-container"><div class="info-text">Processing video and performing object detection...</div></div>', unsafe_allow_html=True)
                    simulate_processing_time(3)  # Adjust the time as needed
                    # Add your object detection code here
                    with tempfile.TemporaryDirectory() as temp_dir:
                        video_path = os.path.join(temp_dir, "uploaded_video.mp4")
                        with open(video_path, "wb") as file:
                            file.write(uploaded_file.read())
                        vef_code = Vef_code(video_path)
                    st.markdown('<div class="info-container"><div class="info-text">Detection Completed</div></div>', unsafe_allow_html=True)
                    # Once the object detection is completed
                    vef_code1 = vef_code['VEF_code']
                    Total_price = Depreciation_price(vef_code1)
                    depreciation_price = Total_price
                    message = f'VEF Code-{vef_code1}<br>Depreciation Price {depreciation_price}'
                    simulate_processing_time(2)
                    st.markdown(f'<div class="info-container"><div class="info-text">{message}</div></div>', unsafe_allow_html=True)
                    
                    #Total_price = Depreciation_price(vef_code['VEF_code'])
                

    with col4:
        # Customizing the button color
        download_button = st.button("Download","ann_video_key")   
    annotated_video_path = os.path.join(current_directory,"annotated_videos")
    output_path_video = glob.glob(annotated_video_path + "/*.mp4")
    if download_button:
        with col4:
            try:
                if not os.path.exists(output_path_video[0]):
                    # Custom container for warning message
                    warning2_container = st.container()
                    # Apply custom styling to the warning container
                    warning2_container.markdown(
                        """
                        <style>
                            .warning2-container {
                                background-color: #000000;
                                padding: 0.5rem;
                                width: 300px;
                                margin-top: 50px;
                                border-radius: 3rem;
                                margin-left: 120px;
                            }
                            .warning-text {
                                color: #ffffff;
                                text-align: center; 
                            }
                        </style>
                        """,
                        unsafe_allow_html=True
                    )

                    # Display warning message within the custom container
                    with warning2_container:
                        st.markdown('<div class="warning-container"><div class="warning-text">Error in video downloading.Please wait to finish the detection process</div></div>', unsafe_allow_html=True)
                else:
                    # Custom container for info message
                    info_container1 = st.container()
                    # Apply custom styling to the info container
                    info_container1.markdown(
                        """
                        <style>
                            .info_container1 {
                                background-color: #000000;
                                padding: 0.5rem;
                                width: 300px;
                                height: 70px; /* Adjust height as needed */
                                margin-top: 50px;
                                border-radius: 1rem;
                                margin-left: 120px;
                                display: flex;
                                align-items: center;
                                justify-content: center;
                            }
                            .info-text {
                                text-align: center;
                                color: #ffffff;
                            }
                        </style>
                        """,
                        unsafe_allow_html=True
                    )
                    # Display info message within the custom container
                    info_container1 = st.empty()
                    with info_container1:
                        st.markdown('<div class="info-container"><div class="info-text">Please wait. Downloading is in progress</div></div>', unsafe_allow_html=True)
                        link = get_video_download_link(output_path_video[0])
                        simulate_processing_time(2)
                        st.markdown('<div class="info-container"><div class="info-text">Download link is ready now</div></div>', unsafe_allow_html=True)
                    st.markdown(get_video_download_link(output_path_video[0]), unsafe_allow_html=True)
                    # Display info message within the custom container
                    info_container1 = st.empty()
                    with info_container1:
                        simulate_processing_time(2)
                        st.markdown(f'<div class="info-container"><div class="info-text">Download Completed</div></div>', unsafe_allow_html=True)
                        simulate_processing_time(3)
            except IndexError:
                # Custom container for info message
                info_container1 = st.container()
                # Apply custom styling to the info container
                info_container1.markdown(
                    """
                    <style>
                        .info_container1 {
                            background-color: #000000;
                            padding: 0.5rem;
                            width: 300px;
                            height: 70px; /* Adjust height as needed */
                            margin-top: 50px;
                            border-radius: 1rem;
                            margin-left: 120px;
                            display: flex;
                            align-items: center;
                            justify-content: center;
                        }
                        .info-text {
                            text-align: center;
                            color: #ffffff;
                        }
                    </style>
                    """,
                    unsafe_allow_html=True
                )
                # Display info message within the custom container
                info_container1 = st.empty()
                with info_container1:
                    st.markdown(f'<div class="info-container"><div class="info-text">Some Error in Video Downloading.<br>Please try again</div></div>', unsafe_allow_html=True)
if __name__ == "__main__":
    vef_run()
