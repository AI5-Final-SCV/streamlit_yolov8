# Python In-built packages
from pathlib import Path
import PIL
import numpy as np
# External packages
import streamlit as st
from ultralytics import YOLO
import cv2

# Local Modules
import settings
import helper


from streamlit_modal import Modal
import streamlit.components.v1 as components

# Setting page layout
st.set_page_config(
    page_title="Object Detection using YOLOv8",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page heading
st.title("Object Detection using YOLOv8")

# Sidebar
st.sidebar.header("FastCampus AI5 S.C.V")

# Model Options
model_size = st.sidebar.radio(
    "Model Size", ['N', 'X'])

smart = False
if st.sidebar.radio("Model Option", ['LifeJacket In Pool', 'SmartInsideAI']) == 'LifeJacket In Pool':
    model_type = st.sidebar.radio("Model Type", ['Detection', 'Segmentation', 'Detection + Segmentation'])
    smart = False
else:
    model_type = st.sidebar.radio("Model Type", ['Detection', 'Segmentation'])
    smart =True

confidence = float(st.sidebar.slider(
    "Select Model Confidence", 25, 100, 40)) / 100

# 경고 팝업
   
def warning():
    modal = Modal(key="Demo Key",title="")
    modal.open()

    if modal.is_open():    
        with modal.container():
            st.write('<div style="font-weight:bold;font-size:24px;color:red">Situations Occur</div>', unsafe_allow_html=True)


            html_string = '''
                <h1>WARNING!!!</h1>
                <script language="javascript">
                document.querySelector("h1").style.color = "red";
                </script>
            '''
            components.html(html_string)
        
            value = st.checkbox("Please check the situation and check the box when resolved")
            if value:
                modal.close()

# Selecting Detection Or Segmentation
if model_size == 'N':
    if smart == False:
        if model_type == 'Detection':
            model_path = Path(settings.DETECTION_MODEL_N)
        elif model_type == 'Segmentation':
            model_path = Path(settings.SEGMENTATION_MODEL_N)
        elif model_type == 'Detection + Segmentation':
            model_path = Path(settings.DETECTION_MODEL_N)
    else:
        if model_type == 'Detection':
            model_path = Path(settings.SMARTDETECTION_MODEL_N)
        elif model_type == 'Segmentation':
            model_path = Path(settings.SMARTSEGMENTATION_MODEL_N)
elif model_size == 'X':
    if smart == False:
        if model_type == 'Detection':
            model_path = Path(settings.DETECTION_MODEL_X)
        elif model_type == 'Segmentation':
            model_path = Path(settings.SEGMENTATION_MODEL_X)
        elif model_type == 'Detection + Segmentation':
            model_path = Path(settings.DETECTION_MODEL_X)
    else:
        if model_type == 'Detection':
            model_path = Path(settings.SMARTDETECTION_MODEL_X)
        elif model_type == 'Segmentation':
            model_path = Path(settings.SMARTSEGMENTATION_MODEL_X)

# Load Pre-trained ML Model
try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

st.sidebar.header("Image/Video Config")
source_radio = st.sidebar.radio(
    "Select Source", settings.SOURCES_LIST)

source_img = None
# If image is selected
if source_radio == settings.IMAGE:
    source_img = st.sidebar.file_uploader(
        "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

    col1, col2 = st.columns(2)

    with col1:
        try:
            if smart == False:
                if source_img is None:
                    default_image_path = str(settings.DEFAULT_IMAGE)
                    default_image = PIL.Image.open(default_image_path)
                    st.image(default_image_path, caption="Default Image",
                            use_column_width=True)
                else:
                    uploaded_image = PIL.Image.open(source_img)
                    st.image(source_img, caption="Uploaded Image",
                            use_column_width=True)
            else:
                if source_img is None:
                    if model_type == 'Detection':
                        default_image_path = str(settings.SMART_DEFAULT_IMAGE)
                        default_image = PIL.Image.open(default_image_path)
                        st.image(default_image_path, caption="Default Image",
                                use_column_width=True)                   
                    elif model_type == 'Segmentation':
                        default_image_path = str(settings.SMART_DEFAULT_SEG_IMAGE)
                        default_image = PIL.Image.open(default_image_path)
                        st.image(default_image_path, caption="Default Image",
                                use_column_width=True)                                
                else:
                    uploaded_image = PIL.Image.open(source_img)
                    st.image(source_img, caption="Uploaded Image",
                            use_column_width=True)                        
        except Exception as ex:
            st.error("Error occurred while opening the image.")
            st.error(ex)

    with col2:
        if source_img is None:
            if smart == False:
                if model_type == 'Detection':
                    default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
                    default_detected_image = PIL.Image.open(
                        default_detected_image_path)
                    st.image(default_detected_image_path, caption='Detected Image',
                            use_column_width=True)
                elif model_type == 'Segmentation':
                    default_detected_image_path = str(settings.DEFAULT_SEG_IMAGE)
                    default_detected_image = PIL.Image.open(
                        default_detected_image_path)
                    st.image(default_detected_image_path, caption='Segmentation Image',
                            use_column_width=True)
                else:
                    default_detected_image_path = str(settings.DEFAULT_DESEG_IMAGE)
                    default_detected_image = PIL.Image.open(
                        default_detected_image_path)
                    st.image(default_detected_image_path, caption='Detection + Segmentation Image',
                            use_column_width=True)
            else:
                if model_type == 'Detection':
                    default_detected_image_path = str(settings.SMART_DEFAULT_DETECT_IMAGE)
                    default_detected_image = PIL.Image.open(
                        default_detected_image_path)
                    st.image(default_detected_image_path, caption='Detected Image',
                            use_column_width=True)
                elif model_type == 'Segmentation':
                    default_detected_image_path = str(settings.SMART_DEFAULT_DESEG_IMAGE)
                    default_detected_image = PIL.Image.open(
                        default_detected_image_path)
                    st.image(default_detected_image_path, caption='Segmentation Image',
                            use_column_width=True)
        else:
            if model_type == 'Detection' or model_type == 'Segmentation':
                if st.sidebar.button('Detect Objects'):
                    res = model.predict(uploaded_image,
                                        conf=confidence
                                        )
                    boxes = res[0].boxes
                    res_plotted = res[0].plot()[:, :, ::-1]
                    st.image(res_plotted, caption='Detected Image',
                            use_column_width=True)
                    try:
                        with st.expander("Detection Results"):
                            for box in boxes:
                                st.write(box.data)
                    except Exception as ex:
                        # st.write(ex)
                        st.write("No image is uploaded yet!")
            
            else:
                model_segmentation_path = str(settings.SEGMENTATION_MODEL)
                model_segmentation = YOLO(model_segmentation_path)
                if st.sidebar.button('Detect Objects'):
                    res = model.predict(uploaded_image, conf=confidence)
                    boxes = res[0].boxes
                    res_plotted = res[0].plot()[:, :, ::-1]

                    frame = np.array(uploaded_image)
                    results2 = model_segmentation(frame)


                    masks = (results2[0].masks.data.cpu().numpy() * 255).astype(np.uint8)

                    resized_mask = cv2.resize(masks[0], (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)

                    colored_mask = np.zeros_like(frame)
                    colored_mask[:, :, 1] = resized_mask  # 색상설정

                    combined_frame = cv2.addWeighted(res_plotted, 1, colored_mask, 0.5, 0)
                    st.image(combined_frame, caption='Detected Image', use_column_width=True)

                    try:
                        with st.expander("Detection Results"):
                            for box in boxes:
                                st.write(box)
                    except:
                        st.write("No objects detected.")
elif source_radio == settings.VIDEO:

    col1, col2 = st.columns(2)
    if model_type == 'Detection' or model_type == 'Segmentation':
            if smart == False:
                with col1:
                    helper.play_stored_video(confidence, model, smart)
            else:
                with col1:
                    helper.play_stored_video(confidence, model, smart)
            
    else:
        if model_size == "N":
            if smart == False:
                model_detection_path = str(settings.DETECTION_MODEL_N)
                model_detection = YOLO(model_detection_path)
                model_segmentation_path = str(settings.SEGMENTATION_MODEL_N)
                model_segmentation = YOLO(model_segmentation_path)
            else:                
                model_detection_path = str(settings.SMARTDETECTION_MODEL_N)
                model_detection = YOLO(model_detection_path)
                model_segmentation_path = str(settings.SMARTSEGMENTATION_MODEL_N)
                model_segmentation = YOLO(model_segmentation_path)
            with col1:
                helper.play_stored_video2(confidence,model_detection, model_segmentation, smart)

        elif model_size == "X":
            if smart == False:
                model_detection_path = str(settings.DETECTION_MODEL_X)
                model_detection = YOLO(model_detection_path)
                model_segmentation_path = str(settings.SEGMENTATION_MODEL_X)
                model_segmentation = YOLO(model_segmentation_path)
            else:                
                model_detection_path = str(settings.SMARTDETECTION_MODEL_X)
                model_detection = YOLO(model_detection_path)
                model_segmentation_path = str(settings.SMARTSEGMENTATION_MODEL_X)
                model_segmentation = YOLO(model_segmentation_path)
            with col1:
                helper.play_stored_video2(confidence,model_detection, model_segmentation, smart)


elif source_radio == settings.YOUTUBE:
    if model_type == 'Detection' or model_type == 'Segmentation':
            if smart == False:
                helper.play_youtube_video(confidence, model)
            else:
                helper.play_youtube_video(confidence, model)
    else:
        if model_size == "N":
            if smart == False:
                model_detection_path = str(settings.DETECTION_MODEL_N)
                model_detection = YOLO(model_detection_path)
                model_segmentation_path = str(settings.SEGMENTATION_MODEL_N)
                model_segmentation = YOLO(model_segmentation_path)
            else:                
                model_detection_path = str(settings.SMARTDETECTION_MODEL_N)
                model_detection = YOLO(model_detection_path)
                model_segmentation_path = str(settings.SMARTSEGMENTATION_MODEL_N)
                model_segmentation = YOLO(model_segmentation_path)
            helper.play_youtube_video2(confidence,model_detection, model_segmentation)
        elif model_size == "X":
            if smart == False:
                model_detection_path = str(settings.DETECTION_MODEL_X)
                model_detection = YOLO(model_detection_path)
                model_segmentation_path = str(settings.SEGMENTATION_MODEL_X)
                model_segmentation = YOLO(model_segmentation_path)
            else:                
                model_detection_path = str(settings.SMARTDETECTION_MODEL_X)
                model_detection = YOLO(model_detection_path)
                model_segmentation_path = str(settings.SMARTSEGMENTATION_MODEL_X)
                model_segmentation = YOLO(model_segmentation_path)
            helper.play_youtube_video2(confidence,model_detection, model_segmentation)
else:
    st.error("Please select a valid source type!")
