from ultralytics import YOLO
import streamlit as st
import cv2
import numpy as np
import settings
import yt_dlp
import app

from streamlit_modal import Modal
import streamlit.components.v1 as components

def load_detection_segmentation_model(detection_model_path, segmentation_model_path):
    detection_model = YOLO(detection_model_path)
    segmentation_model = YOLO(segmentation_model_path)
    return detection_model, segmentation_model


def load_model(model_path):
    model = YOLO(model_path)
    return model


def display_tracker_options():
    display_tracker = st.radio("Display Tracker", ('Yes', 'No'))
    is_display_tracker = True if display_tracker == 'Yes' else False
    if is_display_tracker:
        tracker_type = st.radio("Tracker", ("bytetrack.yaml", "botsort.yaml"))
        return is_display_tracker, tracker_type
    return is_display_tracker, None


def _display_detected_frames(conf, model, st_frame, image, is_display_tracking=None, tracker=None):

    if is_display_tracking:
        res = model.track(image, conf=conf, persist=True, tracker=tracker)
    else:
        res = model.predict(image, conf=conf)

    res_plotted = res[0].plot()
    st_frame.image(res_plotted,
                   caption='Detected Video',
                   channels="BGR",
                   use_column_width=True
                   )


def play_youtube_video(conf, model):
    source_youtube = st.sidebar.text_input("YouTube Video url")


    if st.sidebar.button('Detect Objects'):
        try:
            ydl_opts = {
                'format': 'best',
                'postprocessors': [{
                    'key': 'FFmpegVideoConvertor',
                    'preferedformat': 'mp4',
                }],
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info_dict = ydl.extract_info(source_youtube, download=False)
                video_url = info_dict['url']

            vid_cap = cv2.VideoCapture(video_url)
            st_frame = st.empty()

            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image
                                             )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))

  
def play_youtube_video2(conf, model_detection, model_segmentation):

    source_youtube = st.sidebar.text_input("YouTube Video url")

    if st.sidebar.button('Detect Objects'):
        try:
            ydl_opts = {
                'format': 'best',
                'postprocessors': [{
                    'key': 'FFmpegVideoConvertor',
                    'preferedformat': 'mp4',
                }],
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info_dict = ydl.extract_info(source_youtube, download=False)
                video_url = info_dict['url']

            vid_cap = cv2.VideoCapture(video_url)
            st_frame = st.empty()

            while vid_cap.isOpened():
                success, frame = vid_cap.read()
                if success:
                    results_detection = model_detection(frame)
                    results_segmentation = model_segmentation(frame)

                    annotated_frame = results_detection[0].plot()

                    masks = (results_segmentation[0].masks.data.cpu().numpy() * 255).astype(np.uint8)

                    resized_mask = cv2.resize(masks[0], (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)

                    colored_mask = np.zeros_like(frame)
                    colored_mask[:, :, 1] = resized_mask  # 색상설정

                    combined_frame = cv2.addWeighted(annotated_frame, 1, colored_mask, 0.5, 0)

                    _display_detected_frames(conf,
                                            model_detection,
                                            st_frame,
                                            combined_frame,
                                            None,
                                            None
                                            )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))

def play_stored_video(conf, model, smart):
    video_dict = settings.VIDEOS_DICT if not smart else settings.VIDEOS_DICT_2

    source_vid = st.sidebar.selectbox("Choose a video...", video_dict.keys())
    with open(video_dict.get(source_vid), 'rb') as video_file:
        video_bytes = video_file.read()

    if video_bytes:
        st.video(video_bytes)

    if st.sidebar.button('Detect Video Objects'):
        try:
            vid_cap = cv2.VideoCapture(str(video_dict.get(source_vid)))
            st_frame = st.empty()
            while vid_cap.isOpened():
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image,
                                             None,
                                             None
                                             )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))



def play_stored_video2(conf, model_detection, model_segmentation, smart):
    if not smart:
        video_dict = settings.VIDEOS_DICT
    else:
        video_dict = settings.VIDEOS_DICT_2

    source_vid = st.sidebar.selectbox("Choose a video...", video_dict.keys())

    with open(video_dict.get(source_vid), 'rb') as video_file:
        video_bytes = video_file.read()
    if video_bytes:
        st.video(video_bytes)
           
    if st.sidebar.button('Detect Video Objects'):
        try:
            vid_cap = cv2.VideoCapture(str(video_dict.get(source_vid)))
            st_frame = st.empty()
            while vid_cap.isOpened():
                success, frame = vid_cap.read()
                if success:
                    annotated_frame = frame.copy()

                    results_detection = model_detection(frame)
                    results_segmentation = model_segmentation(frame)

                    annotated_frame= results_detection[0].plot(labels=False, boxes=False)
                    if results_segmentation[0].masks is not None :  # 'NoneType' object has no attribute 'data' error 방지용
                        masks = (results_segmentation[0].masks.data.cpu().numpy() * 255).astype(np.uint8)
                    else:
                        continue

                    resized_mask = cv2.resize(masks[0], (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
                    colored_mask = np.zeros_like(frame)
                    colored_mask[:, :, 1] = resized_mask  
                    
                    for r in results_detection:
                        boxes = r.boxes
                        for box in boxes:
                            b = np.array(box.xyxy[0]).astype(np.int32)
                            obj_mask = np.zeros((frame.shape[0], frame.shape[1], 1), dtype=np.uint8)
                            points = np.array([[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]])
                            fill_mask = cv2.fillPoly(obj_mask, np.int32([points]), color=(255, 0, 0))

                            com_img = cv2.bitwise_and(fill_mask, resized_mask)
                            
                            if com_img.any() and box.cls == 0:
                                cv2.rectangle(annotated_frame, (b[0], b[1]), (b[2], b[3]), (255, 0, 0), 2)
                                cv2.putText(annotated_frame, "no lifejacket", (b[0], b[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                    combined_frame = cv2.addWeighted(annotated_frame, 1, colored_mask, 0.5, 0)

                    _display_detected_frames2(st_frame, combined_frame)
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))

def _display_detected_frames2(st_frame, image):
    st_frame.image(image,
                   caption='Detected Video',
                   channels="BGR",
                   use_column_width=True
                   )
 

 