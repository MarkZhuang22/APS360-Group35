import time
import streamlit as st
import cv2
from  detect import detect
from PIL import Image

import settings


def _display_detected_frames(conf, st_frame, image):
  
    #image = cv2.resize(image, (720, int(720*(9/16))))
    pil_image = Image.fromarray(image)
    original_image = pil_image.convert('RGB')
    out_image = detect(original_image, min_score=conf, max_overlap=0.5, top_k=200, demo =1)

    st_frame.image(out_image,
                   caption='Detected Video',
                   channels="BGR",
                   use_column_width=True
                   )


def play_webcam(conf):
  
    source_webcam = settings.WEBCAM_PATH

    if st.sidebar.button('Detect Objects'):
        try:
            vid_cap = cv2.VideoCapture(source_webcam)
            if not vid_cap.isOpened():
                print("Failed to open webcam.")
            else:
                print("Webcam opened successfully.")
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                             st_frame,
                                             image
                                             )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))


