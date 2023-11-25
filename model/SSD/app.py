# Python In-built packages
from pathlib import Path
import PIL

# External packages
import streamlit as st
from  detect import detect
# Local Modules
import settings
import helper

# Setting page layout
st.set_page_config(
    page_title="Mask Detection using SSD",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.title("Mask Detection using SSD")


st.sidebar.header("SSD Model Config")


st.sidebar.header("Detection")


confidence = float(st.sidebar.slider(
    "Select Model Confidence", 25, 100, 40)) / 100


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
            if source_img is None:
                default_image_path = str(settings.DEFAULT_IMAGE)
                default_image = PIL.Image.open(default_image_path)
                st.image(default_image_path, caption="Default Image",
                         use_column_width=True)
            else:
                uploaded_image = PIL.Image.open(source_img)
                original_image = uploaded_image.convert('RGB')
                st.image(source_img, caption="Uploaded Image",
                         use_column_width=True)
        except Exception as ex:
            st.error("Error occurred while opening the image.")
            st.error(ex)

    with col2:
        if source_img is None:
            default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
            default_detected_image = PIL.Image.open(
                default_detected_image_path)
            st.image(default_detected_image_path, caption='Detected Image',
                     use_column_width=True)
        else:
            if st.sidebar.button('Detect Objects'):
                
                image = detect(original_image, min_score=confidence, max_overlap=0.5, top_k=200,demo =1)
                st.image(image, caption='Detected Image',
                         use_column_width=True)
                boxes = ['no_mask','with_mask']
                try:
                    with st.expander("Detection Results"):
                        for box in boxes:
                            st.write(box)
                except Exception as ex:
                    # st.write(ex)
                    st.write("No image is uploaded yet!")
elif source_radio == settings.WEBCAM:
    helper.play_webcam(confidence)

else:
    st.error("Please select a valid source type!")
