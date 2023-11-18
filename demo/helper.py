from ultralytics import YOLO
import time
import streamlit as st
import cv2
from pytube import YouTube

import settings


def load_model(model_path):
    model = YOLO(model_path)
    return model




