import streamlit as st
from main import run_license_plate_recognition
import os


def app():
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.image(os.path.join("logos","Logo_inpt.PNG"), width=150)
    with col2:
        st.image(os.path.join("logos","ocr.png"), width=150)
    with col3:
        st.image(os.path.join("logos","yolo.png"), width=180)
    st.header("Vehicle License Plate Recognition")
    st.info("Work carried out by Taqi Anas and Ouchida Badreddine", icon="📃")
    st.write("Welcome!")


    with st.form("my_uploader"):
        uploaded_file = st.file_uploader(
            "Upload image", type=["png", "jpg", "jpeg"], accept_multiple_files=False
        )
        submit = st.form_submit_button(label="Process image")
    print("file name is ",uploaded_file)
    if uploaded_file is not None:
        # save uploaded image
        save_path = os.path.join("temp", uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

    if submit:
        # add spinner
        with st.spinner(text="Detecting license plate ..."):
            # display license plate as text
            text = run_license_plate_recognition(save_path).recognize_text()
            st.write(f"Detected License Plate Number: {text}")
            # show uploaded image with bounding box
            best_bb = run_license_plate_recognition(save_path).showBestPrediction()
            st.image(best_bb)


if __name__ == "__main__":
    app()
