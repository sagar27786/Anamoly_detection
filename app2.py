import os
import easyocr
import cv2
import pandas as pd
import re
from PIL import Image, ImageDraw, ImageFont
from fpdf import FPDF
import os
import easyocr
import cv2
import pandas as pd
import re
import matplotlib.pyplot as plt
from natsort import natsorted
from pdf2image import convert_from_path
import os
os.system("chmod +x setup.sh && ./setup.sh")
# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=True)

# Function to extract text using EasyOCR
def extract_text_from_image(image_path):
    results = reader.readtext(image_path, detail=0)
    return " ".join(results)
# Function to convert PDF to images
def convert_pdf_to_images(pdf_path, output_folder, poppler_path):
    pages = convert_from_path(pdf_path=pdf_path, poppler_path=poppler_path)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    image_paths = []
    for i, page in enumerate(pages, start=1):
        img_name = os.path.join(output_folder, f"page-{i}.png")
        page.save(img_name, "PNG")
        image_paths.append(img_name)
    return image_paths
# Function to extract text with multiple patterns
def extract_with_multiple_patterns(text, patterns):
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return float(match.group(1))
    return None

# Function to parse numerical values from the extracted text
def parse_values(text):
    patterns = {
        "Test_ID": r"Appended:\s*([\w\d_.-]+)|(\d{2}\d{2}_\d{2})",
        "Ult SNR": [r"SNR:[:\s]*([\d.]+)"],
        "20dB SNR": [r"20 dB SNR @ ([\d.]+)", r"20 dB SNR ([\d.]+)"],
        "26dB SNR": [r"26 dB SNR @ ([\d.]+)", r"26 dB SNR ([\d.]+)"],
        "30dB SNR": [r"30 dB SNR @ ([\d.]+)", r"30 dB SNR ([\d.]+)"],
        "SNR @ 20dB": [r"20 dBu?V:?\s*([\d.]+)", r"20 dBp?V:?\s*([\d.]+)"],
        "THD": [r"% THD.?@:? ([\d.]+)"],
        "0dB audio": [r"dBr audio:?\s*([\d.]+)"],
        "-3dB audio": [r"dB audio.*?([\d.]+)"],
    }
    parsed_data = {}
    for key, field_patterns in patterns.items():
        if isinstance(field_patterns, list):
            parsed_data[key] = extract_with_multiple_patterns(text, field_patterns)
        else:
            match = re.search(field_patterns, text, re.IGNORECASE)
            parsed_data[key] = float(match.group(1)) if match else None
    expected_columns = {
        "Test_ID": "Test_ID",
        "Ult SNR": "Ult SNR [dB]",
        "20dB SNR": "20dB SNR [dBuV]",
        "26dB SNR": "26dB SNR [dBuV]",
        "30dB SNR": "30dB SNR [dBuV]",
        "SNR @ 20dB": "SNR @ 20dB?V: [dB]",
        "THD": "THD=1% [dBuV]",
        "0dB audio": "0dB audio [mV]",
        "-3dB audio": "-3dB audio [dBuV]",
    }
    aligned_data = {}
    for key, value in parsed_data.items():
        aligned_column = expected_columns.get(key, key)
        aligned_data[aligned_column] = value
    return aligned_data

def is_anomaly(row):
    """
    Checks if any value in the row is outside its predefined range
    and calculates deviation from the closest boundary of the range.
    Does not consider parameters with zero deviation as anomalies.
    """
    ranges = {
        "Ult SNR [dB]": (78, 3),
        "20dB SNR [dBuV]": (4.7, 0.3),
        "26dB SNR [dBuV]": (6, 0.3),
        "30dB SNR [dBuV]": (6.9, 0.3),
        "SNR @ 20dB?V: [dB]": (53, 3),
        "THD=1% [dBuV]": (7.5, 2),
        "0dB audio [mV]": (120, 20),
        "-3dB audio [dBuV]": (12.5, 0.5),
    }
    reasons = []

    # Check Test_ID mismatch
    if row["Test_ID"] != "03.18.01":
        reasons.append("Different Test_ID")
        return reasons

    # Check feature ranges and calculate deviation
    for feature, (base, tolerance) in ranges.items():
        if row[feature] is not None:
            lower_bound = base - tolerance
            upper_bound = base + tolerance

            if row[feature] < lower_bound:  # Below range
                deviation = row[feature] - lower_bound
                if deviation != 0:  # Exclude zero deviation
                    reasons.append(f"{feature} deviated by {deviation:.2f}")
            elif row[feature] > upper_bound:  # Above range
                deviation = row[feature] - upper_bound
                if deviation != 0:  # Exclude zero deviation
                    reasons.append(f"{feature} deviated by {deviation:.2f}")

    return reasons if reasons else None
def process_image(image_path):
    """
    Processes a single image: performs OCR, parses the extracted text,
    checks for anomalies, and returns the processed data.

    Args:
        image_path (str): Path to the image file.

    Returns:
        dict: Parsed data, anomaly status, and reasons.
    """
    try:
        # Read and process the image
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        image = cv2.resize(image, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)

        # Define ROI and crop regions
        x, y, w, h = 1600, 1900, 3000, 1000
        cropped_image = image[y:y+h, x:x+w]

        x, y, w, h = 500, 1780, 2000, 170
        cropped_image2 = image[y:y+h, x:x+w]

        # Perform OCR on cropped regions
        result = reader.readtext(cropped_image2)
        extracted_text2 = " ".join([detection[1] for detection in result])

        # Extract Test_ID using regex
        pattern = r"\d{2}\.\d{2}\.\d{2}"
        match = re.search(pattern, extracted_text2)
        extracted_value = match.group(0) if match else None

        # OCR on the main cropped image
        result = reader.readtext(cropped_image)
        extracted_text = " ".join([detection[1] for detection in result])

        # Parse extracted text
        parsed_data = parse_values(extracted_text)
        parsed_data["Test_ID"] = extracted_value
        parsed_data["Image Name"] = os.path.basename(image_path)

        # Perform anomaly detection
        anomaly_reasons = is_anomaly(pd.Series(parsed_data))
        anomaly_status = bool(anomaly_reasons)
        parsed_data["Anomaly"] = anomaly_status
        parsed_data["Reasons"] = ", ".join(anomaly_reasons) if anomaly_reasons else "No issues detected"

        return parsed_data
    except Exception as e:
        # Handle errors gracefully
        st.error(f"Error processing image {os.path.basename(image_path)}: {e}")
        return {
            "Image Name": os.path.basename(image_path),
            "Anomaly": None,
            "Reasons": f"Error: {e}",
        }

# Function to annotate the image
def annotate_image(image_path, anomaly_status, reasons, output_path):
    """
    Annotates the image with anomaly status and reasons.

    Args:
        image_path (str): Path to the original image.
        anomaly_status (bool): Whether an anomaly was detected.
        reasons (str): Explanation for the anomaly.
        output_path (str): Path to save the annotated image.
    """
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    # Define fonts
    font_path = "C:/Windows/Fonts/Arial.ttf"
    try:
        font_large = ImageFont.truetype(font_path, 50)
        font_small = ImageFont.truetype(font_path, 30)
    except IOError:
        font_large = font_small = ImageFont.load_default()

    # Add cross or tick
    symbol = "❌" if anomaly_status else "✔"
    symbol_color = "red" if anomaly_status else "green"
    draw.text((50, 50), symbol, font=font_large, fill=symbol_color)

    # Add status and reasons
    status_text = "Anomaly Detected" if anomaly_status else "No Anomaly"
    draw.text((150, 50), status_text, font=font_large, fill=symbol_color)
    y_position = 150
    for line in reasons.split(", "):
        draw.text((50, y_position), line, font=font_small, fill="black")
        y_position += 40

    # Save annotated image
    image.save(output_path)

# Function to create a PDF from annotated images
def create_pdf_from_images(image_folder, output_pdf_path):
    """
    Combines annotated images into a single PDF.

    Args:
        image_folder (str): Folder containing annotated images.
        output_pdf_path (str): Path to save the final PDF.
    """
    pdf = FPDF()
    image_files = natsorted([f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    for image_name in image_files:
        image_path = os.path.join(image_folder, image_name)
        pdf.add_page()
        pdf.image(image_path, x=10, y=10, w=190)  # Adjust for your image size
    pdf.output(output_pdf_path)
    print(f"PDF saved at: {output_pdf_path}")

# Function to process and annotate images
# def process_folder_with_annotations(folder_path, output_folder, output_pdf_path):
#     """
#     Processes all images in a folder, annotates them, and combines into a PDF.

#     Args:
#         folder_path (str): Folder containing original images.
#         output_folder (str): Folder to save annotated images.
#         output_pdf_path (str): Path to save the final annotated PDF.
#     """
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)

#     all_data = []  # To store results for all images

#     for image_name in os.listdir(folder_path):
#         image_path = os.path.join(folder_path, image_name)
#         if os.path.isfile(image_path) and image_path.endswith(('.png', '.jpg', '.jpeg')):
#             # Process each image
#             parsed_data = process_image(image_path)

#             # Annotate image
#             try:
#                 output_image_path = os.path.join(output_folder, f"annotated_{image_name}")
#                 annotate_image(
#                     image_path,
#                     parsed_data["Anomaly"],
#                     parsed_data["Reasons"],
#                     output_image_path,
#                 )
#                 parsed_data["Annotated Image Path"] = output_image_path
#             except Exception as e:
#                 st.error(f"Error annotating image {image_name}: {e}")
#                 parsed_data["Annotated Image Path"] = None

#             all_data.append(parsed_data)

#     # Combine annotated images into a single PDF
#     create_pdf_from_images(output_folder, output_pdf_path)

#     return all_data
import os
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
print(tf.__version__)

from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dense, Reshape, InputLayer, Flatten #to define encoder and decoder

from alibi_detect.od import OutlierAE, OutlierVAE
from alibi_detect.utils.visualize import plot_instance_score, plot_feature_outlier_image
from PIL import Image, ImageDraw, ImageFont
from alibi_detect.utils.saving import load_detector
# Initialize VAE model
vae_model_path = r"C:\Users\nxg11007\OneDrive - NXP\Documents\Image_anamoly\od_vae_grayscale_1256.weights"
vae_model = load_detector(vae_model_path)
vae_threshold = 0.008  # Reconstruction loss threshold
#confidence_threshold = 60.0  # Confidence score threshold for "Good" pattern

def crop_and_convert_to_grayscale(image_path):
    x, y, w, h = 320, 370, 1680, 790
    try:
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            return None
        cropped_image = image[y:y+h, x:x+w]
        gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        return gray_image
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def calculate_vae_metrics(gray_image, vae_model):
    try:
        resized_image = cv2.resize(gray_image, (256, 256), interpolation=cv2.INTER_AREA)
        input_image = resized_image.astype('float32') / 255.0  # Normalize
        input_image = input_image.reshape(1, 256, 256, 1)  # Ensure correct shape

        # Perform reconstruction using VAE
        recon_img = vae_model.vae(input_image).numpy()[0]

        # Predict instance score
        prediction = vae_model.predict(input_image)
        instance_score = prediction['data']['instance_score'][0]

        if np.isnan(instance_score):
            return None, None, None

        # Calculate confidence score
        confidence_score = max(0.0, min(1.0, 1 - (instance_score / vae_threshold))) * 100

        return instance_score, confidence_score, recon_img
    except Exception as e:
        return None, None, None

def annotate_image_with_vae(image_path, pattern_status, overall_status, output_path):
    """
    Annotates the image with VAE pattern detection results.
    
    Args:
        image_path (str): Path to the original image.
        reconstruction_loss (float): The reconstruction loss from VAE.
        pattern_status (str): "Good" or "Bad".
        overall_status (str): "Anomalous" (red) or "Non-Anomalous" (green).
        output_path (str): Path to save the annotated image.
    """
    try:
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)
        font_path = "C:/Windows/Fonts/Arial.ttf"

        try:
            font_large = ImageFont.truetype(font_path, 48)
            font_medium = ImageFont.truetype(font_path, 36)
            font_small = ImageFont.truetype(font_path, 28)
        except IOError:
            font_large = font_medium = font_small = ImageFont.load_default()

        # Define header positions
        x_start = 50
        y_start = 50
        row_height = 100
        col_spacing = 500  # Adjust spacing for better alignment

        # Define colors
        parametric_color = "green"  # Always green since it's only called if parameters are correct
        status_color = "red" if pattern_status == "Bad" else "green"  # Red for bad, green for good
        overall_status_color = "red" if overall_status == "Anomalous" else "green"  # Red if anomalous

        # Updated headers
        headers = ["Parametric Range", "Pattern detected by VAE", "Overall Status"]
        values = [
            "Correct",  # Always correct since this function is only called for good parameters
            # f"{reconstruction_loss:.6f}",  # Displaying Reconstruction Loss
            "Bad" if pattern_status == "Bad" else "Good",
            overall_status,
        ]

        # Define text colors
        value_colors = [
            parametric_color,  # Green for parametric correctness
            "blue",  # Reconstruction Loss in blue
            status_color,  # "Bad" in red, "Good" in green
            overall_status_color,  # "Anomalous" in red, "Non-Anomalous" in green
        ]

        # Draw headers and values dynamically
        for i, header in enumerate(headers):
            draw.text((x_start + i * col_spacing, y_start), header, font=font_medium, fill="black")  # Header
            draw.text((x_start + i * col_spacing, y_start + row_height), values[i], font=font_medium, fill=value_colors[i])  # Value

        # Save the annotated image (without reconstructed image)
        image.save(output_path)
        print(f"✅ VAE Annotated image saved to {output_path}")

    except Exception as e:
        print(f"❌ Error annotating image {image_path}: {e}")
def calculate_vae_reconstruction_loss(gray_image, vae_model):
    """
    Calculates VAE reconstruction loss and determines pattern status.

    Args:
        gray_image (numpy array): Grayscale image.
        vae_model: Loaded VAE model.

    Returns:
        float: Reconstruction loss.
        numpy array: Reconstructed image.
    """
    try:
        resized_image = cv2.resize(gray_image, (256, 256), interpolation=cv2.INTER_AREA)
        input_image = resized_image.astype('float32') / 255.0  # Normalize
        input_image = input_image.reshape(1, 256, 256, 1)  # Ensure correct shape

        # Perform reconstruction using VAE
        recon_img = vae_model.vae(input_image).numpy()[0]

        # Predict instance score
        prediction = vae_model.predict(input_image)
        reconstruction_loss = prediction['data']['instance_score'][0]  # Using only reconstruction loss

        # Check for NaN values
        if np.isnan(reconstruction_loss):
            print(f"Error: Reconstruction loss is NaN for the input image.")
            return None, None

        return reconstruction_loss, recon_img
    except Exception as e:
        print(f"Error calculating VAE reconstruction loss: {e}")
        return None, None
def process_folder_with_annotations(folder_path, output_folder, output_pdf_path, vae_model):
    """
    Processes images with correct parameters, checks patterns using VAE, and generates a final PDF.

    Args:
        folder_path (str): Folder containing original images.
        output_folder (str): Folder to save annotated images.
        output_pdf_path (str): Path to save the final annotated PDF.
        vae_model: Loaded VAE model for pattern detection.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    all_data = []  # To store results for all images

    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)
        if os.path.isfile(image_path) and image_path.endswith(('.png', '.jpg', '.jpeg')):
            # Process each image for parametric anomaly detection
            parsed_data = process_image(image_path)

            # Skip images with parametric anomalies
            if not parsed_data["Anomaly"]:
                # If parameters are correct, proceed with VAE pattern detection
                gray_image = crop_and_convert_to_grayscale(image_path)
                if gray_image is not None:
                    # Perform VAE-based pattern analysis (Only Reconstruction Loss)
                    reconstruction_loss, recon_img = calculate_vae_reconstruction_loss(gray_image, vae_model)

                    if reconstruction_loss is not None:
                        pattern_status = "Good" if reconstruction_loss <= 0.008 else "Bad"
                        overall_status = "Good" if pattern_status == "Good" else "Bad"

                        # Annotate the image with VAE results
                        output_image_path = os.path.join(output_folder, f"vae_annotated_{image_name}")
                        annotate_image_with_vae(
                            image_path, reconstruction_loss, pattern_status, overall_status, recon_img, output_image_path
                        )

                        # Update parsed data with VAE analysis
                        parsed_data["VAE Pattern"] = pattern_status
                        parsed_data["Reconstruction Loss"] = reconstruction_loss
                        parsed_data["Overall Status"] = overall_status
                        parsed_data["Annotated Image Path"] = output_image_path

                        # Mark as anomaly if the pattern is "Bad"
                        if pattern_status == "Bad":
                            parsed_data["Anomaly"] = True
                            parsed_data["Reasons"] += ", Bad pattern detected"
                    else:
                        parsed_data["Reasons"] += ", VAE pattern check failed"
                else:
                    parsed_data["Reasons"] += ", Grayscale conversion failed"

            # Append parsed data regardless of status
            all_data.append(parsed_data)

    # Generate a final PDF containing all annotated anomalous images
    create_pdf_from_images(output_folder, output_pdf_path)

    return all_data
import os
import streamlit as st
from tempfile import TemporaryDirectory
from pdf2image import convert_from_path
from pathlib import Path
import pandas as pd

# Set up the app layout and configuration
st.set_page_config(
    page_title="Anomaly Detection in Analog Radio",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar Logo Integration
logo_path = r"C:\Users\nxg11007\Downloads\pngwing.com.png"  # Replace with your logo's local file path
if Path(logo_path).is_file():
    st.sidebar.image(logo_path, use_container_width=True)
else:
    st.sidebar.warning("Logo file not found. Please check the path.")

# Sidebar Configuration Panel
st.sidebar.header("⚙️ Configuration Panel")
st.sidebar.write("Upload your PDF for anomaly detection.")

# Upload PDF file
uploaded_pdf = st.sidebar.file_uploader("📤 Upload a PDF file for anomaly detection:", type=["pdf"])



st.title("📄 Anomaly Detection in Analog Radio")

st.markdown("---")
import time
# Show file details and processing button
if uploaded_pdf:
    st.success(f"✅ PDF file '{uploaded_pdf.name}' uploaded successfully.")
    st.write(f"### File Details")
    st.write(f"- **File Name:** {uploaded_pdf.name}")
    #st.write(f"- File Type: {uploaded_pdf.type}")
    st.write(f"- **File Size:** {uploaded_pdf.size / 1024:.2f} KB")

    # Add processing button
# Add processing button
    if st.button("🔍 Start Processing"):
        with st.spinner("🚀 Initializing the processing pipeline....."):
            time.sleep(2)
            try:
                with TemporaryDirectory() as temp_dir:
                    # Save uploaded PDF temporarily
                    temp_pdf_path = os.path.join(temp_dir, uploaded_pdf.name)
                    with open(temp_pdf_path, "wb") as f:
                        f.write(uploaded_pdf.getbuffer())

                    # Convert PDF to images
                    st.info("📷 Converting PDF to images...")
                    image_folder = os.path.join(temp_dir, "images")
                    os.makedirs(image_folder, exist_ok=True)
                    images = convert_from_path(temp_pdf_path, poppler_path=r"C:\Users\nxg11007\OneDrive - NXP\Documents\Release-24.08.0-0\poppler-24.08.0\Library\bin")
                    image_paths = []
                    progress_bar = st.progress(0)  # Initialize the progress bar
                    conversion_progress = st.empty()  # Placeholder for dynamic text updates

                    for i, page in enumerate(images, start=1):
                        img_path = os.path.join(image_folder, f"page-{i}.png")
                        page.save(img_path, "PNG")
                        image_paths.append(img_path)

                        # Update progress and dynamic text
                        progress_percentage = int((i / len(images)) * 100)
                        progress_bar.progress(progress_percentage)  # Update the progress bar
                        conversion_progress.text(f"Converting page {i}/{len(images)} to image...")  # Dynamic text update

                        time.sleep(1)  # Smooth transition for each image conversion
                    time.sleep(4)

                    # Final success message and progress update
                    progress_bar.progress(100)  # Set progress bar to 100% on completion
                    conversion_progress.success(f"✅ All {len(image_paths)} pages converted to images successfully!")

                    # Process and annotate images
                    time.sleep(5)
                    st.info("📊 Detecting anomalies and annotating images...")
                    annotated_folder = os.path.join(temp_dir, "annotated_images_277")
                    os.makedirs(annotated_folder, exist_ok=True)

                    results = []
                    image_count = len(os.listdir(image_folder))

                    # Initialize progress bar and placeholder for dynamic updates
                    with st.spinner("🔍 Processing images for anomalies..."):
                        progress_bar = st.progress(0)
                        progress_placeholder = st.empty()

                        for idx, image_name in enumerate(os.listdir(image_folder), start=1):
                            image_path = os.path.join(image_folder, image_name)

                            if os.path.isfile(image_path) and image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                                try:
                                    # Process image for parametric anomaly detection
                                    parsed_data = process_image(image_path)

                                    if parsed_data["Anomaly"]:
                                        # If parametric anomaly detected, annotate image directly
                                        output_image_path = os.path.join(annotated_folder, f"annotated_{image_name}")
                                        annotate_image(image_path, True, parsed_data["Reasons"], output_image_path)
                                        parsed_data["Overall Status"] = "Anomalous"  # Marking parametric anomalies as anomalous

                                    else:
                                        # If parameters are correct, perform pattern checking using VAE
                                        gray_image = crop_and_convert_to_grayscale(image_path)
                                        if gray_image is not None:
                                            reconstruction_loss, _ = calculate_vae_reconstruction_loss(gray_image, vae_model)

                                            if reconstruction_loss is not None:
                                                pattern_status = "Good" if reconstruction_loss <= 0.008 else "Bad"
                                                overall_status = "Non-Anomalous" if pattern_status == "Good" else "Anomalous"
                                                reason = "Pattern detected by VAE is bad" if pattern_status == "Bad" else "Parameters are in range and Pattern detected by VAE is good"

                                                # 🔴 FIX: Ensure output_path is passed correctly
                                                output_image_path = os.path.join(annotated_folder, f"vae_annotated_{image_name}")
                                                annotate_image_with_vae(image_path, pattern_status, overall_status, output_image_path)

                                                # Update parsed data
                                                parsed_data["Overall Status"] = overall_status
                                                parsed_data["Reasons"] = reason
                                                parsed_data["Annotated Image Path"] = output_image_path  # Add annotated image path

                                                # Mark as anomaly if pattern is "Bad"
                                                if pattern_status == "Bad":
                                                    parsed_data["Anomaly"] = True
                                            else:
                                                parsed_data["Overall Status"] = "Anomalous"
                                                parsed_data["Reasons"] = "VAE pattern check failed"
                                        else:
                                            parsed_data["Overall Status"] = "Anomalous"
                                            parsed_data["Reasons"] = "Grayscale conversion failed"

                                    # ✅ FIX: Ensure anomaly is represented with "✅" in UI
                                    parsed_data["Anomaly"] = "✅" if parsed_data["Overall Status"] == "Anomalous" else ""

                                    results.append(parsed_data)
                                    # Update progress bar and message dynamically
                                    progress_percentage = int((idx / image_count) * 100)
                                    progress_bar.progress(progress_percentage)
                                    progress_placeholder.info(f"🔄 Processing image {idx}/{image_count}...")  # Dynamic update
                                    time.sleep(0.5)  # Smooth transition for processing
                                except Exception as e:
                                    st.error(f"❌ Error processing image {image_name}: {e}")

                    progress_placeholder.success("✅ All images processed successfully!")  # Final success message
                    time.sleep(3)  # Smooth transition after processing

                    # Generate the final annotated PDF
                    st.info("📑 Generating the final annotated PDF...")
                    final_pdf_path = os.path.join(temp_dir, "annotated_output_277.pdf")
                    create_pdf_from_images(annotated_folder, final_pdf_path)  # Call your PDF generation function
                    progress_bar.progress(100)  # Set progress bar to 100%
                    time.sleep(2)  # Simulate PDF generation time
                    st.success("📥 Final annotated PDF generated successfully!")
                    time.sleep(2)

                    # Display results in a table
                    # Display results in a table
# Display results in a table
                    st.write("### 📝 Anomaly Detection Results")
                    results_df = pd.DataFrame(results)

                    # Replace anomaly values with ✅ (tick) for anomalies and blank for non-anomalies
                    results_df["Anomaly"] = results_df["Anomaly"].apply(lambda x: "✅" if x else "")

                    # Display only required columns
                    st.dataframe(results_df[["Image Name", "Anomaly", "Reasons", "Overall Status"]])
                    time.sleep(2)  # Smooth transition

                    # Add a download button for the final PDF
                    with open(final_pdf_path, "rb") as pdf_file:
                        st.download_button(
                            label="📥 Download Annotated PDF",
                            data=pdf_file,
                            file_name="Annotated_Anomalies.pdf",
                            mime="application/pdf"
                        )
            except Exception as e:
                st.error(f"An error occurred during processing: {e}")
else:
    st.warning("📂 Please upload a PDF file to start the process.")