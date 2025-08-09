import streamlit as st
import cv2
import pytesseract
from PIL import Image
import numpy as np
import io
import json
import base64
from google import genai

def image_to_base64(pil_image):
    """
    Converts a PIL Image object to a base64-encoded string.
    This is necessary to send the image data to the Gemini API.
    """
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str

def call_gemini_api_for_tikz(api_key, prompt, pil_image):
    """
    Makes a call to the Gemini API with the given prompt and image
    using the google-generativeai SDK.
    """
    # Configure the API key
    genai.configure(api_key=api_key)

    # Use the GenerativeModel class for a multimodal request
    model = genai.GenerativeModel('gemini-2.5-flash-preview-05-20')
    
    try:
        # Create a list of parts for the prompt, including text and image
        contents = [prompt, pil_image]
        
        response = model.generate_content(contents)
        
        # Return the generated text
        return response.text

    except Exception as e:
        st.error(f"An API error occurred: {e}")
        return None

def generate_tikz_code(image, api_key):
    """
    This is the core function demonstrating the LLM pipeline.
    It performs OCR and then builds a prompt for the LLM with the image data.
    """
    if not api_key:
        st.error("Gemini API key is not set. Please add it to your Streamlit secrets.")
        return None, None

    try:
        # Convert the PIL image to a NumPy array for OpenCV
        open_cv_image = np.array(image.convert('RGB'))
        open_cv_image = open_cv_image[:, :, ::-1].copy() # Convert RGB to BGR

        # --- Image Preprocessing and OCR ---
        gray_image = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
        text_from_image = pytesseract.image_to_string(gray_image).strip()
        
        # --- Prepare the prompt for the LLM ---
        prompt = f"""
        You are an expert in generating LaTeX code for commutative diagrams using the tikz-cd package.
        The following text was extracted from an image of a commutative diagram:

        "{text_from_image}"

        Based on the image provided, generate the complete and correct TikZ-cd LaTeX code to reproduce the diagram.
        Ensure the code is enclosed within a document class and includes the necessary packages.
        Make sure the diagram is centered. Do not add any extra explanations or text, just the full LaTeX code.
        If you cannot infer the diagram, provide a basic 2x2 diagram as a default.
        """
        
        st.write("### Generated Prompt for the LLM:")
        st.code(prompt)

        # --- Call the Gemini API ---
        st.info("Calling Gemini API to generate TikZ-cd code...")
        tikz_output = call_gemini_api_for_tikz(api_key, prompt, image)
        
        return tikz_output, open_cv_image

    except Exception as e:
        st.error(f"An error occurred during image processing or API call: {e}")
        return None, None

# --- Streamlit UI ---

st.set_page_config(page_title="Diagram to TikZ-cd Converter", layout="centered")

st.title("Diagram to TikZ-cd Converter")
st.markdown("Upload an image of a commutative diagram and get the LaTeX code, powered by the Gemini API.")

# Retrieve the API key from Streamlit secrets
if 'GEMINI_API_KEY' in st.secrets:
    api_key = st.secrets['GEMINI_API_KEY']
    st.success("Gemini API key loaded from secrets.")
else:
    api_key = None
    st.warning("Gemini API key not found in secrets. Please add it to your app's secrets.")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Read the image file and convert to a PIL Image object
    image_bytes = io.BytesIO(uploaded_file.getvalue())
    pil_image = Image.open(image_bytes)

    # Display the uploaded image
    st.write("### Original Diagram")
    st.image(pil_image, caption="Uploaded Image", use_container_width=True)

    st.markdown("---")

    # Generate the TikZ-cd code and display the processed image
    if st.button("Generate Code"):
        with st.spinner("Processing image and generating code..."):
            tikz_output, processed_image = generate_tikz_code(pil_image, api_key)

        if tikz_output and processed_image is not None:
            st.write("### Generated TikZ-cd Code")
            st.code(tikz_output, language='latex')

            # Add a download button for the LaTeX code
            st.download_button(
                label="Download TikZ Code",
                data=tikz_output,
                file_name="diagram.tex",
                mime="text/plain"
            )
