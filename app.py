import streamlit as st
import streamlit.components.v1 as components
import cv2
import pytesseract
from PIL import Image
import numpy as np
import io
import base64
from google import genai
from google.genai import types
import os
import time

def image_to_base64(pil_image):
    """
    Converts a PIL Image object to a base64-encoded string.
    This is necessary to send the image data to the Gemini API.
    """
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str

def call_gemini_api_for_tikz(api_key, prompt, pil_image, config):
    """
    Makes a call to the Gemini API with the given prompt and image
    using the google-genai SDK.
    """
    try:
        # Create a client instance first
        client = genai.Client(api_key=api_key)

        # Create a list of parts for the prompt, including text and image
        contents = [prompt, pil_image]
        
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config=config
        )
        
        # Return the generated text
        return response.text

    except Exception as e:
        st.error(f"An API error occurred: {e}")
        return None

def extract_tikz_only(full_latex_code):
    """
    Extracts only the tikzcd environment from the full LaTeX code.
    """
    start_tag = "\\begin{tikzcd}"
    end_tag = "\\end{tikzcd}"
    
    start_index = full_latex_code.find(start_tag)
    end_index = full_latex_code.find(end_tag) + len(end_tag)
    
    if start_index != -1 and end_index != -1:
        return full_latex_code[start_index:end_index]
    
    return "Error: TikZ-cd environment not found."

def generate_tikz_code(image, api_key, progress_bar):
    """
    This is the core function demonstrating the LLM pipeline.
    It performs OCR and then builds a prompt for the LLM with the image data.
    """
    if not api_key:
        st.error("Gemini API key is not set. Please add it to your Streamlit secrets.")
        return None, None
    
    # We will also add the API key to environment variables so that it can be picked up by the SDK.
    os.environ['GOOGLE_API_KEY'] = api_key

    # Define the generation configuration with thinking_config
    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=0)
    )

    try:
        # Update progress bar
        progress_bar.progress(10, text="1. Preprocessing image...")

        # Convert the PIL image to a NumPy array for OpenCV
        open_cv_image = np.array(image.convert('RGB'))
        open_cv_image = open_cv_image[:, :, ::-1].copy() # Convert RGB to BGR

        # --- Image Preprocessing and OCR ---
        gray_image = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
        text_from_image = pytesseract.image_to_string(gray_image).strip()
        
        progress_bar.progress(40, text="2. Extracting text and features...")

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
        
        progress_bar.progress(70, text="3. Calling Gemini API...")

        # --- Call the Gemini API ---
        tikz_output = call_gemini_api_for_tikz(api_key, prompt, image, config)
        
        progress_bar.progress(100, text="Done!")

        return tikz_output

    except Exception as e:
        st.error(f"An error occurred during image processing or API call: {e}")
        return None

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

# Use columns for a side-by-side layout with a small spacer column
col1, col_spacer, col2 = st.columns([1, 0.1, 1])

with col1:
    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        # Read the image file and convert to a PIL Image object
        image_bytes = io.BytesIO(uploaded_file.getvalue())
        pil_image = Image.open(image_bytes)

        st.write("### Original Diagram")
        st.image(pil_image, caption="Uploaded Image", use_container_width=True)
        
        # Use a button to trigger code generation and set a session state
        if st.button("Generate TikZ Code"):
            st.session_state.show_output = True
            st.session_state.tikz_output = None # Clear previous output
            st.session_state.pil_image = pil_image # Store image for the next run

with col2:
    if 'show_output' in st.session_state and st.session_state.show_output and uploaded_file is not None:
        # Check if output has already been generated
        if 'tikz_output' not in st.session_state or st.session_state.tikz_output is None:
            progress_bar = st.progress(0, text="Starting...")
            # Call the generation function and store the result
            st.session_state.tikz_output = generate_tikz_code(st.session_state.pil_image, api_key, progress_bar)
            # Clear the progress bar after completion
            time.sleep(1)
            progress_bar.empty()

        if st.session_state.tikz_output is not None:
            st.write("### Generation Complete!")
            
            st.write("#### Generated TikZ-cd Code")
            st.code(st.session_state.tikz_output, language='latex')
            
            # Button to copy code to clipboard
            if st.button("Copy TikZ Code"):
                code_to_copy = extract_tikz_only(st.session_state.tikz_output)
                # Create a simple JavaScript command to copy the text
                js_copy = f"""
                <script>
                    navigator.clipboard.writeText(`{code_to_copy}`);
                    alert('Code copied to clipboard!');
                </script>
                """
                components.html(js_copy)

