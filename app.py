import streamlit as st
import cv2
import pytesseract
from PIL import Image
import numpy as np
import io
from google import genai
from google.genai import types
import os
import time

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
        # Define the prompt for the user's image
        user_prompt_text = f"""
        You are an expert LaTeX typesetter specializing in beautiful and correct commutative diagrams. 
        Your task is to accurately translate the visual diagram into its tikz-cd representation, paying close attention to labels and arrow directions.

        The following text was extracted from an image of a commutative diagram:

        "{text_from_image}"

        Based on the image provided, generate the complete and correct TikZ-cd LaTeX code to reproduce the diagram.
        Ensure the code is enclosed within a document class and includes the necessary packages.
        Make sure the diagram is centered. Do not add any extra explanations or text, just the full LaTeX code.
        Double check to make sure the code compiles correctly.
        If you cannot infer the diagram, provide a basic 2x2 diagram as a default.
        """

        # --- Prepare a few-shot example ---
        try:
            example_image_path = "examples/fiber_product.png"
            example_image = Image.open(example_image_path)
            example_tikz_code = r"""\documentclass{article}
\usepackage{tikz-cd}
\begin{center}
\begin{tikzcd}
T
\arrow[drr, bend left, "x"]
\arrow[ddr, bend right, "y"]
\arrow[dr, dotted, "{(x,y)}" description] & & \\
& X \times_Z Y \arrow[r, "p"] \arrow[d, "q"]
& X \arrow[d, "f"] \\
& Y \arrow[r, "g"]
& Z
\end{tikzcd}
\end{center}
\end{document}"""

            contents = [
                {"role": "user", "parts": [
                    "Here is a simple example diagram and the correct tikz-cd code to reproduce it. Please follow this style and syntax.",
                    example_image
                ]},
                {"role": "model", "parts": [
                    example_tikz_code
                ]},
                {"role": "user", "parts": [
                    user_prompt_text,
                    image
                ]}
            ]
        except FileNotFoundError:
            st.warning(f"Example image '{example_image_path}' not found. Using a standard prompt.")
            contents = [{"role": "user", "parts": [user_prompt_text, image]}]


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
col1, col2 = st.columns(2,gap="large")

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