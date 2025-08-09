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

def call_gemini_api_for_tikz(api_key, content_list):
    """
    Makes a few-shot call to the Gemini API with the given content list
    using the google-generativeai SDK.
    """
    try:
        client = genai.Client(api_key=api_key)
                
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=content_list,
            config=types.GenerateContentConfig(thinking_config=types.ThinkingConfig(thinking_budget=-1) # Dynamic thinking
            ),
        )
        
        return response.text

    except Exception as e:
        st.error(f"An API error occurred: {e}")
        return None

def generate_tikz_code(image, api_key, progress_bar):
    """
    This is the core function demonstrating the few-shot LLM pipeline.
    It loads the example data, performs OCR, and then builds a multi-part
    prompt with the example data and the user's image.
    """
    if not api_key:
        st.error("Gemini API key is not set. Please add it to your Streamlit secrets.")
        return None

    try:
        # Load the few-shot example data inside the function
        try:
            examples_dir = "examples"
            example_image_path = os.path.join(examples_dir, "fiber_product.png")
            example_tikz_path = os.path.join(examples_dir, "fiber_product.txt")
            example_image = Image.open(example_image_path)
            with open(example_tikz_path, "r") as f:
                example_tikz_code = f.read()
        except FileNotFoundError:
            st.error(f"Few-shot example files '{os.path.basename(example_image_path)}' or '{os.path.basename(example_tikz_path)}' not found in the '{examples_dir}' folder. Please ensure they are there.")
            return None, None

        # Update progress bar
        progress_bar.progress(10, text="1. Preprocessing image...")

        # --- Image Preprocessing and OCR (on the user's image) ---
        open_cv_image = np.array(image.convert('RGB'))
        open_cv_image = open_cv_image[:, :, ::-1].copy()
        gray_image = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
        text_from_image = pytesseract.image_to_string(gray_image).strip()
        
        progress_bar.progress(40, text="2. Building few-shot prompt...")

        # --- Prepare the multi-part prompt for the LLM ---
        # This list of parts is the core of few-shot prompting.
        prompt_parts = [
            "You are an expert LaTeX typesetter specializing in commutative diagrams. Your task is to accurately translate diagrams into TikZ-cd code. Below is one example. Please follow its style and format for the subsequent image.",
            example_image,
            f"Here is the correct TikZ-cd LaTeX code for the above diagram:\n\n```latex\n{example_tikz_code}\n```\n\n",
            f"Now, based on this example and the OCR text from the new image below, generate the complete and correct TikZ-cd LaTeX code. The extracted text is: '{text_from_image}'\n\nEnsure the code is enclosed within a document class and includes the necessary packages. Make sure the diagram is centered. Do not add any extra explanations or text, just the full LaTeX code. Double check to make sure the code compiles correctly. If you cannot infer the diagram, provide a basic 2x2 diagram as a default.",
            image  # The user's uploaded image
        ]
        
        progress_bar.progress(70, text="3. Calling Gemini API...")

        # --- Call the Gemini API with the multi-part prompt ---
        tikz_output = call_gemini_api_for_tikz(api_key, prompt_parts)
        
        progress_bar.progress(100, text="Done!")

        return tikz_output

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