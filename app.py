import streamlit as st
import cv2
import pytesseract
from PIL import Image
import numpy as np
import io
from google import genai
from google.genai import types
import os

examples_dir = "examples"

# --- Gemini API Call Function ---
def call_gemini_api_for_tikz(api_key, content_list):
    """
    Makes a few-shot call to the Gemini API with the given content list
    using the google-genai SDK.
    """
    try:
        client = genai.Client(api_key=api_key)
                
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=content_list,
            config=types.GenerateContentConfig(thinking_config=types.ThinkingConfig(thinking_budget=-1) # Dynamic thinking,
``            ),
        )
        
        return response.text

    except Exception as e:
        st.error(f"An API error occurred: {e}")
        return None

def build_few_shot_prompt(text_from_image, image, examples):
    """Builds the multi-part prompt for the Gemini API."""
    prompt_parts = [
        f"You are an expert LaTeX typesetter specializing in commutative diagrams. Below are a few examples to guide your style and format. Please follow them for the subsequent image. The extracted text from the new image is: '{text_from_image}'"
    ]
    
    for example_image, example_tikz_code in examples:
        prompt_parts.append(example_image)
        prompt_parts.append(f"Here is the correct TikZ-cd LaTeX code for the above diagram:\n\n```latex\n{example_tikz_code}\n```\n\n")

    prompt_parts.append(f"Based on these examples, generate the complete and correct TikZ-cd LaTeX code for the new image below. Ensure the code is enclosed within a document class and includes the necessary packages. Do not add any extra explanations or text, just the full LaTeX code. Pay close attention to the arrow styles (e.g., solid, dashed, double-headed) and the overall shape of the diagram (e.g., square, triangle, cube). Check to make sure the code compiles correctly.")
    prompt_parts.append(image)
    return prompt_parts

def generate_tikz_code(image, api_key, progress_bar, examples):
    """
    This is the core function demonstrating the few-shot pipeline.
    It performs OCR and then builds a multi-part prompt with the
    example data and the user's image.
    """

    try:
        # Update progress bar
        progress_bar.progress(10, text="1. Preprocessing image...")

        # --- Image Preprocessing and OCR (on the user's image) ---
        try:
            open_cv_image = np.array(image.convert('RGB'))
            open_cv_image = open_cv_image[:, :, ::-1].copy()
            gray_image = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
            
            _, binary_image = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY)
            
            # Use the OCR text to inform the prompt
            text_from_image = pytesseract.image_to_string(binary_image, config='--psm 6').strip()
            
        except Exception as e:
            st.error(f"Error during OCR processing: {e}")
            text_from_image = ""

        progress_bar.progress(40, text="2. Building few-shot prompt...")
        prompt_parts = build_few_shot_prompt(text_from_image, image, examples)
        
        progress_bar.progress(70, text="3. Calling Gemini API...")

        tikz_output = call_gemini_api_for_tikz(api_key, prompt_parts)
        
        progress_bar.progress(100, text="Done!")

        return tikz_output

    except Exception as e:
        st.error(f"An error occurred during image processing or API call: {e}")
        return None, None

@st.cache_data
def load_examples(example_names, examples_dir):
    examples = []
    try:
        for name in example_names:
            image_path = os.path.join(examples_dir, f"{name}.png")
            tikz_path = os.path.join(examples_dir, f"{name}.txt")
            example_image = Image.open(image_path)
            with open(tikz_path, "r") as f:
                example_tikz_code = f.read()
            examples.append((example_image, example_tikz_code))
        return examples
    except FileNotFoundError:
        st.error(f"One or more few-shot example files not found in the '{examples_dir}' folder. Please ensure all files for {example_names} are there.")
        st.stop()

# --- Streamlit UI ---
st.set_page_config(page_title="Diagram to TikZ-cd Converter", layout="centered")
st.title("Diagram to TikZ-cd Converter")
st.markdown("Upload an image of a commutative diagram and get the LaTeX code, powered by the Gemini API with few-shot prompting.")

# Retrieve the API key from Streamlit secrets
if 'GEMINI_API_KEY' in st.secrets:
    api_key = st.secrets['GEMINI_API_KEY']
    st.success("Gemini API key loaded from secrets.")
else:
    api_key = None
    st.warning("Gemini API key not found in secrets. Please add it to your app's secrets.")
    st.stop()

#Define the examples to use for few-shot prompting
examples_dir = "examples"
example_names = ['fiber_product', 'snake', 'cube']
examples = load_examples(example_names, examples_dir)

# --- Main App Layout ---
# Use columns for a cleaner layout
col1, col2 = st.columns([2, 2], gap="large")

with col1:
    st.write("### 1. Upload Diagram")
    uploaded_file = st.file_uploader(
        "Choose an image of a commutative diagram...",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_file is not None:
        # To read file as bytes:
        image_bytes = io.BytesIO(uploaded_file.getvalue())
        pil_image = Image.open(image_bytes)
        
        st.image(pil_image, caption="Uploaded Diagram", use_container_width=True)

        # Store the uploaded image in the session state
        st.session_state.uploaded_image = pil_image

if st.button("Generate TikZ Code", disabled=(st.session_state.get('uploaded_image') is None)):
    if 'uploaded_image' in st.session_state:
        pil_image = st.session_state.uploaded_image
        progress_bar = st.progress(0, text="Starting...")

        # Generate the TikZ code
        tikz_output = generate_tikz_code(pil_image, api_key, progress_bar, examples)

        if tikz_output:
            st.session_state.tikz_output = tikz_output
            # Render the LaTeX code to an image
            st.session_state.rendered_image = render_latex(tikz_output)
        else:
            st.session_state.tikz_output = None
            st.session_state.rendered_image = None

        progress_bar.empty()

with col2:
    st.write("### 2. Generated Code")
    if 'tikz_output' in st.session_state and st.session_state.tikz_output:
        st.code(st.session_state.tikz_output, language='latex')
    else:
        st.info("The generated LaTeX code will appear here.")