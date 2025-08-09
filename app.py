import streamlit as st
import cv2
import pytesseract
from PIL import Image
import numpy as np
import io

try:
    pytesseract.pytesseract.tesseract_cmd = r'/usr/local/bin/tesseract' # Adjust this path as needed
except FileNotFoundError:
    st.warning("Pytesseract executable not found. Please make sure Tesseract is installed and the path is correct.")

def generate_tikz_code(image):
    """
    This is the core function where the AI/CV logic would go.
    """
    try:
        # Convert the PIL image to a NumPy array for OpenCV
        open_cv_image = np.array(image.convert('RGB'))
        open_cv_image = open_cv_image[:, :, ::-1].copy() # Convert RGB to BGR

        # --- Image Preprocessing ---
        # Convert to grayscale
        gray_image = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)

        # Apply Canny edge detection
        edges = cv2.Canny(gray_image, 50, 150)

        # --- Simulate OCR and Diagram Logic (Placeholder) ---
        # In a real app, you would use more advanced image processing and ML models
        # to detect nodes, arrows, and text.
        text_from_image = pytesseract.image_to_string(gray_image)

        # Let's create a placeholder for the TikZ-cd code based on the OCR output
        st.write("### OCR Text Detected:")
        st.code(text_from_image)

        # The following is a hardcoded example of what a TikZ-cd diagram might look like.
        # This part would be dynamically generated in a full application.
        tikz_code = r"""
\documentclass{article}
\usepackage{tikz-cd}
\usepackage{amsmath}

\begin{document}

\begin{center}
\begin{tikzcd}
A \arrow[r, "f"] \arrow[d, "g"'] & B \arrow[d, "h"] \\
C \arrow[r, "k"'] & D
\end{tikzcd}
\end{center}

\end{document}
"""

        # Return the generated code and the edge-detected image for visualization
        return tikz_code, edges

    except Exception as e:
        st.error(f"An error occurred during image processing: {e}")
        return None, None

# --- Streamlit UI ---

st.set_page_config(page_title="TikZ-cd Generator", layout="centered")

st.title("Diagram to TikZ-cd Converter")
st.markdown("Upload an image of a commutative diagram and get the LaTeX code.")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Read the image file and convert to a PIL Image object
    image_bytes = io.BytesIO(uploaded_file.getvalue())
    pil_image = Image.open(image_bytes)

    # Display the uploaded image
    st.write("### Original Diagram")
    st.image(pil_image, caption="Uploaded Image", use_column_width=True)

    st.markdown("---")

    # Generate the TikZ-cd code and display the processed image
    with st.spinner("Processing image and generating code..."):
        tikz_output, processed_image = generate_tikz_code(pil_image)

    if tikz_output and processed_image is not None:
        st.write("### Processed Image (Edges Detected)")
        st.image(processed_image, caption="Computer Vision Output", use_column_width=True)

        st.markdown("---")
        
        st.write("### Generated TikZ-cd Code")
        st.code(tikz_output, language='latex')

        # Add a download button for the LaTeX code
        st.download_button(
            label="Download TikZ Code",
            data=tikz_output,
            file_name="diagram.tex",
            mime="text/plain"
        )
