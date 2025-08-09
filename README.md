Diagram to TikZ-cd Converter
This project is a Streamlit web application that converts an image of a commutative diagram into its corresponding LaTeX code using the tikz-cd package. It leverages the power of multi-modal large language models (LLMs) with a robust pipeline that includes few-shot prompting and Retrieval-Augmented Generation (RAG).
✨ Features
Multi-modal AI: Uses the Gemini API to understand and process both the visual layout of the diagram and the text within the image.
Few-Shot Prompting: The model is guided by a set of pre-defined examples, ensuring a consistent and accurate output format.
Retrieval-Augmented Generation (RAG): The application retrieves relevant information from the official tikz-cd documentation to provide the model with up-to-date and authoritative context, making it highly effective for complex diagrams.
Optical Character Recognition (OCR): Employs pytesseract and OpenCV to extract text labels from the diagram image.
Streamlit Web Interface: Provides an easy-to-use graphical user interface for uploading images and viewing the generated code.
🚀 Prerequisites
Before running the application, you need to ensure you have the following:
Python 3.8+
Gemini API Key: A valid API key from Google AI Studio.
tikz-cd Documentation: The PDF file from the official documentation.
Tesseract OCR: The Tesseract OCR engine installed on your system.
📦 Setup & Installation
Clone the repository:
git clone <your-repo-url>
cd <your-repo-name>


Install Python dependencies:
pip install -r requirements.txt

(Note: You'll need to create a requirements.txt file containing the project's dependencies: streamlit, google-generativeai, opencv-python, pytesseract, Pillow, PyPDF2, scikit-learn).
Create your secrets file:
Create a .streamlit folder in your project's root directory. Inside this folder, create a secrets.toml file and add your Gemini API key:
[secrets]
GEMINI_API_KEY = "YOUR_API_KEY_HERE"


Place the documentation file:
Create a directory named docs in the project root and place the official tikz-cd-doc.pdf file inside it.
Set up example files:
Create an examples directory in the project root. This directory should contain the few-shot examples: fiber_product.png, fiber_product.txt, snake.png, snake.txt, cube.png, and cube.txt.
💻 How to Use
Run the Streamlit application:
streamlit run your_main_script_name.py


Access the web interface:
Open your web browser and navigate to the local URL provided by Streamlit (e.g., http://localhost:8501).
Upload a diagram:
Use the file uploader to select and upload an image file of a commutative diagram.
Generate the code:
Click the "Generate TikZ Code" button. The application will process the image, retrieve documentation, and generate the corresponding LaTeX code in a new section on the right side of the screen.
📁 Project Structure
your-project-name/
├── .streamlit/
│   └── secrets.toml        # Stores your API key
├── examples/
│   ├── cube.png
│   ├── cube.txt
│   ├── fiber_product.png
│   ├── fiber_product.txt
│   ├── snake.png
│   └── snake.txt
├── docs/
│   └── tikz-cd-doc.pdf     # Official documentation for RAG
├── requirements.txt        # Python dependencies
└── app.py  # The main application file


