FROM python:3.9-slim

WORKDIR /app

# The texlive-pictures package includes tikz-cd
RUN apt-get update && apt-get install -y --no-install-recommends \
    texlive-latex-base \
    texlive-pictures \
    texlive-fonts-recommended \
    pdftoppm \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt ./

# Install any needed Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application's source code
COPY . .

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Run the Streamlit app when the container launches
CMD ["streamlit", "run", "app.py"]
