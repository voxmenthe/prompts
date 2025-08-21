# Use an official Python image as a parent image
FROM python:3.11

# Set the working directory in the container
WORKDIR /app

# Copy requirements.txt and package.json to the working directory
COPY requirements.txt /app/

# Install Python and Node.js dependencies
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py /app/
COPY src /app/src/

# Make port 80 available to the world outside this container
EXPOSE 8501

# Command to run the application
CMD ["streamlit", "run", "app.py"]
