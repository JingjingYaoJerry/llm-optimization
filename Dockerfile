# Specify the official PyTorch image from Docker Hub with the latest tag as the base image for a new Docker image, providing a pre-configured environment for deep learning.
FROM pytorch/pytorch:latest

# Set the working directory in the container to /app, which is where the code will be located and executed.
WORKDIR /app

# Install dependencies based on requirements.txt, preventing pip from saving downloaded packages in a cache which helps keep the final Docker image size small
RUN pip install --no-cache-dir -r requirements.txt

# Copy the contents of the current directory on the host machine to the /app directory in the container, making the code and necessary files available for execution within the container.
COPY . .

# Set the default command to run the finetune.py script when the container starts, which will execute the fine-tuning process for the language model.
CMD ["python", "src/finetune.py"]