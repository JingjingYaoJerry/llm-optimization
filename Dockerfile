# Specify the official PyTorch image from Docker Hub with the latest tag as the base image for a new Docker image, providing a pre-configured environment for deep learning.
FROM pytorch/pytorch:latest
# Install peft and transformers, preventing pip from saving downloaded packages in a cache which helps keep the final Docker image size small
RUN pip install --no-cache-dir peft transformers
