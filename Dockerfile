# Start with pytorch image
FROM gw000/keras-full

# Document who is responsible for this image
MAINTAINER Srishti Bhargava <sb7261@nyu.edu>

# Expose any ports the app is expecting in the environment
ENV PORT 8001
EXPOSE $PORT

# Set up a working folder and install the pre-reqs
WORKDIR /app
ADD requirements.txt /app
RUN pip install -r requirements.txt

# Add the code as the last Docker layer because it changes the most
#ADD mnist-cloud/mnist-deploy.yaml /app/mnist-deploy.yaml
#ADD mnist-cloud/mnist-train-job.yaml /app/mnist-train-job.yaml
ADD glove.6B.100d.txt /app/glove.6B.100d.txt
ADD text_classification_dataset1.csv /app/text_classification_dataset1.csv
ADD rnn.py  /app/rnn.py


# Run the service
ENTRYPOINT [ "python3", "rnn.py" ]