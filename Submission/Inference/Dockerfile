# Start with pytorch image
FROM gw000/keras-full

# Document who is responsible for this image
MAINTAINER Daniel Sabba <ds1952@nyu.edu>

# Expose any ports the app is expecting in the environment
ENV PORT 8001
EXPOSE $PORT

# Set up a working folder and install the pre-reqs
WORKDIR /app
ADD requirements.txt /app
RUN pip3 install -r requirements.txt

# Add the code as the last Docker layer because it changes the most
#ADD mnist-cloud/mnist-deploy.yaml /app/mnist-deploy.yaml
#ADD mnist-cloud/mnist-train-job.yaml /app/mnist-train-job.yaml
#ADD dictionary.json /app/dictionary.json
#ADD model.json /app/model.json
#ADD model.h5 /app/model.h5
ADD inference-script-snn-svm.py  /app/inference-script-snn-svm.py


# Run the service
ENTRYPOINT [ "python3", "inference-script-snn-svm.py" ]