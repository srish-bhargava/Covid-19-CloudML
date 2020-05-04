# Start with pytorch image
FROM gw000/keras-full
#FROM ecoron/python36-sklearn
#FROM python:3

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
#ADD glove.6B.100d.txt /app/glove.6B.100d.txt
ADD Airlinetweets.csv /app/Airlinetweets.csv
ADD makeModel.py  /app/makeModel.py
#ADD svm_classifier.py /app/svm_classifier.py
ADD dictionary.json /app/dictionary.json

# Run the service
ENTRYPOINT [ "python3", "makeModel.py" ]
#ENTRYPOINT [ "python3", "svm_classifier.py" ]