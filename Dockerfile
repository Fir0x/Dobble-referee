FROM python:3.8-slim-bullseye
 
RUN pip install --no-cache \
    joblib \
    numpy \
    opencv-python-headless \
    scikit-image \
    scikit-learn
 
WORKDIR /app/
 
CMD bash