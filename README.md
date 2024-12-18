# Final Project - Convolutional Neural Networks
This is the repository for my CECS 456 final project, which was to build and train a deep learning model. I chose to use a convolutional neural network to analyze a dataset of chest X-rays, with the goal being to predict whether it was a normal X-ray, or the X-ray of someone diagnosed with pneumonia.  

## Testing the model:  
To use the model, just download the 'model.py' file and run it within your machine. The model automatically downloads the dataset onto your machine, so no extra setup is needed. Depending on your hardware, the epochs may take a while to run.  

### Before Use:
Before running the program for the first time, please import packages using the following commands so it can function:  
- pip install kagglehub
- pip install numpy
- pip install matplotlib
- pip install tensorflow
- pip install keras

After the model finishes training, it will display results on training accuracy and training loss, as well as its predictions on 20 images from the testing dataset.
