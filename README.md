# MNIST-Digit-Prediction
Program in Jupyter Notebook that predicts the number handwritten in a 28x28 pixel image from the pixel color values.

MNIST Neural Network Classifier
This project is a simple neural network for classifying handwritten digits from the MNIST dataset. The implementation is done using Python and the Jupyter Notebook.

Steps for Execution
1. Environment Setup
Ensure you have the required libraries installed:
    numpy
    matplotlib
    scikit-learn

You can install these using pip:
    pip install numpy matplotlib scikit-learn

2. Fetch and Prepare the MNIST Dataset
    1. Import the necessary libraries.
    2. Fetch the MNIST dataset using fetch_openml.
    3. Convert mnist.data and mnist.target to integers.
    4. Verify the shape of the data and target.
    5. Print the current working directory.

3. Model Training
    1. Set the printing options for numpy arrays.
    2. Create a subset of the first 1000 images and normalize the data.
    3. Initialize the weights for the neural network.
    4. Define the activation functions relu, relu2deriv, and softmax.
    5. Define a function to calculate the prediction target.
    6. Train the model for 300 iterations:
        - For each row in the dataset, calculate the layers and errors.
        - Update the weights.
        - Print the error every 10 iterations.
    7. Save the trained weights to CSV files.

4. Model Testing
    Select Weights
        1. Specify the names of the weight files to be used for testing.

    Test Batch of Images / Accuracy Measurement
        1. Set the printing options for numpy arrays.
        2. Define the test offset and test size.
        3. Create a subset of the test images and normalize the data.
        4. Load the saved weights from the CSV files.
        5. Define the relu function.
        6. Test the model on the batch of images:
            - Calculate the layers and predicted class for each image.
            - Compare the predicted class with the correct answer.
            - Calculate and print the model accuracy.

5. Test 1 Image
    1. Set the printing options for numpy arrays.
    2. Specify the index of the test image.
    3. Create a subset of the test image and normalize the data.
    4. Load the saved weights from the CSV files.
    5. Define the relu function.
    6. Test the model on the single image:
        - Calculate the layers and predicted class for the image.
        - Compare the predicted class with the correct answer.
        - Print whether the prediction was correct or incorrect.
    7. Display the test image using Matplotlib.

Things to Be Careful With
    - Ensure the MNIST dataset is correctly fetched and converted to integers.
    - Verify the shapes of the data and target arrays.
    - Normalize the data before training and testing the model.
    - Save the trained weights after training and load them correctly for testing.
    - Check the paths and names of the weight files to avoid FileNotFoundError.
    - Display the test image correctly by reshaping it to (28, 28).

By following these steps, you can successfully train and test a neural network for classifying handwritten digits from the MNIST dataset.