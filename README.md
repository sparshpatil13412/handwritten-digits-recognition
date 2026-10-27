# MNIST Handwritten Digit Recognition using PyTorch

A deep learning project that uses a fully connected neural network built with PyTorch to recognize handwritten digits from the MNIST dataset. The model processes grayscale images and classifies each image into one of ten digit classes.

## Overview

Handwritten digit recognition is a fundamental computer vision task and a practical introduction to neural networks and deep learning.

This project explores image preprocessing, neural network architecture, activation functions, backpropagation, optimization, and model evaluation. It also investigates how different architectural choices and training configurations affect classification performance.

## Features

- Loads and processes the MNIST handwritten digit dataset.
- Splits the training dataset into training and validation sets.
- Implements a fully connected neural network using PyTorch.
- Experiments with different hidden-layer sizes and activation functions.
- Evaluates different training durations and learning rates.
- Trains the model using backpropagation and the Adam optimizer.
- Uses Cross-Entropy Loss for multiclass classification.
- Evaluates performance using validation and test accuracy and loss.
- Generates predictions and class probabilities for handwritten images.
- Records hyperparameter experiment results in spreadsheet files.

## Dataset

The project uses the MNIST dataset, which contains grayscale images of handwritten digits.

- Training dataset: sixty thousand images.
- Test dataset: ten thousand images.
- Image dimensions: twenty-eight by twenty-eight pixels.
- Image format: grayscale.
- Output classes: digits from zero to nine.

Each image is associated with a label representing the correct digit.

Dataset: [MNIST Database of Handwritten Digits](https://yann.lecun.com/exdb/mnist/)

The original training dataset is split into training and validation subsets. The separate test dataset is retained for final model evaluation.

## Neural Network Architecture

The final model is implemented using PyTorch's `torch.nn` module.

The network consists of a flattening operation, two hidden fully connected layers, and an output layer.

- Flatten converts each image into a vector of pixel values.
- The first fully connected layer transforms the input features into a hidden representation.
- ReLU introduces nonlinearity after the first hidden layer.
- The second fully connected layer learns a more compact representation.
- Tanh introduces nonlinearity after the second hidden layer.
- The output layer produces raw scores for the possible digit classes.

### Architecture Flow

```text
Input Image
    |
  Flatten
    |
Fully Connected Layer
    |
   ReLU
    |
Fully Connected Layer
    |
   Tanh
    |
  Output Layer
    |
 Output Logits
```

The output layer produces logits rather than probabilities. The predicted digit is selected from the class with the highest logit.

## Training Process

The model is trained using mini-batches of images and their corresponding labels.

During training, images pass through the network to generate predictions. Cross-Entropy Loss measures the difference between the predicted scores and the correct labels. Backpropagation calculates gradients, and the Adam optimizer updates the model's trainable parameters.

This process repeats across multiple training epochs to improve classification performance.

## Loss Function

The project uses PyTorch's `CrossEntropyLoss` for multiclass classification.

```python
loss_fn = nn.CrossEntropyLoss()
```

The loss function operates directly on raw logits and integer class labels. An explicit Softmax operation is not required before calculating the loss.

## Optimizer

The Adam optimizer updates the model's weights and biases using the gradients calculated during backpropagation.

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0005
)
```

The learning rate controls the magnitude of parameter updates during training.

## Prediction and Softmax

The network generates logits for the possible digit classes. The predicted digit is selected using the highest-scoring output.

```python
prediction = output.argmax(dim=1)
```

When class probabilities are needed, Softmax converts the logits into a probability distribution.

```python
probabilities = torch.softmax(output, dim=1)
```

The probabilities indicate the model's relative preference for each class. They are useful for examining predictions, although a high probability does not guarantee that a prediction is correct.

## Model Evaluation

The model is evaluated on the validation dataset during experimentation and on the separate test dataset to assess its performance on unseen MNIST images.

Accuracy measures the proportion of correctly classified images, while Cross-Entropy Loss measures the quality of the model's predictions.

The model is placed in evaluation mode using `model.eval()`, and `torch.inference_mode()` disables gradient tracking during evaluation and prediction.

The project also tests the trained model on separate handwritten images to investigate how well it handles inputs outside the original MNIST dataset.

## Hyperparameter Experiments

The project explores several model and training configurations to understand their effects on classification performance.

Experiments include:

- Comparing different hidden-layer sizes.
- Comparing ReLU, Sigmoid, and Tanh activation functions.
- Evaluating different training durations.
- Comparing learning rates.
- Recording training loss, validation loss, and validation accuracy.

Experiment results are stored in spreadsheet files for comparison and analysis. The selected configuration is based on the observed validation results.

## Technologies Used

- Python for programming and model implementation.
- PyTorch for neural network construction, training, and inference.
- Torchvision for loading the MNIST dataset and preprocessing images.
- Matplotlib for visualizing images and dataset information.
- Scikit-learn for splitting the dataset into training and validation subsets.
- Pandas for organizing experimental results.
- OpenPyXL for exporting experiment results to Excel spreadsheets.

## Installation and Setup

Clone the repository and navigate to the project directory.

```bash
git clone https://github.com/sparshpatil13412/handwritten-digits-recognition.git
cd handwritten-digits-recognition
```

Create and activate a virtual environment.

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

Install the required dependencies.

```bash
pip install torch torchvision matplotlib scikit-learn pandas openpyxl
```

Run the Python script or notebook containing the project implementation. The MNIST dataset can be downloaded automatically by the dataset-loading code.

## Results

The project evaluates the model using training loss, validation loss, validation accuracy, test loss, and test accuracy.

The experimental results are recorded in spreadsheet files to compare model configurations and training settings.

The final performance should be reported using the actual results produced by the final model. Predictions on individually selected handwritten images are demonstrations and should not be treated as a substitute for evaluation on the complete test dataset.

## Key Concepts Learned

- Image preprocessing and tensor representations.
- Neural network architecture and trainable parameters.
- Forward propagation and backpropagation.
- Activation functions and nonlinear transformations.
- Multiclass classification.
- Cross-Entropy Loss.
- Gradient-based optimization using Adam.
- Learning rates and training epochs.
- Training, validation, and test datasets.
- Model generalization and classification errors.
- Logits, Softmax, and class probabilities.
- Hyperparameter experimentation and performance comparison.

## Future Improvements

- Visualize predictions alongside the corresponding handwritten images.
- Analyze misclassified examples to understand model limitations.
- Improve preprocessing for handwritten images from external sources.
- Compare fully connected networks with convolutional neural networks.
- Save and reload trained model weights for inference without retraining.
- Evaluate additional performance metrics and generalization techniques.

## Author

**Sparsh Patil**

GitHub: [sparshpatil13412](https://github.com/sparshpatil13412)

## License

This project is distributed under the MIT License.
