# MNIST Handwritten Digit Recognition using PyTorch

A machine learning project that uses a neural network built with PyTorch to recognize handwritten digits from the MNIST dataset. The model learns patterns from grayscale images and classifies each image into one of ten digit classes (0–9).

## Overview

Handwritten digit recognition is a fundamental computer vision task and a practical introduction to neural networks and deep learning.

This project explores how a neural network processes image data, learns through backpropagation, updates its parameters using an optimizer, and evaluates its performance on unseen images.

## Features

- Loads and processes the MNIST handwritten digit dataset.
- Builds a neural network using PyTorch.
- Uses fully connected (`Linear`) layers and ReLU, Sigmoid, Tanh activation functions.
- Trains the model using backpropagation and the Adam optimizer.
- Uses Cross-Entropy Loss for multiclass classification.
- Evaluates predictions using test accuracy.
- Uses Softmax to convert output logits into class probabilities when needed.

## Dataset

The project uses the **MNIST dataset**, a collection of handwritten digit images.

- Training set: 60,000 images
- Test set: 10,000 images
- Image dimensions: 28 × 28 pixels
- Image format: Grayscale
- Number of classes: 10 (digits 0–9)

Each image is associated with a label representing the correct digit.

Dataset: [MNIST Database of Handwritten Digits](http://yann.lecun.com/exdb/mnist/)

## Neural Network Architecture

The model is implemented using PyTorch's `torch.nn` module.

The architecture consists of:

1. **Flatten:** Converts each 28 × 28 image into a vector of 784 values.
2. **Fully connected layer:** Transforms 784 input features into 196 features.
3. **ReLU activation:** Introduces nonlinearity.
4. **Fully connected layer:** Transforms 196 features into 98 features.
5. **Sigmoid activation:** Introduces another nonlinear transformation.
6. **Fully connected layer:** Transforms 98 features into 49 features.
7. **Tanh activation:** Processes the learned representation.
8. **Output layer:** Produces 10 scores corresponding to digits 0–9.

### Architecture Flow

```text
Input Image (28 × 28)
        |
      Flatten
        |
   Linear (784 → 196)
        |
       ReLU
        |
   Linear (196 → 98)
        |
     Sigmoid
        |
    Linear (98 → 49)
        |
       Tanh
        |
    Linear (49 → 10)
        |
   Output Logits
```

The output layer produces raw scores, called logits. The predicted digit is selected using the highest-scoring output.

## Training Process

The model is trained using the following procedure:

1. Load a batch of training images and their labels.
2. Pass the images through the neural network.
3. Calculate the loss using `nn.CrossEntropyLoss()`.
4. Calculate gradients using `loss.backward()`.
5. Update the model parameters using the Adam optimizer.
6. Repeat the process over multiple epochs.

### Loss Function

Cross-Entropy Loss measures how well the model's output scores correspond to the correct digit labels.

```python
loss_fn = nn.CrossEntropyLoss()
```

The model returns raw logits during training. Softmax is not applied before Cross-Entropy Loss because the loss function handles the required log-softmax computation internally.

### Optimizer

The Adam optimizer updates the trainable weights and biases based on the calculated gradients.

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)
```

The learning rate controls the scale of the optimizer's updates.

## Prediction and Softmax

For each image, the model produces ten logits, one for each possible digit.

The predicted class can be obtained using:

```python
predictions = outputs.argmax(dim=1)
```

When probabilities are required, Softmax can be applied to the logits:

```python
probabilities = torch.softmax(outputs, dim=1)
```

These probabilities sum to approximately 1 for each image. The largest probability corresponds to the predicted class.

## Model Evaluation

The trained model is evaluated on the test dataset, which contains images not used for training.

Test accuracy is calculated as:

$$ \text{Accuracy} = \frac{\text{Correct Predictions}}{\text{Total Predictions}} \times 100 $$


During evaluation, `model.eval()` switches the model to evaluation mode, while `torch.no_grad()` disables gradient tracking to reduce unnecessary computation.

The resulting test accuracy measures how often the model correctly classifies the test images.

## Technologies Used

- **Python** — Programming language
- **PyTorch** — Neural network implementation and training
- **Torchvision** — Dataset and image utilities, if used in the implementation
- **Matplotlib** — Image visualization, if used in the implementation
- **Scikit-Learn** — Split data into training and validation

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/sparshpatil13412/handwritten-digits-recognition
cd handwritten-digits-recognition
```

Replace the placeholders with your actual GitHub repository URL and folder name.

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Install only the packages required by your implementation.

### 4. Run the project

Run the Python script containing your dataset preparation, model definition, training loop, and evaluation code:

```bash
python main.py
```

Replace `main.py` with your actual script filename.

## Results

Record your measured results after training the model.

| Metric | Result |
|---|---|
| Number of training images | 48,000 |
| Number of validation images | 12,000 |
| Number of test images | 10,000 |
| Number of output classes | 10 |
| Number of training epochs | 10 |
| Validation accuracy | 99.62% |
| Test accuracy | --- |

The final test accuracy depends on the architecture, training configuration, preprocessing, and random initialization.

## Key Concepts Learned

- Tensors and image representations
- Neural network layers and trainable parameters
- Forward propagation
- Activation functions
- Multiclass classification
- Cross-Entropy Loss
- Backpropagation and gradients
- Optimization using Adam
- Softmax and class probabilities
- Model evaluation and generalization

## Future Improvements

- Visualize predictions alongside the original handwritten images.
- Analyze misclassified digits.
- Compare different network depths and hidden-layer sizes.
- Experiment with learning rates, batch sizes, and training epochs.
- Compare a fully connected neural network with a convolutional neural network (CNN).
- Save and reload trained model weights for inference.

## Author

**Sparsh Patil**

GitHub: [sparshpatil13412](https://github.com/sparshpatil13412)

## License

**MIT License**
