'''
A deep learning project that uses a neural network built with PyTorch to recognize handwritten digits (0–9) from the MNIST dataset.
The model learns patterns from 60,000 training images, each measuring 28 × 28 pixels, and evaluates its ability to classify unseen handwritten digits using 10,000 test images.
The project explores image preprocessing, neural network architecture, model training, loss optimization, and classification accuracy to demonstrate how deep learning can solve image recognition tasks.
'''
import torchvision
import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

##DOWNLOAD MNIST DATASET
# Where the dataset will be stored
data_path = "./data"

# Download the training dataset
train_dataset = datasets.MNIST(
    root=data_path,
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

# Download the test dataset
test_dataset = datasets.MNIST(
    root=data_path,
    train=False,
    download=True,
    transform=transforms.ToTensor()
)

print("Training images:", len(train_dataset))
print("Test images:", len(test_dataset))

# Display one image
image, label = train_dataset[0]

print("Image shape:", image.shape)
print("Digit:", label)

# Display a grid of 10 images from the training dataset
def display_images(dataset, name):
    fig, axes = plt.subplots(2, 5, figsize=(10, 5))
    fig.suptitle(f"Sample Images from {name} Dataset")
    for i, ax in enumerate(axes.flat):
        image, label = dataset[i]

        ax.imshow(image.squeeze(0), cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"Label: {label}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()

display_images(train_dataset, "Training(before splitting)")

##SPLIT DATASET INTO TRAINING AND VALIDATION SETS
train_data, val_data = train_test_split(
    train_dataset,
    test_size=0.2,
    random_state=42
)

display_images(train_data, "Training(after splitting)")
display_images(val_data, "Validation")

##CREATING THE MODEL
#define the neural network architecture
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

val_loader = DataLoader(
    val_data,
    batch_size=64,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)

def model_structure_training_val():
    class DigitClassifier(nn.Module):
        def __init__(self):
            super().__init__()

            self.network = nn.Sequential(
                nn.Flatten(),
                nn.Linear(28 * 28, 196),
                nn.ReLU(),
                nn.Linear(196, 98),
                nn.Sigmoid(),
                nn.Linear(98, 49),
                nn.Tanh(),
                nn.Linear(49, 10)  # Output layer for 10 classes (digits 0-9)
            )

        def forward(self, x):
            return self.network(x)

    model = DigitClassifier()

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    #training the model
    epochs = 10
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")

    #evaluating the model on validation dataset
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    print(f"Accuracy on Validation Set: {accuracy:.2f}%")

    return model

model = model_structure_training_val()