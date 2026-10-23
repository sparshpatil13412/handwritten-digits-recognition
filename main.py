'''
A deep learning project that uses a neural network built with PyTorch to recognize handwritten digits (0–9) from the MNIST dataset.
The model learns patterns from 60,000 training images, each measuring 28 × 28 pixels, and evaluates its ability to classify unseen handwritten digits using 10,000 test images.
The project explores image preprocessing, neural network architecture, model training, loss optimization, and classification accuracy to demonstrate how deep learning can solve image recognition tasks.
'''
import torchvision
from PIL import Image
import pandas as pd
from itertools import product
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

print("Training images after splitting:", len(train_data))
print("Validation images after splitting:", len(val_data))
display_images(train_data, "Training(after splitting)")
display_images(val_data, "Validation")


##CREATING THE MODEL
#define the neural network architecture
train_loader = DataLoader(
    train_data,
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

layer1_options = [196, 128]
layer2_options = [98, 64]

architectures = list(product(layer1_options, layer2_options))

for architecture in architectures:
    print(f"Architectures : {architecture}")

# creating function to find the best architecture, learning rate and number of epochs for the model
def model_structure_training_val_1():
    class DigitClassifier(nn.Module):
        def __init__(self):
            super().__init__()

            self.network = nn.Sequential(
                nn.Flatten(),
                nn.Linear(28 * 28, architectures[i][0]), #changing the number of neurons between 196, 128 to find the best architecture for the model
                nn.ReLU(),
                nn.Linear(architectures[i][0], architectures[i][1]),#changing the number of neurons between 98, 64 to find the best architecture for the model
                nn.Sigmoid(),
                nn.Linear(architectures[i][1], 10)  # Output layer for 10 classes (digits 0-9)
            )

        def forward(self, x):
            return self.network(x)

    model = DigitClassifier()

    loss_fn = nn.CrossEntropyLoss()
    #changing learning rate between 0.01, 0.001, 0.0005, 0.0001 for training the model
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) #0.001 was the suitable learning rate for the model as accuracy was the highest on the validation set with this learning rate

    #changing learning rate between 5, 10, 15, 20, 25, 30 for training the model
    epochs = 10 #30 was the suitable number of epochs for the model to converge and achieve good accuracy on the validation set
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

        if epoch == range(epochs)[-1]:
            final_epoch = epoch + 1

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")

    #evaluating the model on validation dataset on various epochs
    
    # Evaluate on the validation dataset
    model.eval()
    correct = 0
    total = 0
    total_val_loss = 0.0

    with torch.inference_mode():
        for images, labels in val_loader:
            outputs = model(images)

            loss = loss_fn(outputs, labels)
            total_val_loss += loss.item() * labels.size(0)

            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    val_loss = total_val_loss / total

    print(f"Validation Accuracy for {architectures[i][0]} → {architectures[i][1]}: {accuracy:.2f}%")
    print(f"Validation Loss for {architectures[i][0]} → {architectures[i][1]}: {val_loss:.4f}")

    return model, accuracy, val_loss, final_epoch, avg_loss, architectures[i]

#calling the function to find the best architecture, learning rate and number of epochs for the model
accuracy_list = []
val_loss_list = []
final_epoch_list = []
architectures_list = []
training_loss_list = []
sr_no = range(1, len(architectures) + 1)
for i in range(len(architectures)):
    model, accuracy, val_loss, final_epoch, avg_loss, arch = model_structure_training_val_1()
    arch = str(arch).replace("(", "").replace(")", "").replace(",", " →")
    accuracy_list.append(accuracy)
    val_loss_list.append(val_loss)
    final_epoch_list.append(final_epoch)
    architectures_list.append(arch)
    training_loss_list.append(avg_loss)

data = {
    'Sr No': sr_no,
    'Architecture': architectures_list,
    'Validation Accuracy': accuracy_list,
    'Validation Loss': val_loss_list,
    'Training Loss': training_loss_list,
    'Epochs': final_epoch_list
}
df = pd.DataFrame(data)
excel_sheet = df.to_excel('architecture_results.xlsx', index=False)
