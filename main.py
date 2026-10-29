'''
A deep learning project that uses a neural network built with PyTorch to recognize handwritten digits (0–9) from the MNIST dataset.
The model learns patterns from 60,000 training images, each measuring 28 × 28 pixels, and evaluates its ability to classify unseen handwritten digits using 10,000 test images.
The project explores image preprocessing, neural network architecture, model training, loss optimization, and classification accuracy to demonstrate how deep learning can solve image recognition tasks.
'''
from PIL import Image
import pandas as pd
from itertools import product
import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os  

##DOWNLOAD MNIST DATASET
# Where the dataset will be stored
data_path = "./data"

# Download the training dataset
MNIST_NORMALIZE = transforms.Normalize((0.1307,), (0.3081,))#Normalizing the pixels
train_transform = transforms.Compose([
    transforms.ToTensor(),
    MNIST_NORMALIZE,
])

train_dataset = datasets.MNIST(
    root=data_path,
    train=True,
    download=True,
    transform=train_transform
)

# Download the test dataset
test_dataset = datasets.MNIST(
    root=data_path,
    train=False,
    download=True,
    transform=train_transform
)

print("Training images:", len(train_dataset))
print("Test images:", len(test_dataset))

# Display one image
image, label = train_dataset[0]

print("Image shape:", image.shape)
print("Digit:", label)

#displaying all trhe digits with their counts and  a histogram to study the skewness
Y = train_dataset.targets

filtered_data = {}
for number in range(10):
    Y_filtered = Y[Y == number]
    digit_count = Y_filtered.size(0)

    filtered_data[number] = digit_count


plt.bar(filtered_data.keys(), filtered_data.values())#graph to see the counts
plt.title('Digit Count', fontsize=14, fontweight='bold')
plt.xlabel('Number of digits', fontsize=12)
plt.ylabel('Digits', fontsize=12)
plt.show()

plt.hist(filtered_data.values(), bins=10)#graph to analyze the skewness
plt.title('Skewness of digits', fontsize=14, fontweight='bold')
plt.xlabel('Number of digits', fontsize=12)
plt.ylabel('Digits', fontsize=12)
plt.show()

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
    random_state=42,
    stratify=train_dataset.targets
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

layer1_options = [196, 128, 98]
layer2_options = [98, 64, 32]

architectures = list(product(layer1_options, layer2_options))

for architecture in architectures:
    print(f"Architectures : {architecture}")

# creating function to find the best architecture for the model
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
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

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

    accuracy = round(accuracy, 2)
    val_loss = round(val_loss, 4)
    avg_loss = round(avg_loss, 4)
    return model, accuracy, val_loss, final_epoch, avg_loss, architectures[i]

#calling the function to find the best architecture for the model
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

excel_sheet = 'architecture_results.xlsx'
if os.path.exists(excel_sheet):
    print(f"File '{excel_sheet}' already exists.")
else:
    print(f"File '{excel_sheet}' does not exist. Creating a new file.")
    df.to_excel(excel_sheet, index=False)


activation_functions1 = {
    'ReLU': nn.ReLU(),
    'Sigmoid': nn.Sigmoid(),
    'Tanh': nn.Tanh()
}
activation_functions2 = {
    'ReLU': nn.ReLU(),
    'Sigmoid': nn.Sigmoid(),
    'Tanh': nn.Tanh()
}

act_funct_options = list(product(activation_functions1.values(), activation_functions2.values()))

for act_funct in act_funct_options:
    print(f"Activation Functions : {act_funct}")

# creating function to find the best combination of activation functions for the model with architecture 784 → 196 → 32 → 10 as it has the best val accuracy and val loss
def model_structure_training_val_2():
    class DigitClassifier(nn.Module):
        def __init__(self):
            super().__init__()

            self.network = nn.Sequential(
                nn.Flatten(),
                nn.Linear(28 * 28, 196),
                act_funct_options[i][0],
                nn.Linear(196, 32),
                act_funct_options[i][1],
                nn.Linear(32, 10), # Output layer for 10 classes (digits 0-9)
            )

        def forward(self, x):
            return self.network(x)

    model = DigitClassifier()

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

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

    print(f"Validation Accuracy for {act_funct_options[i][0]}, {act_funct_options[i][1]} activation functions: {accuracy:.2f}%")
    print(f"Validation Loss for {act_funct_options[i][0]}, {act_funct_options[i][1]} activation functions: {val_loss:.4f}")

    accuracy = round(accuracy, 2)
    val_loss = round(val_loss, 4)
    avg_loss = round(avg_loss, 4)
    return model, accuracy, val_loss, final_epoch, avg_loss, act_funct_options[i]

#calling the function to find the best activation functions for the model
accuracy_list = []
val_loss_list = []
final_epoch_list = []
activation_functions_list = []
training_loss_list = []
sr_no = range(1, len(act_funct_options) + 1)
for i in range(len(act_funct_options)):
    model, accuracy, val_loss, final_epoch, avg_loss, act_func = model_structure_training_val_2()
    act_func = str(act_func).replace("(", "").replace(")", "")
    accuracy_list.append(accuracy)
    val_loss_list.append(val_loss)
    final_epoch_list.append(final_epoch)
    activation_functions_list.append(act_func)
    training_loss_list.append(avg_loss)

data = {
    'Sr No': sr_no,
    'Activation Functions': activation_functions_list,
    'Validation Accuracy': accuracy_list,
    'Validation Loss': val_loss_list,
    'Training Loss': training_loss_list,
    'Epochs': final_epoch_list
}

df = pd.DataFrame(data)

excel_sheet = 'activation_functions_results.xlsx'
if os.path.exists(excel_sheet):
    print(f"File '{excel_sheet}' already exists.")
else:
    print(f"File '{excel_sheet}' does not exist. Creating a new file.")
    df.to_excel(excel_sheet, index=False)


epochs_total = [10, 20, 30]
#creating function to find the best number of epochs for the model
def model_structure_training_val_3(epochs):
    class DigitClassifier(nn.Module):
        def __init__(self):
            super().__init__()

            self.network = nn.Sequential(
                nn.Flatten(),
                nn.Linear(28 * 28, 196),
                nn.ReLU(),
                nn.Linear(196, 32),
                nn.Tanh(),
                nn.Linear(32, 10), # Output layer for 10 classes (digits 0-9)
            )

        def forward(self, x):
            return self.network(x)

    model = DigitClassifier()

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = epochs
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

        print(f"Validation Accuracy for {epochs} epochs: {accuracy:.2f}%")
        print(f"Validation Loss for {epochs} epochs: {val_loss:.4f}")

        accuracy = round(accuracy, 2)
    val_loss = round(val_loss, 4)
    avg_loss = round(avg_loss, 4)
    return model, accuracy, val_loss, final_epoch, avg_loss

#calling the function to find the best activation functions for the model
accuracy_list = []
val_loss_list = []
final_epoch_list = []
training_loss_list = []
sr_no = range(1, len(epochs_total) + 1)

for epochs in epochs_total:
    model, accuracy, val_loss, final_epoch, avg_loss = model_structure_training_val_3(epochs)
    accuracy_list.append(accuracy)
    val_loss_list.append(val_loss)
    final_epoch_list.append(final_epoch)
    training_loss_list.append(avg_loss)

data = {
    'Sr No': sr_no,
    'Validation Accuracy': accuracy_list,
    'Validation Loss': val_loss_list,
    'Training Loss': training_loss_list,
    'Epochs': final_epoch_list
}

df = pd.DataFrame(data)

excel_sheet = 'epochs_result.xlsx'
if os.path.exists(excel_sheet):
    print(f"File '{excel_sheet}' already exists.")
else:
    print(f"File '{excel_sheet}' does not exist. Creating a new file.")
    df.to_excel(excel_sheet, index=False)


learning_rates = [0.1, 0.001, 0.0005, 0.00001]
#creating function to find the best learning rate for the model
def model_structure_training_val_4(lr):
    class DigitClassifier(nn.Module):
        def __init__(self):
            super().__init__()

            self.network = nn.Sequential(
                nn.Flatten(),
                nn.Linear(28 * 28, 196),
                nn.ReLU(),
                nn.Linear(196, 32),
                nn.Tanh(),
                nn.Linear(32, 10), # Output layer for 10 classes (digits 0-9)
            )

        def forward(self, x):
            return self.network(x)

    model = DigitClassifier()

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    epochs = 20 #using epochs as 20 as it was the best number of epochs for the model
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

        print(f"Validation Accuracy for {epochs} epochs: {accuracy:.2f}%")
        print(f"Validation Loss for {epochs} epochs: {val_loss:.4f}")

        accuracy = round(accuracy, 2)
    val_loss = round(val_loss, 4)
    avg_loss = round(avg_loss, 4)
    return model, accuracy, val_loss, final_epoch, avg_loss

#calling the function to find the best activation functions for the model
accuracy_list = []
val_loss_list = []
final_epoch_list = []
training_loss_list = []
learning_rates_list = learning_rates
sr_no = range(1, len(learning_rates) + 1)

for lr in learning_rates:
    model, accuracy, val_loss, final_epoch, avg_loss = model_structure_training_val_4(lr)
    accuracy_list.append(accuracy)
    val_loss_list.append(val_loss)
    final_epoch_list.append(final_epoch)
    training_loss_list.append(avg_loss)

data = {
    'Sr No': sr_no,
    'Learning Rate' : learning_rates_list,
    'Validation Accuracy': accuracy_list,
    'Validation Loss': val_loss_list,
    'Training Loss': training_loss_list,
    'Epochs': final_epoch_list
}

df = pd.DataFrame(data)

excel_sheet = 'learning_rate_results.xlsx'
if os.path.exists(excel_sheet):
    print(f"File '{excel_sheet}' already exists.")
else:
    print(f"File '{excel_sheet}' does not exist. Creating a new file.")
    df.to_excel(excel_sheet, index=False)

'''
Creating final model with best selected metrics.
The final metrics:
1.Architecture : 196 → 32
2.Activation Functions : RelU() → Tanh()
3.Epochs : 20
4.Learning Rate: 0.0005
'''

class DigitClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 196),
            nn.ReLU(),
            nn.Linear(196,32),
            nn.Tanh(),
            nn.Linear(32,10)
        )

    def forward(self, x):
        return self.network(x)

model = DigitClassifier()

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

epochs = 20
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

    avg_loss = total_loss/len(train_loader)
    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")

#Evaluate on the validation set
model.eval()
correct = 0
total = 0
total_val_loss = 0

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

    print(f"Validation Accuracy for {epochs} epochs: {accuracy:.2f}%")
    print(f"Validation Loss for {epochs} epochs: {val_loss:.4f}")

#Evaluate on the test set
model.eval()
correct = 0
total = 0
total_val_loss = 0

with torch.inference_mode():
    for images, labels in test_loader:
        outputs = model(images)

        loss = loss_fn(outputs, labels)
        total_val_loss += loss.item() * labels.size(0)
        
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    accuracy = 100 * correct / total
    val_loss = total_val_loss / total

    print(f"Test Accuracy for {epochs} epochs: {accuracy:.2f}%")
    print(f"Test Loss for {epochs} epochs: {val_loss:.4f}")

#Predicting on test examples
image1 = Image.open("test_image1.png").convert("L")  # Convert to grayscale
image2 = Image.open("test_image2.png").convert("L")  # Convert to grayscale
image3 = Image.open("test_image3.png").convert("L")  # Convert to grayscale
# Resize the image to 28x28 pixels
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    MNIST_NORMALIZE
])

image1 = transform(image1).unsqueeze(0)  # Add batch dimension
image2 = transform(image2).unsqueeze(0)  # Add batch dimension
image3 = transform(image3).unsqueeze(0)  # Add batch dimension

images = [image1, image2, image3]

for i, image in enumerate(images):
    model.eval()
    with torch.inference_mode():
        output = model(image)
        probabilities = torch.softmax(output, dim=1)
        prediction = output.argmax(dim=1).item()

        confidence = probabilities[0][prediction].item() * 100
        print(f"Prediction for test image {i + 1}: {prediction} with confidence {confidence:.2f}%")
        print(f"Probabilities for test image {i + 1}: {probabilities}\n\n")
