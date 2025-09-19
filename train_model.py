import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm

# import modules
from load_dataset import ShapeDataset
from model import SimpleCNN

def train_model(dataset_path, batch_size, epochs, learning_rate, dropout, img_size, num_classes, num_images, checkpoint_path = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando o dispositivo: {device}")

    images_zip_path = os.path.join(dataset_path, "images.zip")
    labels_zip_path = os.path.join(dataset_path, "labels.zip")
    
    transform = transforms.Compose([
        #transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
    ])

    dataset = ShapeDataset(images_zip_path, labels_zip_path, transform=transform)
    n_total = len(dataset)
    n_train = int(0.8 * n_total)
    n_val = n_total - n_train
    train_dataset, val_dataset = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Instantiate the model   
    model = SimpleCNN(dropout=dropout, img_size=img_size, num_classes=num_classes).to(device)
    
    if checkpoint_path:
        if os.path.exists(checkpoint_path):
            print(f"Loading weights")
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        else:
            print(f"{checkpoint_path} not found. Starting from scratch")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    train_acc_list, val_acc_list = [], []

    epoch_bar = tqdm(range(epochs), desc="Treinamento")
    for epoch in epoch_bar:
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # A saída agora é [batch_size, num_classes], sem o .squeeze()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            
            # MUDA A FORMA DE CALCULAR A PREDIÇÃO
            # torch.max encontra o índice (a classe) com o maior valor
            _, predictions = torch.max(outputs, 1)
            
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        # validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predictions = torch.max(outputs, 1)
                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)

        # update matrics
        train_acc = 100 * correct / total
        val_acc = 100 * val_correct / val_total
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)
        scheduler.step()
        
        metrics = {
            'Train Acc': f"{train_acc:.2f}%",
            'Val Acc': f"{val_acc:.2f}%",
        }
        epoch_bar.set_postfix(metrics)

    # Lógica para salvar modelo e gráfico... (permanece similar)
    # ...
    filename_base = os.path.join(dataset_path, f"Validation_{n_total}_imgs_{epochs}_epochs")
    model_path = f"{filename_base}.pth"

    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Plot
    plot_path = f"{filename_base}_accuracy.png"
    plt.figure(figsize=(10, 6))
    plt.plot(train_acc_list, label="Train Accurancy", marker="o")
    plt.plot(val_acc_list, label="Val Accurancy", marker="s")
    plt.title(f"Accuracy per Epoch {num_images} imgs e {epochs}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)

    return val_acc_list[-1] if val_acc_list else 0.0