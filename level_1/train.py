import sys
import os
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from shared.dataset import get_flowers102_dataloaders
from shared.visualization import plot_training_curves

def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, num_epochs=10, device='cuda'):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    history = {
        'train_loss': [], 'val_loss': [], 
        'train_acc': [], 'val_acc': []
    }

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                # Track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())

            # Deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, history

def main():
    # Setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Data Loading
    print("Initializing DataLoaders...")
    dataloaders, dataset_sizes, class_names = get_flowers102_dataloaders(batch_size=32)
    
    # 2. Model Setup (ResNet50 Baseline)
    print("Setting up ResNet50 baseline...")
    model = models.resnet50(pretrained=True)
    
    # Freeze all layers generally for baseline? 
    # The prompt says "Like using ResNet50 transfer learning".
    # Usually for Level 1 baseline, we freeze backbone and train head.
    for param in model.parameters():
        param.requires_grad = False
        
    # Replace last layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 102) # Flowers102 has 102 classes
    model = model.to(device)
    
    # 3. Training Config
    criterion = nn.CrossEntropyLoss()
    # Optimizer for only the final layer
    optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)
    
    # 4. Train
    print("Starting training...")
    # 5-10 epochs should be enough for a baseline check
    model, history = train_model(model, dataloaders, dataset_sizes, criterion, optimizer, num_epochs=10, device=device)
    
    # 5. Save Results
    os.makedirs('results', exist_ok=True)
    torch.save(model.state_dict(), 'models/level_1_baseline.pth')
    plot_training_curves(history, save_path='results/level_1_curves.png')
    
    # 6. Final Test Evaluation
    print("Running final evaluation on Test set...")
    model.eval()
    running_corrects = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloaders['test']:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
            total += labels.size(0)
            
    test_acc = running_corrects.double() / total
    print(f'Level 1 Baseline Test Accuracy: {test_acc:.4f}')

    # Check Pass Condition
    if test_acc >= 0.85:
        print("✅ Level 1 Requirement Met (Accuracy >= 85%)")
    else:
        print("⚠️ Level 1 Requirement Not Met. Consider more epochs or unfreezing layers.")

if __name__ == '__main__':
    main()
