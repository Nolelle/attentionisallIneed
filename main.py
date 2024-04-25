import torch
import torch.nn as nn
import torch.optim as optim

from model import Transformer
from dataset import MyDataset
from torch.utils.data import DataLoader

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model
    model = Transformer().to(device)
    
    # Prepare data
    train_dataset = MyDataset(train=True)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    for epoch in range(10):
        model.train()
        for data, targets in train_loader:
            data, targets = data.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, targets)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            print(f'Epoch [{epoch+1}/10], Loss: {loss.item():.4f}')

    print("Training complete.")

if __name__ == "__main__":
    main()
