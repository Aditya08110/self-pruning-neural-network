import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os

from model import PrunableNet, PrunableLinear
from utils import sparsity_loss, calculate_sparsity


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


transform = transforms.ToTensor()

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform
)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform
)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64)


model = PrunableNet().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()


lambda_sparse = 1e-2


for epoch in range(10):
    model.train()
    
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        
        cls_loss = criterion(outputs, labels)
        sp_loss = sparsity_loss(model)
        
        loss = cls_loss + lambda_sparse * sp_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch+1} done")


model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
sparsity = calculate_sparsity(model)

print(f"\nFinal Results:")
print(f"Accuracy: {accuracy:.2f}%")
print(f"Sparsity: {sparsity:.2f}%")


gates_all = []

for module in model.modules():
    if isinstance(module, PrunableLinear):
        gates = torch.sigmoid(module.gate_scores).detach().cpu().numpy().flatten()
        gates_all.extend(gates)


plt.figure()

plt.hist(gates_all, bins=50)
plt.title("Gate Value Distribution")
plt.xlabel("Gate Value")
plt.ylabel("Frequency")


save_path = os.path.join(os.getcwd(), "gate_distribution.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight')

print(f"Graph saved at: {save_path}")

# Show plot
plt.show()