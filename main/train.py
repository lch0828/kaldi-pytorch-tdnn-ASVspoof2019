import torch
import torch.nn as nn
import torch.optim as optim
from dataset import MFCCDataset
from tdnn import TDNN
from ftdnn import F_TDNN
import sys


feature_dim = int(float(sys.argv[1]))
window_size = int(float(sys.argv[2]))
model_type = str(sys.argv[3])

if torch.cuda.is_available(): 
    print('GPU available')
    device = "cuda:0" 
else: 
    device = "cpu" 

print('Loading dataset...')
dataset = MFCCDataset(f"data/train/{window_size}/train_{window_size}_data.t", f"data/train/{window_size}/train_{window_size}_label.t")
data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

dev_dataset = MFCCDataset(f"data/dev/{window_size}/dev_{window_size}_data.t", f"data/dev/{window_size}/dev_{window_size}_label.t")
val_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

print('Defining model...')
if model_type == 'tdnn':
    model = TDNN(feature_dim, window_size)
elif model_type == 'ftdnn':
    model = F_TDNN(feature_dim)
model.cuda()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.BCELoss()

train_losses = []
dev_losses = []
min_dev_loss = 1000.0
print('Training...')
for epoch in range(10):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(data_loader, 0):
        optimizer.zero_grad()

        inputs = inputs.cuda()
        labels = labels.cuda()
        #labels = torch.nn.functional.one_hot(labels, num_classes=2).float()
        labels = torch.unsqueeze(labels, 1).float()

        # forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # backward pass and optimization step
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:    # print every 100 mini-batches
            print('[Epoch %d, Batch %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 100))
            train_losses.append(running_loss)
            running_loss = 0.0

            model.eval() # set model to evaluation mode
            with torch.no_grad():
                val_loss = 0.0
                for j, (val_inputs, val_labels) in enumerate(val_loader, 0):
                    val_inputs = val_inputs.cuda()
                    val_labels = val_labels.cuda()
                    val_labels = torch.unsqueeze(val_labels, 1).float()
                    val_outputs = model(val_inputs)
                    val_loss += criterion(val_outputs, val_labels).item()
                val_loss /= len(val_loader)
                dev_losses.append(val_loss)
            print('val loss: ', val_loss)
            model.train() # set model back to training mode

            if i == 99 or val_loss < min_dev_loss:
                torch.save(model.state_dict(), f"./model/{model_type}_{window_size}_model_weights.pt")
                min_dev_loss = val_loss

with open(f"./loss/{model_type}_{window_size}_train_loss.txt", 'w') as f:
    for line in train_losses:
        f.write(f"{line}\n")
with open(f"./loss{model_type}_{window_size}_dev_loss.txt", 'w') as f:
    for line in dev_losses:
        f.write(f"{line}\n")
