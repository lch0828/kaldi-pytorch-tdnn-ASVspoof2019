import torch.nn as nn   
import torch
import math
from tdnn import TDNN 
from ftdnn import F_TDNN
import sys


feature_dim = int(float(sys.argv[1]))
window_size = int(float(sys.argv[2]))
model_type = str(sys.argv[3])


print('Defining model...')
if model_type == 'tdnn':
    model = TDNN(feature_dim, window_size)
elif model_type == 'ftdnn':
    model = F_TDNN(feature_dim)
model.cuda()

model.load_state_dict(torch.load(f"./model/{model_type}_{window_size}_model_weights.pt"))
model.eval()

print('Loading data...')
protocol_path = '../data/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt'
data = torch.load(f"data/eval/eval_data.t")

with open(protocol_path, 'r') as f:
    protocol_Y = [line.split()[1] for line in f]
with open(protocol_path, 'r') as f:
    labels = [line.split()[4] for line in f]

sequence_length = window_size
truncated_shape = (sequence_length, 40)

print('Evaling...')
cm_score = []
for k, l in zip(protocol_Y, labels):
    original_tensor = data[k]
    dim = original_tensor.shape[0]
    
    num_tensors = int(dim / sequence_length)
    truncated_tensors = []
    for i in range(num_tensors):
        start_idx = i * truncated_shape[0]
        end_idx = (i + 1) * truncated_shape[0]
        truncated_tensor = original_tensor[start_idx:end_idx, :]
        truncated_tensors.append(truncated_tensor)

    remaining = original_tensor[num_tensors*truncated_shape[0]:, :]
    last_tensor = torch.zeros(truncated_shape)
    last_tensor[:remaining.shape[0], :] = remaining
    truncated_tensors.append(last_tensor)

    score_total = 0
    with torch.no_grad():
        for inputs in truncated_tensors:
            inputs = torch.unsqueeze(inputs, dim=0)
            inputs = inputs.cuda()
            outputs = model(inputs)
            score_total += outputs.tolist()[0][0]
    
    score_total /= num_tensors + 1
    if score_total == 0.0:
        score_total = 1.0e-20 
    elif score_total == 1.0:
        score_total = 0.99999
    cm_score.append(f"{k} - {l} {math.log((score_total) / (1 - score_total))}")

print('Writing score...')
with open(f"./cm_score/{model_type}_{window_size}_cm_score.txt", 'w') as f:
    for line in cm_score:
        f.write(line+'\n')

