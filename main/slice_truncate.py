import os 
import torch
import numpy as np
import torch.nn.functional as F

LENGTH =[100, 200, 400, 600]


MODE = ['train', 'dev', 'eval']

for mode in MODE:
    for sequence_length in LENGTH:
        truncated_shape = (sequence_length, 40)
        print(f"Preparing {mode} data {sequence_length}") 
        scp_path = f"../data/mfcc/feature/ASVspoof2019_LA_{mode}/raw_mfcc_ASVspoof2019_LA_{mode}.1.scp"
        data = torch.load(f"data/{mode}/{mode}_data.t")                           
        label = torch.load(f"data/{mode}/{mode}_label.t")
        with open(scp_path, 'r') as f:                             
            feats_scp = [line.split()[0] for line in f] 

        X = []
        Y = []
        for k in data:
            if k in label: # ignore data that has no label
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

                for t in truncated_tensors:
                    X.append(t)
                    Y.append(label[k])


        print(f"Saving {mode} data") 
        torch.save(X, f"./data/{mode}/{sequence_length}/{mode}_{sequence_length}_data.t")
        torch.save(Y, f"./data/{mode}/{sequence_length}/{mode}_{sequence_length}_label.t")
