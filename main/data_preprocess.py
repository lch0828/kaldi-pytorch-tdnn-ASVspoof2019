import torch
import os
import torchaudio.kaldi_io as kio

MODE = ['train', 'dev', 'eval']

for mode in MODE:
    print(f"Preparing {mode} data") 
    scp_path = f"../data/mfcc/feature/ASVspoof2019_LA_{mode}/raw_mfcc_ASVspoof2019_LA_{mode}.1.scp"
    ark_path = f"../data/mfcc/feature/ASVspoof2019_LA_{mode}/raw_mfcc_ASVspoof2019_LA_{mode}.1.ark"
    if mode == 'train':
        label_path = f"../data/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.{mode}.trn.txt"
    else:
        label_path = f"../data/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.{mode}.trl.txt"

    with open(scp_path, 'r') as f:
        feats_scp = [line.split()[0] for line in f]

    ark_file = dict(list(kio.read_mat_ark(ark_path)))

    data = {}
    for key in ark_file:
        data[os.path.split(key)[-1]] = ark_file[key]

    label = {}
    with open(label_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split(' ')

            if parts[4] == 'bonafide':
                label[parts[1]] = torch.tensor(1)
            else:
                label[parts[1]] = torch.tensor(0)


    print(f"Saving {mode} data") 
    torch.save(data, f"./data/{mode}/{mode}_data.t")
    torch.save(label, f"./data/{mode}/{mode}_label.t")
