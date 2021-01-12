#!/usr/bin/python3

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import pickle
import os
import librosa
from torch.utils.data.dataloader import default_collate

class ASVspoof2019(Dataset):
    def __init__(self, access_type, path_to_features, part='train', feature='LFCC', feat_len=750, pad_chop=True, padding='repeat', genuine_only=False):
        super(ASVspoof2019, self).__init__()
        self.access_type = access_type
        self.path_to_features = path_to_features
        self.part = part
        self.ptf = os.path.join(path_to_features, self.part)
        self.feat_len = feat_len
        self.feature = feature
        self.pad_chop = pad_chop
        self.padding = padding
        self.genuine_only = genuine_only
        if self.access_type == 'LA':
            self.tag = {"-": 0, "A01": 1, "A02": 2, "A03": 3, "A04": 4, "A05": 5, "A06": 6, "A07": 7, "A08": 8, "A09": 9,
                      "A10": 10, "A11": 11, "A12": 12, "A13": 13, "A14": 14, "A15": 15, "A16": 16, "A17": 17, "A18": 18,
                      "A19": 19}
        elif self.access_type == 'PA':
            self.tag = {"-": 0, "AA": 1, "AB": 2, "AC": 3, "BA": 4, "BB": 5, "BC": 6, "CA": 7, "CB": 8, "CC": 9}
        else:
            raise ValueError("Access type should be LA or PA!")
        self.label = {"spoof": 1, "bonafide": 0}
        self.all_files = librosa.util.find_files(os.path.join(self.ptf, self.feature), ext="pt")
        if self.genuine_only:
            assert self.access_type == "LA"
            num_bonafide = {"train": 2580, "dev": 2548, "eval": 7533}
            self.all_files = self.all_files[:num_bonafide[self.part]]

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        filepath = self.all_files[idx]
        basename = os.path.basename(filepath)
        all_info = basename.split(".")[0].split("_")
        assert len(all_info) == 6
        featureTensor = torch.load(filepath)
        this_feat_len = featureTensor.shape[1]
        if self.pad_chop:
            if this_feat_len > self.feat_len:
                startp = np.random.randint(this_feat_len - self.feat_len)
                featureTensor = featureTensor[:, startp:startp + self.feat_len, :]
            if this_feat_len < self.feat_len:
                if self.padding == 'zero':
                    featureTensor = padding_Tensor(featureTensor, self.feat_len)
                elif self.padding == 'repeat':
                    featureTensor = repeat_padding_Tensor(featureTensor, self.feat_len)
                else:
                    raise ValueError('Padding should be zero or repeat!')
        else:
            pass
        filename =  "_".join(all_info[1:4])
        tag = self.tag[all_info[4]]
        label = self.label[all_info[5]]
        return featureTensor, tag, label, this_feat_len

    def collate_fn(self, samples):
        if self.pad_chop:
            return default_collate(samples)
        else:
            feat_mat = [sample[0].transpose(0,1) for sample in samples]
            from torch.nn.utils.rnn import pad_sequence
            feat_mat = pad_sequence(feat_mat, True).transpose(1,2)
            tag = [sample[1] for sample in samples]
            label = [sample[2] for sample in samples]
            this_len = [sample[3] for sample in samples]

            return feat_mat, default_collate(tag), default_collate(label), default_collate(this_len)

class LibriGenuine(Dataset):
    def __init__(self, path_to_features, part='train', feature='LFCC', feat_len=750, padding='repeat'):
        super(LibriGenuine, self).__init__()
        self.ptf = path_to_features
        self.feat_len = feat_len
        self.feature = feature
        self.padding = padding

        self.all_files = librosa.util.find_files(os.path.join(self.ptf, self.feature), ext="pt")
        if part == 'train':
            self.all_files = self.all_files[:80000]
        elif part == 'dev':
            self.all_files = self.all_files[80000:]
        else:
            raise ValueError("Genuine speech should be added only in train or dev set!")
    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        filepath = self.all_files[idx]
        featureTensor = torch.load(filepath)
        this_feat_len = featureTensor.shape[1]
        if this_feat_len > self.feat_len:
            startp = np.random.randint(this_feat_len-self.feat_len)
            featureTensor = featureTensor[:, startp:startp+self.feat_len, :]
        if this_feat_len < self.feat_len:
            if self.padding == 'zero':
                featureTensor = padding_Tensor(featureTensor, self.feat_len)
            elif self.padding == 'repeat':
                featureTensor = repeat_padding_Tensor(featureTensor, self.feat_len)
            else:
                raise ValueError('Padding should be zero or repeat!')

        return featureTensor, 0, 0

    # def collate_fn(self, samples):
    #     return default_collate(samples)


def padding(spec, ref_len):
    width, cur_len = spec.shape
    assert ref_len > cur_len
    padd_len = ref_len - cur_len
    return torch.cat((spec, torch.zeros(width, padd_len, dtype=spec.dtype)), 1)

def padding_Tensor(spec, ref_len):
    _, cur_len, width = spec.shape
    assert ref_len > cur_len
    padd_len = ref_len - cur_len
    return torch.cat((spec, torch.zeros((1, padd_len, width), dtype=spec.dtype)), 1)

def repeat_padding(spec, ref_len):
    mul = int(np.ceil(ref_len / spec.shape[1]))
    spec = spec.repeat(1, mul)[:, :ref_len]
    return spec

def repeat_padding_Tensor(spec, ref_len):
    mul = int(np.ceil(ref_len / spec.shape[1]))
    spec = spec.repeat(1, mul, 1)[:, :ref_len, :]
    return spec



if __name__ == "__main__":
    path_to_features = '/data2/neil/ASVspoof2019LA/'  # if run on GPU
    training_set = ASVspoof2019("LA", path_to_features, 'train',
                                'LFCC', feat_len=750, pad_chop=False, padding='repeat')
    feat_mat, tag, label, this_len = training_set[2999]
    print(len(training_set))
    print(this_len)
    print(feat_mat.shape)
    print(tag)
    print(label)

    # samples = [training_set[26], training_set[27], training_set[28], training_set[29]]
    # out = training_set.collate_fn(samples)

    trainDataLoader = DataLoader(training_set, batch_size=32, shuffle=True, num_workers=0, collate_fn=training_set.collate_fn)
    feat_mat_batch, tags, labels, this_len = [d for d in next(iter(trainDataLoader))]
    print(feat_mat_batch.shape)
    print(this_len)
    # print(feat_mat_batch)


    # asvspoof = ASVspoof2019("LA", "/data2/neil/ASVspoof2019LA/", part='train', feature='LFCC', feat_len=750, padding='repeat', genuine_only=True)
    # print(len(asvspoof))
    # featTensor, tag, label = asvspoof[2579]
    # print(featTensor.shape)
    # # print(filename)
    # print(tag)
    # print(label)

    # libritts = LIBRITTS(root="/data/neil", download=True)
