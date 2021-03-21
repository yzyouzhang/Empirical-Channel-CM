#!/usr/bin/python3

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import pickle
import os
import librosa
from feature_extraction import LFCC
from torch.utils.data.dataloader import default_collate

lfcc = LFCC(320, 160, 512, 16000, 20, with_energy=False)
wavform = torch.Tensor(np.expand_dims([0]*3200, axis=0))
lfcc_silence = lfcc(wavform)
silence_pad_value = lfcc_silence[:,0,:].unsqueeze(0)

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
            if self.part in ["train", "dev"]:
                num_bonafide = {"train": 2580, "dev": 2548}
                self.all_files = self.all_files[:num_bonafide[self.part]]
            else:
                res = []
                for item in self.all_files:
                    if "bonafide" in item:
                        res.append(item)
                self.all_files = res
                assert len(self.all_files) == 7355

    def __len__(self):
        return len(self.all_files)
        # return 4000

    def __getitem__(self, idx):
        filepath = self.all_files[idx]
        basename = os.path.basename(filepath)
        all_info = basename.split(".")[0].split("_")
        # assert len(all_info) == 6
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
                elif self.padding == 'silence':
                    featureTensor = silence_padding_Tensor(featureTensor, self.feat_len)
                else:
                    raise ValueError('Padding should be zero or repeat!')
        else:
            pass
        filename =  "_".join(all_info[1:4])
        tag = self.tag[all_info[4]]
        label = self.label[all_info[5]]
        return featureTensor, filename, tag, label, 2019

    def collate_fn(self, samples):
        if self.pad_chop:
            return default_collate(samples)
        else:
            # feat_mat = [sample[0].transpose(0,1) for sample in samples]
            # from torch.nn.utils.rnn import pad_sequence
            # feat_mat = pad_sequence(feat_mat, True).transpose(1,2)
            max_len = max([sample[0].shape[1] for sample in samples]) + 1
            feat_mat = [repeat_padding_Tensor(sample[0], max_len) for sample in samples]

            tag = [sample[1] for sample in samples]
            label = [sample[2] for sample in samples]
            # this_len = [sample[3] for sample in samples]

            # return feat_mat, default_collate(tag), default_collate(label), default_collate(this_len)
            return default_collate(feat_mat), default_collate(tag), default_collate(label)

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

        return featureTensor, 'name', 0, 0

    # def collate_fn(self, samples):
    #     return default_collate(samples)

class VCC2020(Dataset):
    def __init__(self, path_to_features="/data2/neil/VCC2020/", feature='LFCC', feat_len=750, pad_chop=True, padding='repeat', genuine_only=False):
        super(VCC2020, self).__init__()
        self.ptf = path_to_features
        self.feat_len = feat_len
        self.feature = feature
        self.pad_chop = pad_chop
        self.padding = padding
        self.genuine_only = genuine_only
        self.tag = {"-": 0, "SOU": 20, "T01": 21, "T02": 22, "T03": 23, "T04": 24, "T05": 25, "T06": 26, "T07": 27, "T08": 28, "T09": 29,
                    "T10": 30, "T11": 31, "T12": 32, "T13": 33, "T14": 34, "T15": 35, "T16": 36, "T17": 37, "T18": 38, "T19": 39,
                    "T20": 40, "T21": 41, "T22": 42, "T23": 43, "T24": 44, "T25": 45, "T26": 46, "T27": 47, "T28": 48, "T29": 49,
                    "T30": 50, "T31": 51, "T32": 52, "T33": 53, "TAR": 54}
        self.label = {"spoof": 1, "bonafide": 0}
        self.all_files = librosa.util.find_files(os.path.join(self.ptf, self.feature), ext="pt")

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        filepath = self.all_files[idx]
        basename = os.path.basename(filepath)
        all_info = basename.split(".")[0].split("_")
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
                elif self.padding == 'silence':
                    featureTensor = silence_padding_Tensor(featureTensor, self.feat_len)
                else:
                    raise ValueError('Padding should be zero or repeat!')
        else:
            pass
        tag = self.tag[all_info[-2]]
        label = self.label[all_info[-1]]
        # assert label == 1
        return featureTensor, basename, tag, label, 2020

    def collate_fn(self, samples):
        return default_collate(samples)


class ASVspoof2015(Dataset):
    def __init__(self, path_to_features, part='train', feature='LFCC', feat_len=750, pad_chop=True, padding='repeat', genuine_only=False):
        super(ASVspoof2015, self).__init__()
        self.path_to_features = path_to_features
        self.part = part
        self.ptf = os.path.join(path_to_features, self.part)
        self.feat_len = feat_len
        self.feature = feature
        self.pad_chop = pad_chop
        self.padding = padding
        self.tag = {"human": 0, "S1": 1, "S2": 2, "S3": 3, "S4": 4, "S5": 5,
                    "S6": 6, "S7": 7, "S8": 8, "S9": 9, "S10": 10}
        self.label = {"spoof": 1, "human": 0}
        self.all_files = librosa.util.find_files(os.path.join(self.ptf, self.feature), ext="pt")

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        filepath = self.all_files[idx]
        basename = os.path.basename(filepath)
        all_info = basename.split(".")[0].split("_")
        assert len(all_info) == 4
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
                elif self.padding == 'silence':
                    featureTensor = silence_padding_Tensor(featureTensor, self.feat_len)
                else:
                    raise ValueError('Padding should be zero or repeat!')
        else:
            pass
        filename =  all_info[1]
        tag = self.tag[all_info[-2]]
        label = self.label[all_info[-1]]
        return featureTensor, filename, tag, label, 2015

    def collate_fn(self, samples):
        if self.pad_chop:
            return default_collate(samples)
        else:
            # feat_mat = [sample[0].transpose(0,1) for sample in samples]
            # from torch.nn.utils.rnn import pad_sequence
            # feat_mat = pad_sequence(feat_mat, True).transpose(1,2)
            max_len = max([sample[0].shape[1] for sample in samples]) + 1
            feat_mat = [repeat_padding_Tensor(sample[0], max_len) for sample in samples]

            tag = [sample[1] for sample in samples]
            label = [sample[2] for sample in samples]
            # this_len = [sample[3] for sample in samples]

            # return feat_mat, default_collate(tag), default_collate(label), default_collate(this_len)
            return default_collate(feat_mat), default_collate(tag), default_collate(label)


class ASVspoof2019LAtrain_withChannel(Dataset):
    def __init__(self, channel, path_to_features='/dataNVME/neil/ASVspoof2019LAChannel/', feature='LFCC', feat_len=750, pad_chop=True, padding='repeat', genuine_only=False):
        super(ASVspoof2019LAtrain_withChannel, self).__init__()
        self.channel = channel
        self.path_to_features = path_to_features
        self.ptf = path_to_features
        self.feat_len = feat_len
        self.feature = feature
        self.pad_chop = pad_chop
        self.padding = padding
        self.genuine_only = genuine_only
        self.tag = {"-": 0, "A01": 1, "A02": 2, "A03": 3, "A04": 4, "A05": 5, "A06": 6, "A07": 7, "A08": 8, "A09": 9,
                  "A10": 10, "A11": 11, "A12": 12, "A13": 13, "A14": 14, "A15": 15, "A16": 16, "A17": 17, "A18": 18,
                  "A19": 19}
        self.label = {"spoof": 1, "bonafide": 0}
        self.channel = {'amr[br=5k15]': 0, 'amrwb[br=15k85]': 1, 'g711[law=u]': 2, 'g722[br=56k]': 3,
                        'g722[br=64k]': 4, 'g726[law=a,br=16k]': 5, 'g728': 6, 'g729a': 7, 'gsmfr': 8,
                        'silk[br=20k]': 9, 'silk[br=5k]': 10, 'silkwb[br=10k,loss=5]': 11, 'silkwb[br=30k]': 12}
        all_files = librosa.util.find_files(self.ptf, ext="pt")
        self.all_files = list(filter(lambda x: channel in x, all_files))
        if self.genuine_only:
            pass

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        filepath = self.all_files[idx]
        basename = os.path.basename(filepath)
        all_info = basename.split(".")[0].split("_")
        # assert len(all_info) == 7
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
                elif self.padding == 'silence':
                    featureTensor = silence_padding_Tensor(featureTensor, self.feat_len)
                else:
                    raise ValueError('Padding should be zero or repeat!')
        else:
            pass
        filename =  "_".join(all_info[1:4])
        tag = self.tag[all_info[4]]
        label = self.label[all_info[5]]
        channel = self.channel[all_info[6]]
        return featureTensor, filename, tag, label, channel

    def collate_fn(self, samples):
        if self.pad_chop:
            return default_collate(samples)

class ASVspoof2019LAtrain_plusChannel(Dataset):
    def __init__(self, access_type, path_to_features, part='train', feature='LFCC', feat_len=750,
                 pad_chop=True, padding='repeat', genuine_only=False):
        super(ASVspoof2019LAtrain_plusChannel, self).__init__()
        self.pad_chop = pad_chop
        self.asvspoof2019 = ASVspoof2019(access_type, path_to_features, part,
                               feature, feat_len, pad_chop, padding, genuine_only)
        self.asvspoof2019channel = ASVspoof2019LAtrain_withChannel(access_type, feature='LFCC', feat_len=750,
                 pad_chop=True, padding='repeat', genuine_only=False)

    def __len__(self):
        return len(self.asvspoof2019) + len(self.asvspoof2019channel)

    def __getitem__(self, idx):
        if idx < len(self.asvspoof2019):
            featureTensor, filename, tag, label = self.asvspoof2019[idx]
            channel = 13
        else:
            featureTensor, filename, tag, label, channel = self.asvspoof2019channel[idx - len(self.asvspoof2019)]
        return featureTensor, filename, tag, label, channel

    def collate_fn(self, samples):
        if self.pad_chop:
            return default_collate(samples)


class ASVspoof2019LAtrain_resilient(Dataset):
    def __init__(self, path_to_features="/data2/neil/ASVspoof2019LA/", path_to_channeled="/dataNVME/neil/ASVspoof2019LAChannel/", feature='LFCC', feat_len=750, pad_chop=True, padding='repeat', genuine_only=False):
        super(ASVspoof2019LAtrain_resilient, self).__init__()
        self.path_to_features = path_to_features
        self.path_to_channeled = path_to_channeled
        self.path_to_features = path_to_features
        self.ptf = os.path.join(path_to_features, "train")
        self.feat_len = feat_len
        self.feature = feature
        self.pad_chop = pad_chop
        self.padding = padding
        self.genuine_only = genuine_only
        self.tag = {"-": 0, "A01": 1, "A02": 2, "A03": 3, "A04": 4, "A05": 5, "A06": 6, "A07": 7, "A08": 8, "A09": 9,
                    "A10": 10, "A11": 11, "A12": 12, "A13": 13, "A14": 14, "A15": 15, "A16": 16, "A17": 17, "A18": 18,
                    "A19": 19}
        self.label = {"spoof": 1, "bonafide": 0}
        self.channel = ['amrwb[br=15k85]', 'g722[br=56k]',
                        'g722[br=64k]', 'silkwb[br=10k,loss=5]', 'silkwb[br=30k]']
        self.original_all_files = librosa.util.find_files(os.path.join(self.ptf, self.feature), ext="pt")
        channeled_all_files = librosa.util.find_files(self.path_to_channeled, ext="pt")
        self.channeled_all_files = [list(filter(lambda x: channelx in x, channeled_all_files)) for channelx in self.channel]

    def __len__(self):
        return len(self.original_all_files)

    def __getitem__(self, idx):
        filepath = self.original_all_files[idx]
        basename = os.path.basename(filepath)
        all_info = basename.split(".")[0].split("_")
        featureTensor = torch.load(filepath)
        rand = np.random.randint(0, 5)
        filepath_channeled = self.channeled_all_files[rand][idx]
        featureTensor_channeled = torch.load(filepath_channeled)
        this_feat_len = max(featureTensor.shape[1], featureTensor_channeled.shape[1])
        # print(this_feat_len)
        # print(featureTensor_channeled.shape[1])
        if not featureTensor.shape[1] == featureTensor_channeled.shape[1]:
            if featureTensor.shape[1] < featureTensor_channeled.shape[1]:
                featureTensor = silence_padding_Tensor(featureTensor, featureTensor_channeled.shape[1])
            else:
                featureTensor_channeled = silence_padding_Tensor(featureTensor_channeled, featureTensor.shape[1])
            # print(self.channel[rand])

        if self.pad_chop:
            if this_feat_len > self.feat_len:
                startp = np.random.randint(this_feat_len - self.feat_len)
                featureTensor = featureTensor[:, startp:startp + self.feat_len, :]
                featureTensor_channeled = featureTensor_channeled[:, startp:startp + self.feat_len, :]
            if this_feat_len < self.feat_len:
                if self.padding == 'zero':
                    featureTensor = padding_Tensor(featureTensor, self.feat_len)
                    featureTensor_channeled = padding_Tensor(featureTensor_channeled, self.feat_len)
                elif self.padding == 'repeat':
                    featureTensor = repeat_padding_Tensor(featureTensor, self.feat_len)
                    featureTensor_channeled = repeat_padding_Tensor(featureTensor_channeled, self.feat_len)
                elif self.padding == 'silence':
                    featureTensor = silence_padding_Tensor(featureTensor, self.feat_len)
                    featureTensor_channeled = silence_padding_Tensor(featureTensor_channeled, self.feat_len)
                else:
                    raise ValueError('Padding should be zero or repeat!')
        else:
            pass
        filename = "_".join(all_info[1:4])
        tag = self.tag[all_info[4]]
        label = self.label[all_info[5]]
        channel = self.channel[rand]
        return featureTensor, featureTensor_channeled, filename, tag, label, channel

    def collate_fn(self, samples):
        if self.pad_chop:
            return default_collate(samples)


class ASVspoof2019LAtrain_DeviceResilient(Dataset):
    def __init__(self, path_to_features="/data2/neil/ASVspoof2019LA/", path_to_deviced="/dataNVME/neil/ASVspoof2019LADevice/", feature='LFCC', feat_len=750, pad_chop=True, padding='repeat', genuine_only=False):
        super(ASVspoof2019LAtrain_DeviceResilient, self).__init__()
        self.path_to_features = path_to_features
        self.path_to_deviced = path_to_deviced
        self.path_to_features = path_to_features
        self.ptf = os.path.join(path_to_features, "train")
        self.feat_len = feat_len
        self.feature = feature
        self.pad_chop = pad_chop
        self.padding = padding
        self.genuine_only = genuine_only
        self.tag = {"-": 0, "A01": 1, "A02": 2, "A03": 3, "A04": 4, "A05": 5, "A06": 6, "A07": 7, "A08": 8, "A09": 9,
                    "A10": 10, "A11": 11, "A12": 12, "A13": 13, "A14": 14, "A15": 15, "A16": 16, "A17": 17, "A18": 18,
                    "A19": 19}
        self.label = {"spoof": 1, "bonafide": 0}
        self.devices = ['AKSPKRS80sUk002-16000', 'AKSPKRSVinUk002-16000', 'Doremi-16000', 'RCAPB90-16000',
                        'ResloRBRedLabel-16000', 'AKSPKRSSpeaker002-16000', 'BehritoneirRecording-16000',
                        'OktavaML19-16000', 'ResloRB250-16000', 'SonyC37Fet-16000']
        self.original_all_files = librosa.util.find_files(os.path.join(self.ptf, self.feature), ext="pt")
        self.deviced_all_files = [librosa.util.find_files(os.path.join(self.path_to_deviced, devicex), ext="pt") for devicex in self.devices]

    def __len__(self):
        return len(self.original_all_files)

    def __getitem__(self, idx):
        filepath = self.original_all_files[idx]
        basename = os.path.basename(filepath)
        all_info = basename.split(".")[0].split("_")
        featureTensor = torch.load(filepath)
        rand = np.random.randint(0, len(self.devices))
        filepath_deviced = self.deviced_all_files[rand][idx]
        featureTensor_deviced = torch.load(filepath_deviced)
        this_feat_len = max(featureTensor.shape[1], featureTensor_deviced.shape[1])
        # print(this_feat_len)
        # print(featureTensor_deviced.shape[1])
        if not featureTensor.shape[1] == featureTensor_deviced.shape[1]:
            if featureTensor.shape[1] < featureTensor_deviced.shape[1]:
                featureTensor = silence_padding_Tensor(featureTensor, featureTensor_deviced.shape[1])
            else:
                featureTensor_deviced = silence_padding_Tensor(featureTensor_deviced, featureTensor.shape[1])
            # print(self.channel[rand])

        if self.pad_chop:
            if this_feat_len > self.feat_len:
                startp = np.random.randint(this_feat_len - self.feat_len)
                featureTensor = featureTensor[:, startp:startp + self.feat_len, :]
                featureTensor_deviced = featureTensor_deviced[:, startp:startp + self.feat_len, :]
            if this_feat_len < self.feat_len:
                if self.padding == 'zero':
                    featureTensor = padding_Tensor(featureTensor, self.feat_len)
                    featureTensor_deviced = padding_Tensor(featureTensor_deviced, self.feat_len)
                elif self.padding == 'repeat':
                    featureTensor = repeat_padding_Tensor(featureTensor, self.feat_len)
                    featureTensor_deviced = repeat_padding_Tensor(featureTensor_deviced, self.feat_len)
                elif self.padding == 'silence':
                    featureTensor = silence_padding_Tensor(featureTensor, self.feat_len)
                    featureTensor_deviced = silence_padding_Tensor(featureTensor_deviced, self.feat_len)
                else:
                    raise ValueError('Padding should be zero or repeat!')
        else:
            pass
        filename = "_".join(all_info[1:4])
        tag = self.tag[all_info[4]]
        label = self.label[all_info[5]]
        device = self.devices[rand]
        return featureTensor, featureTensor_deviced, filename, tag, label, device

    def collate_fn(self, samples):
        if self.pad_chop:
            return default_collate(samples)


class ASVspoof2019LA_DeviceAdversarial(Dataset):
    def __init__(self, path_to_features="/data2/neil/ASVspoof2019LA/", path_to_deviced="/dataNVME/neil/ASVspoof2019LADevice", part="train", feature='LFCC', feat_len=750, pad_chop=True, padding='repeat', genuine_only=False):
        super(ASVspoof2019LA_DeviceAdversarial, self).__init__()
        self.path_to_features = path_to_features
        suffix = {"train" : "", "dev":"Dev", "eval": "Eval"}
        self.path_to_deviced = path_to_deviced + suffix[part]
        self.path_to_features = path_to_features
        self.ptf = os.path.join(path_to_features, part)
        self.feat_len = feat_len
        self.feature = feature
        self.pad_chop = pad_chop
        self.padding = padding
        self.genuine_only = genuine_only
        self.tag = {"-": 0, "A01": 1, "A02": 2, "A03": 3, "A04": 4, "A05": 5, "A06": 6, "A07": 7, "A08": 8, "A09": 9,
                    "A10": 10, "A11": 11, "A12": 12, "A13": 13, "A14": 14, "A15": 15, "A16": 16, "A17": 17, "A18": 18,
                    "A19": 19}
        self.label = {"spoof": 1, "bonafide": 0}
        self.devices = ['AKSPKRS80sUk002-16000', 'AKSPKRSVinUk002-16000', 'Doremi-16000', 'RCAPB90-16000',
                        'ResloRBRedLabel-16000', 'AKSPKRSSpeaker002-16000', 'BehritoneirRecording-16000',
                        'OktavaML19-16000', 'ResloRB250-16000', 'SonyC37Fet-16000']
        if part == "eval":
            self.devices = ['AKSPKRS80sUk002-16000', 'AKSPKRSVinUk002-16000', 'Doremi-16000', 'RCAPB90-16000',
                        'ResloRBRedLabel-16000', 'AKSPKRSSpeaker002-16000', 'BehritoneirRecording-16000',
                        'OktavaML19-16000', 'ResloRB250-16000', 'SonyC37Fet-16000',  'iPadirRecording-16000', 'iPhoneirRecording-16000']
        # self.devices = ['Doremi-16000',
        #                 'ResloRBRedLabel-16000',
        #                 'SonyC37Fet-16000']
        self.original_all_files = librosa.util.find_files(os.path.join(self.ptf, self.feature), ext="pt")
        self.deviced_all_files = [librosa.util.find_files(os.path.join(self.path_to_deviced, devicex), ext="pt") for devicex in self.devices]

    def __len__(self):
        return len(self.original_all_files) * (len(self.devices) + 1)
        # return 220

    def __getitem__(self, idx):
        device_idx = idx % (len(self.devices) + 1)
        filename_idx = idx // (len(self.devices) + 1)
        if device_idx == 0:
            filepath = self.original_all_files[filename_idx]
        else:
            filepath = self.deviced_all_files[device_idx-1][filename_idx]
        basename = os.path.basename(filepath)
        all_info = basename.split(".")[0].split("_")
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
                elif self.padding == 'silence':
                    featureTensor = silence_padding_Tensor(featureTensor, self.feat_len)
                else:
                    raise ValueError('Padding should be zero or repeat!')
        else:
            pass
        filename = "_".join(all_info[1:4])
        tag = self.tag[all_info[4]]
        label = self.label[all_info[5]]
        return featureTensor, filename, tag, label, device_idx

    def collate_fn(self, samples):
        if self.pad_chop:
            return default_collate(samples)


class ASVspoof2019LA_DeviceAdversarial_Eval(Dataset):
    def __init__(self, path_to_features="/data2/neil/ASVspoof2019LA/", path_to_deviced="/dataNVME/neil/ASVspoof2019LADeviceEval", channel="AKSPKRS80sUk002-16000", feature='LFCC', feat_len=750, pad_chop=True, padding='repeat', genuine_only=False):
        super(ASVspoof2019LA_DeviceAdversarial_Eval, self).__init__()
        self.path_to_features = path_to_features
        self.path_to_deviced = path_to_deviced
        self.path_to_features = path_to_features
        self.ptf = os.path.join(path_to_features, "eval")
        self.feat_len = feat_len
        self.feature = feature
        self.pad_chop = pad_chop
        self.padding = padding
        self.genuine_only = genuine_only
        self.tag = {"-": 0, "A01": 1, "A02": 2, "A03": 3, "A04": 4, "A05": 5, "A06": 6, "A07": 7, "A08": 8, "A09": 9,
                    "A10": 10, "A11": 11, "A12": 12, "A13": 13, "A14": 14, "A15": 15, "A16": 16, "A17": 17, "A18": 18,
                    "A19": 19}
        self.label = {"spoof": 1, "bonafide": 0}
        # self.devices = ['AKSPKRS80sUk002-16000', 'AKSPKRSVinUk002-16000', 'Doremi-16000', 'RCAPB90-16000',
        #                 'ResloRBRedLabel-16000', 'AKSPKRSSpeaker002-16000', 'BehritoneirRecording-16000',
        #                 'OktavaML19-16000', 'ResloRB250-16000', 'SonyC37Fet-16000']
        # if part == "eval":
        #     self.devices = ['AKSPKRS80sUk002-16000', 'AKSPKRSVinUk002-16000', 'Doremi-16000', 'RCAPB90-16000',
        #                 'ResloRBRedLabel-16000', 'AKSPKRSSpeaker002-16000', 'BehritoneirRecording-16000',
        #                 'OktavaML19-16000', 'ResloRB250-16000', 'SonyC37Fet-16000',  'iPadirRecording-16000', 'iPhoneirRecording-16000']
        # self.devices = ['Doremi-16000',
        #                 'ResloRBRedLabel-16000',
        #                 'SonyC37Fet-16000']
        # self.original_all_files = librosa.util.find_files(os.path.join(self.ptf, self.feature), ext="pt")
        self.deviced_all_files = librosa.util.find_files(os.path.join(self.path_to_deviced, channel), ext="pt")

    def __len__(self):
        return len(self.deviced_all_files)
        # return 220

    def __getitem__(self, idx):
        filename_idx = idx
        filepath = self.deviced_all_files[filename_idx]
        basename = os.path.basename(filepath)
        all_info = basename.split(".")[0].split("_")
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
                elif self.padding == 'silence':
                    featureTensor = silence_padding_Tensor(featureTensor, self.feat_len)
                else:
                    raise ValueError('Padding should be zero or repeat!')
        else:
            pass
        filename = "_".join(all_info[1:4])
        tag = self.tag[all_info[4]]
        label = self.label[all_info[5]]
        return featureTensor, filename, tag, label

    def collate_fn(self, samples):
        if self.pad_chop:
            return default_collate(samples)


def padding_Tensor(spec, ref_len):
    _, cur_len, width = spec.shape
    assert ref_len > cur_len
    padd_len = ref_len - cur_len
    return torch.cat((spec, torch.zeros((1, padd_len, width), dtype=spec.dtype)), 1)

def repeat_padding_Tensor(spec, ref_len):
    mul = int(np.ceil(ref_len / spec.shape[1]))
    spec = spec.repeat(1, mul, 1)[:, :ref_len, :]
    return spec

def silence_padding_Tensor(spec, ref_len):
    _, cur_len, width = spec.shape
    assert ref_len > cur_len
    padd_len = ref_len - cur_len
    return torch.cat((silence_pad_value.repeat(1, padd_len, 1).to(spec.device), spec), 1)



if __name__ == "__main__":
    # path_to_features = '/data2/neil/ASVspoof2019LA/'  # if run on GPU
    # training_set = ASVspoof2019("LA", path_to_features, 'train',
    #                             'LFCC', feat_len=750, pad_chop=False, padding='repeat')
    # feat_mat, tag, label = training_set[2999]
    # print(len(training_set))
    # # print(this_len)
    # print(feat_mat.shape)
    # print(tag)
    # print(label)

    # samples = [training_set[26], training_set[27], training_set[28], training_set[29]]
    # out = training_set.collate_fn(samples)

    training_set = ASVspoof2019LAtrain_resilient()
    feat_mat, feat_mat_chan, filename, tag, label, channel = training_set[299]
    print(len(training_set))
    print(tag)
    print(label)


    trainDataLoader = DataLoader(training_set, batch_size=32, shuffle=True, num_workers=0, collate_fn=training_set.collate_fn)
    feat_mat_batch, _, tags, labels = [d for d in next(iter(trainDataLoader))]
    print(feat_mat_batch.shape)
    # print(this_len)
    # print(feat_mat_batch)


    # asvspoof = ASVspoof2019("LA", "/data2/neil/ASVspoof2019LA/", part='train', feature='LFCC', feat_len=750, padding='repeat', genuine_only=True)
    # print(len(asvspoof))
    # featTensor, tag, label = asvspoof[2579]
    # print(featTensor.shape)
    # # print(filename)
    # print(tag)
    # print(label)

    # libritts = LIBRITTS(root="/data/neil", download=True)
