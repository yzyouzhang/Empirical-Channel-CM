import raw_dataset as dataset
from feature_extraction import LFCC
import os
import torch
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

cuda = torch.cuda.is_available()
print('Cuda device available: ', cuda)
device = torch.device("cuda" if cuda else "cpu")

from scipy.fftpack import fft, ifft, fftshift, ifftshift, next_fast_len
import numpy as np
import copy

# for part_ in ["train", "dev", "eval"]:
#     asvspoof_raw = dataset.ASVspoof2019Raw("LA", "/data/neil/DS_10283_3336/", "/data/neil/DS_10283_3336/LA/ASVspoof2019_LA_cm_protocols/", part=part_)
#     target_dir = os.path.join("/data2/neil/ASVspoof2019LASW", part_, "LFCC")
#     if not os.path.exists(target_dir):
#         os.makedirs(target_dir)
#     lfcc = LFCC(320, 160, 512, 16000, 20, with_energy=False)
#     lfcc = lfcc.to(device)
#     for idx in tqdm(range(len(asvspoof_raw))):
#         waveform, filename, tag, label = asvspoof_raw[idx]
#         waveform = spectral_whitening(waveform.squeeze(0).numpy())
#         waveform = torch.from_numpy(np.expand_dims(waveform, axis=0))
#         waveform = waveform.to(device)
#         lfccOfWav = lfcc(waveform)
#         torch.save(lfccOfWav, os.path.join(target_dir, "%05d_%s_%s_%s.pt" % (idx, filename, tag, label)))
#     print("Done!")

# vcc2020 = dataset.VCC2020Raw()
# print(len(vcc2020))
# target_dir = "/data2/neil/VCC2020/LFCC/"
# lfcc = LFCC(320, 160, 512, 16000, 20, with_energy=False)
# lfcc = lfcc.to(device)
# for idx in range(len(vcc2020)):
#     print("Processing", idx)
#     waveform, filename, tag, label = vcc2020[idx]
#     waveform = waveform.to(device)
#     lfccOfWav = lfcc(waveform)
#     torch.save(lfccOfWav, os.path.join(target_dir, "%04d_%s_%s_%s.pt" %(idx, filename, tag, label)))
# print("Done!")

# for part_ in ["train", "dev", "eval"]:
#     asvspoof_raw = dataset.ASVspoof2015Raw("/data/neil/ASVspoof2015/wav", "/data/neil/ASVspoof2015/CM_protocol", part=part_)
#     target_dir = os.path.join("/data2/neil/ASVspoof2015", part_, "LFCC")
#     lfcc = LFCC(320, 160, 512, 16000, 20, with_energy=False)
#     lfcc = lfcc.to(device)
#     for idx in range(len(asvspoof_raw)):
#         print("Processing", idx)
#         waveform, filename, tag, label = asvspoof_raw[idx]
#         waveform = waveform.to(device)
#         lfccOfWav = lfcc(waveform)
#         torch.save(lfccOfWav, os.path.join(target_dir, "%05d_%s_%s_%s.pt" % (idx, filename, tag, label)))
#     print("Done!")

asvspoof2019channel = dataset.ASVspoof2019LARaw_withDevice()
print(len(asvspoof2019channel))
target_dir = "/dataNVME/neil/ASVspoof2019LADeviceEval"
lfcc = LFCC(320, 160, 512, 16000, 20, with_energy=False)
lfcc = lfcc.to(device)
for idx in tqdm(range(len(asvspoof2019channel))):
    waveform, filename, tag, label, channel = asvspoof2019channel[idx]
    waveform = waveform.to(device)
    lfccOfWav = lfcc(waveform)
    if not os.path.exists(os.path.join(target_dir, channel)):
        os.makedirs(os.path.join(target_dir, channel))
    torch.save(lfccOfWav, os.path.join(target_dir, channel, "%06d_%s_%s_%s_%s.pt" %(idx, filename, tag, label, channel)))
print("Done!")
