import dataset
from feature_extraction import LFCC
import os
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

cuda = torch.cuda.is_available()
print('Cuda device available: ', cuda)
device = torch.device("cuda" if cuda else "cpu")

# vctk = dataset.VCTK_092(root="/data/neil/VCTK", download=False)
# target_dir = "/dataNVME/neil/VCTK/LFCC"
# lfcc = LFCC(320, 160, 512, 16000, 20, with_energy=False)
# for idx in range(len(vctk)):
#     print("Processing", idx)
#     waveform, sample_rate, utterance, speaker_id, utterance_id = vctk[idx]
#     lfccOfWav = lfcc(waveform)
#     torch.save(lfccOfWav, os.path.join(target_dir, "%d_%s_%s.pt" %(idx, speaker_id, utterance_id)))
# print("Done!")

# librispeech = dataset.LIBRISPEECH(root="/data/neil")
# target_dir = "/dataNVME/neil/libriSpeech/LFCC"
# lfcc = LFCC(320, 160, 512, 16000, 20, with_energy=False)
# lfcc = lfcc.to(device)
# for idx in range(len(librispeech)):
#     print("Processing", idx)
#     waveform, sample_rate, utterance, speaker_id, chapter_id, utterance_id = librispeech[idx]
#     waveform = waveform.to(device)
#     lfccOfWav = lfcc(waveform)
#     torch.save(lfccOfWav, os.path.join(target_dir, "%d_%s_%s_%s.pt" %(idx, speaker_id, chapter_id, utterance_id)))
# print("Done!")

# for part_ in ["train", "dev", "eval"]:
#     asvspoof_raw = dataset.ASVspoof2019Raw("LA", "/data/neil/DS_10283_3336/", "/data/neil/DS_10283_3336/LA/ASVspoof2019_LA_cm_protocols/", part=part_)
#     target_dir = os.path.join("/dataNVME/neil/ASVspoof2019LA", part_, "LFCC")
#     lfcc = LFCC(320, 160, 512, 16000, 20, with_energy=False)
#     lfcc = lfcc.to(device)
#     for idx in range(len(asvspoof_raw)):
#         print("Processing", idx)
#         waveform, filename, tag, label = asvspoof_raw[idx]
#         waveform = waveform.to(device)
#         lfccOfWav = lfcc(waveform)
#         torch.save(lfccOfWav, os.path.join(target_dir, "%d_%s_%s_%s.pt" % (idx, filename, tag, label)))
#     print("Done!")

libritts = dataset.LIBRITTS(root="/data/neil")
print(len(libritts))
target_dir = "/dataNVME/neil/libriTTS/train-clean-100/LFCC/"
lfcc = LFCC(320, 160, 512, 16000, 20, with_energy=False)
lfcc = lfcc.to(device)
for idx in range(len(libritts)):
    print("Processing", idx)
    waveform, sample_rate, text, normed_text, speaker_id, chapter_id, utterance_id = libritts[idx]
    waveform = waveform.to(device)
    lfccOfWav = lfcc(waveform)
    torch.save(lfccOfWav, os.path.join(target_dir, "%d_%s_%s_%s.pt" %(idx, speaker_id, chapter_id, utterance_id)))
print("Done!")
