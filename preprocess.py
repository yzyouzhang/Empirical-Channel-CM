import dataset
from feature_extraction import LFCC
import os
import torch

vctk = dataset.VCTK_092(root="/data/neil/VCTK", download=False)
target_dir = "/dataNVME/neil/VCTK/LFCC"
lfcc = LFCC(320, 160, 512, 16000, 20, with_energy=False)
for idx in range(len(vctk)):
    print("Processing", idx)
    waveform, sample_rate, utterance, speaker_id, utterance_id = vctk[idx]
    lfccOfWav = lfcc(waveform)
    torch.save(lfccOfWav, os.path.join(target_dir, "%d_%s_%s.pt" %(idx, speaker_id, utterance_id)))
print("Done!")


