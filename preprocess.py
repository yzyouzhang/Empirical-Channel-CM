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

def spectral_whitening(tr, smooth=None, filter=None,
                       waterlevel=1e-8, corners=2, zerophase=True):
    """
    Apply spectral whitening to data
    Data is divided by its smoothed (Default: None) amplitude spectrum.
    :param tr: trace to manipulate
    :param smooth: length of smoothing window in Hz
        (default None -> no smoothing)
    :param filter: filter spectrum with bandpass after whitening
        (tuple with min and max frequency)
        (default None -> no filter)
    :param waterlevel: waterlevel relative to mean of spectrum
    :param mask_again: weather to mask array after this operation again and
        set the corresponding data to 0
    :param corners: parameters parsing to filter,
    :param zerophase: parameters parsing to filter
    :return: whitened data
    """

    sr = 16000
    data = np.copy(tr)
    # data = _fill_array(data, fill_value=0)
    # mask = np.ma.getmask(data)

    # transform to frequency domain
    nfft = next_fast_len(len(data))
    spec = fft(data, nfft)

    # amplitude spectrum
    spec_ampl = np.abs(spec)

    # normalization
    spec_ampl /= np.max(spec_ampl)
    spec_ampl_raw = np.copy(spec_ampl)

    # smooth
    if smooth:
        smooth = int(smooth * nfft / sr)
        spec_ampl = ifftshift(smooth_func(fftshift(spec_ampl), smooth))
        spec_ampl_smth = np.copy(spec_ampl)

    # save guard against division by 0
    spec_ampl[spec_ampl < waterlevel] = waterlevel

    # make the spectrum have the equivalent amplitude before/after smooth
    if smooth:
        scale = np.max(spec_ampl_raw) / np.max(spec_ampl_smth)
        spec /= spec_ampl * scale
    else:
        spec /= spec_ampl

    # FFT back to time domain
    ret = np.real(ifft(spec, nfft)[:len(data)])
    tr = ret

    # filter
    if filter is not None:
        tr.filter(type="bandpass", freqmin=filter[0], freqmax=filter[1],
                  corners=corners, zerophase=zerophase)

    return tr

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

# libritts = dataset.LIBRITTS(root="/data/neil")
# print(len(libritts))
# target_dir = "/dataNVME/neil/libriTTS/train-clean-100/LFCC/"
# lfcc = LFCC(320, 160, 512, 16000, 20, with_energy=False)
# lfcc = lfcc.to(device)
# for idx in range(len(libritts)):
#     print("Processing", idx)
#     waveform, sample_rate, text, normed_text, speaker_id, chapter_id, utterance_id = libritts[idx]
#     waveform = waveform.to(device)
#     lfccOfWav = lfcc(waveform)
#     torch.save(lfccOfWav, os.path.join(target_dir, "%d_%s_%s_%s.pt" %(idx, speaker_id, chapter_id, utterance_id)))
# print("Done!")

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
