#!/usr/bin/python3

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import scipy.io as sio
import pickle
import os
import librosa
from torch.utils.data.dataloader import default_collate
import warnings
from typing import Any, Tuple, Union
from pathlib import Path
from utils import download_url, extract_archive, walk_files


torch.set_default_tensor_type(torch.FloatTensor)

SampleType = Tuple[Tensor, int, str, str, str]

def torchaudio_load(filepath):
    wave, sr = librosa.load(filepath, sr=16000)
    waveform = torch.Tensor(np.expand_dims(wave, axis=0))
    return [waveform, sr]

def load_librispeech_item(fileid: str,
                          path: str,
                          ext_audio: str,
                          ext_txt: str) -> Tuple[Tensor, int, str, int, int, int]:
    speaker_id, chapter_id, utterance_id = fileid.split("-")

    file_text = speaker_id + "-" + chapter_id + ext_txt
    file_text = os.path.join(path, speaker_id, chapter_id, file_text)

    fileid_audio = speaker_id + "-" + chapter_id + "-" + utterance_id
    file_audio = fileid_audio + ext_audio
    file_audio = os.path.join(path, speaker_id, chapter_id, file_audio)

    # Load audio
    waveform, sample_rate = torchaudio_load(file_audio)

    # Load text
    with open(file_text) as ft:
        for line in ft:
            fileid_text, utterance = line.strip().split(" ", 1)
            if fileid_audio == fileid_text:
                break
        else:
            # Translation not found
            raise FileNotFoundError("Translation not found for " + fileid_audio)

    return (
        waveform,
        sample_rate,
        utterance,
        int(speaker_id),
        int(chapter_id),
        int(utterance_id),
    )

class LIBRISPEECH(Dataset):
    """Create a Dataset for LibriSpeech.
    Args:
        root (str or Path): Path to the directory where the dataset is found or downloaded.
        url (str, optional): The URL to download the dataset from,
            or the type of the dataset to dowload.
            Allowed type values are ``"dev-clean"``, ``"dev-other"``, ``"test-clean"``,
            ``"test-other"``, ``"train-clean-100"``, ``"train-clean-360"`` and
            ``"train-other-500"``. (default: ``"train-clean-100"``)
        folder_in_archive (str, optional):
            The top-level directory of the dataset. (default: ``"LibriSpeech"``)
        download (bool, optional):
            Whether to download the dataset if it is not found at root path. (default: ``False``).
    """

    _ext_txt = ".trans.txt"
    _ext_audio = ".flac"

    def __init__(self,
                 root: Union[str, Path],
                 basename: str = "train-clean-360",
                 folder_in_archive: str = "LibriSpeech") -> None:

        # Get string representation of 'root' in case Path object is passed
        root = os.fspath(root)

        folder_in_archive = os.path.join(folder_in_archive, basename)

        self._path = os.path.join(root, folder_in_archive)

        self._walker = sorted(str(p.stem) for p in Path(self._path).glob('*/*/*' + self._ext_audio))

    def __getitem__(self, n: int) -> Tuple[Tensor, int, str, int, int, int]:
        """Load the n-th sample from the dataset.
        Args:
            n (int): The index of the sample to be loaded
        Returns:
            tuple: ``(waveform, sample_rate, utterance, speaker_id, chapter_id, utterance_id)``
        """
        fileid = self._walker[n]
        return load_librispeech_item(fileid, self._path, self._ext_audio, self._ext_txt)

    def __len__(self) -> int:
        return len(self._walker)


class VCTK_092(Dataset):
    """Create VCTK 0.92 Dataset
    Args:
        root (str): Root directory where the dataset's top level directory is found.
        mic_id (str): Microphone ID. Either ``"mic1"`` or ``"mic2"``. (default: ``"mic2"``)
        download (bool, optional):
            Whether to download the dataset if it is not found at root path. (default: ``False``).
        url (str, optional): The URL to download the dataset from.
            (default: ``"https://datashare.is.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip"``)
        audio_ext (str, optional): Custom audio extension if dataset is converted to non-default audio format.
    Note:
        * All the speeches from speaker ``p315`` will be skipped due to the lack of the corresponding text files.
        * All the speeches from ``p280`` will be skipped for ``mic_id="mic2"`` due to the lack of the audio files.
        * Some of the speeches from speaker ``p362`` will be skipped due to the lack of  the audio files.
        * See Also: https://datashare.is.ed.ac.uk/handle/10283/3443
    """

    def __init__(
        self,
        root: str,
        mic_id: str = "mic2",
        download: bool = False,
        url: str = "https://datashare.is.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip",
        audio_ext=".flac",
    ):
        if mic_id not in ["mic1", "mic2"]:
            raise RuntimeError(
                f'`mic_id` has to be either "mic1" or "mic2". Found: {mic_id}'
            )

        archive = os.path.join(root, "VCTK-Corpus-0.92.zip")

        self._path = os.path.join(root, "VCTK-Corpus-0.92")
        self._txt_dir = os.path.join(self._path, "txt")
        self._audio_dir = os.path.join(self._path, "wav48_silence_trimmed")
        self._mic_id = mic_id
        self._audio_ext = audio_ext

        if download:
            if not os.path.isdir(self._path):
                if not os.path.isfile(archive):
                    checksum = {"https://datashare.is.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip": "8a6ba2946b36fcbef0212cad601f4bfa"}.get(url, None)
                    download_url(url, root, hash_value=checksum, hash_type="md5")
                extract_archive(archive, self._path)

        if not os.path.isdir(self._path):
            raise RuntimeError(
                "Dataset not found. Please use `download=True` to download it."
            )

        # Extracting speaker IDs from the folder structure
        self._speaker_ids = sorted(os.listdir(self._txt_dir))
        self._sample_ids = []

        """
        Due to some insufficient data complexity in the 0.92 version of this dataset,
        we start traversing the audio folder structure in accordance with the text folder.
        As some of the audio files are missing of either ``mic_1`` or ``mic_2`` but the
        text is present for the same, we first check for the existence of the audio file
        before adding it to the ``sample_ids`` list.
        Once the ``audio_ids`` are loaded into memory we can quickly access the list for
        different parameters required by the user.
        """
        for speaker_id in self._speaker_ids:
            if speaker_id == "p280" and mic_id == "mic2":
                continue
            utterance_dir = os.path.join(self._txt_dir, speaker_id)
            for utterance_file in sorted(
                f for f in os.listdir(utterance_dir) if f.endswith(".txt")
            ):
                utterance_id = os.path.splitext(utterance_file)[0]
                audio_path_mic = os.path.join(
                    self._audio_dir,
                    speaker_id,
                    f"{utterance_id}_{mic_id}{self._audio_ext}",
                )
                if speaker_id == "p362" and not os.path.isfile(audio_path_mic):
                    continue
                self._sample_ids.append(utterance_id.split("_"))

    def _load_text(self, file_path) -> str:
        with open(file_path) as file_path:
            return file_path.readlines()[0]

    def _load_audio(self, file_path) -> Tuple[Tensor, int]:
        return torchaudio_load(file_path)

    def _load_sample(self, speaker_id: str, utterance_id: str, mic_id: str) -> SampleType:
        utterance_path = os.path.join(
            self._txt_dir, speaker_id, f"{speaker_id}_{utterance_id}.txt"
        )
        audio_path = os.path.join(
            self._audio_dir,
            speaker_id,
            f"{speaker_id}_{utterance_id}_{mic_id}{self._audio_ext}",
        )

        # Reading text
        utterance = self._load_text(utterance_path)

        # Reading FLAC
        waveform, sample_rate = self._load_audio(audio_path)

        return (waveform, sample_rate, utterance, speaker_id, utterance_id)

    def __getitem__(self, n: int) -> SampleType:
        """Load the n-th sample from the dataset.
        Args:
            n (int): The index of the sample to be loaded
        Returns:
            tuple: ``(waveform, sample_rate, utterance, speaker_id, utterance_id)``
        """
        speaker_id, utterance_id = self._sample_ids[n]
        return self._load_sample(speaker_id, utterance_id, self._mic_id)

    def __len__(self) -> int:
        return len(self._sample_ids)

class ASVspoof2019Raw(Dataset):
    def __init__(self, access_type, path_to_database, path_to_protocol, part='train'):
        super(ASVspoof2019Raw, self).__init__()
        self.access_type = access_type
        self.ptd = path_to_database
        self.part = part
        self.path_to_audio = os.path.join(self.ptd, access_type, 'ASVspoof2019_'+access_type+'_'+ self.part +'/flac/')
        self.path_to_protocol = path_to_protocol
        self.padding = padding
        protocol = os.path.join(self.path_to_protocol, 'ASVspoof2019.'+access_type+'.cm.'+ self.part + '.trl.txt')
        if self.part == "eval":
            protocol = os.path.join(self.ptd, access_type, 'ASVspoof2019_' + access_type +
                                    '_cm_protocols/ASVspoof2019.' + access_type + '.cm.' + self.part + '.trl.txt')
        if self.access_type == 'LA':
            self.tag = {"-": 0, "A01": 1, "A02": 2, "A03": 3, "A04": 4, "A05": 5, "A06": 6, "A07": 7, "A08": 8, "A09": 9,
                      "A10": 10, "A11": 11, "A12": 12, "A13": 13, "A14": 14, "A15": 15, "A16": 16, "A17": 17, "A18": 18,
                      "A19": 19}
        else:
            self.tag = {"-": 0, "AA": 1, "AB": 2, "AC": 3, "BA": 4, "BB": 5, "BC": 6, "CA": 7, "CB": 8, "CC": 9}
        self.label = {"spoof": 1, "bonafide": 0}

        # # would not work if change data split but this csv is only for feat_len
        # self.csv = pd.read_csv(self.ptf + "Set_csv.csv")

        with open(protocol, 'r') as f:
            audio_info = [info.strip().split() for info in f.readlines()]
            self.all_info = audio_info

    def __len__(self):
        return len(self.all_info)

    def __getitem__(self, idx):
        speaker, filename, _, tag, label = self.all_info[idx]
        filepath = os.path.join(self.path_to_audio, filename + ".flac")
        waveform, sr = torchaudio_load(filepath)

        return waveform, filename, tag, label

    def collate_fn(self, samples):
        return default_collate(samples)


class ASVspoof2019(Dataset):
    def __init__(self, access_type, path_to_database, path_to_features, path_to_protocol, part='train', feature='LFCC',
                 genuine_only=False, feat_len=650, pad_chop=True, padding='zero'):
        super(ASVspoof2019, self).__init__()
        self.access_type = access_type
        self.ptd = path_to_database
        self.path_to_features = path_to_features
        self.part = part
        self.ptf = os.path.join(path_to_features, self.part)
        self.path_to_audio = os.path.join(self.ptd, access_type, 'ASVspoof2019_'+access_type+'_'+ self.part +'/flac/')
        self.genuine_only = genuine_only
        self.feat_len = feat_len
        self.feature = feature
        self.pad_chop = pad_chop
        self.path_to_protocol = path_to_protocol
        self.padding = padding
        protocol = os.path.join(self.path_to_protocol, 'ASVspoof2019.'+access_type+'.cm.'+ self.part + '.trl.txt')
        if self.part == "eval":
            protocol = os.path.join(self.ptd, access_type, 'ASVspoof2019_' + access_type +
                                    '_cm_protocols/ASVspoof2019.' + access_type + '.cm.' + self.part + '.trl.txt')
        if self.access_type == 'LA':
            self.tag = {"-": 0, "A01": 1, "A02": 2, "A03": 3, "A04": 4, "A05": 5, "A06": 6, "A07": 7, "A08": 8, "A09": 9,
                      "A10": 10, "A11": 11, "A12": 12, "A13": 13, "A14": 14, "A15": 15, "A16": 16, "A17": 17, "A18": 18,
                      "A19": 19}
        else:
            self.tag = {"-": 0, "AA": 1, "AB": 2, "AC": 3, "BA": 4, "BB": 5, "BC": 6, "CA": 7, "CB": 8, "CC": 9}
        self.label = {"spoof": 1, "bonafide": 0}

        # # would not work if change data split but this csv is only for feat_len
        # self.csv = pd.read_csv(self.ptf + "Set_csv.csv")

        with open(protocol, 'r') as f:
            audio_info = [info.strip().split() for info in f.readlines()]
            if genuine_only:
                assert self.part in ["train", "dev"]
                if self.access_type == "LA":
                    num_bonafide = {"train": 2580, "dev": 2548}
                    self.all_info = audio_info[:num_bonafide[self.part]]
                else:
                    self.all_info = audio_info[:5400]
            else:
                self.all_info = audio_info

    def __len__(self):
        return len(self.all_info)

    def __getitem__(self, idx):
        speaker, filename, _, tag, label = self.all_info[idx]
        try:
            with open(self.ptf + '/'+ filename + self.feature + '.pkl', 'rb') as feature_handle:
                feat_mat = pickle.load(feature_handle)
        except:
            # add this exception statement since we may change the data split
            def the_other(train_or_dev):
                assert train_or_dev in ["train", "dev"]
                res = "dev" if train_or_dev == "train" else "train"
                return res
            with open(os.path.join(self.path_to_features, the_other(self.part)) + '/'+ filename + self.feature + '.pkl', 'rb') as feature_handle:
                feat_mat = pickle.load(feature_handle)

        feat_mat = torch.from_numpy(feat_mat)
        this_feat_len = feat_mat.shape[1]
        # assert self.csv.at[idx, "feat_len"] == feat_mat.shape[1]
        if this_feat_len > self.feat_len:
            startp = np.random.randint(this_feat_len-self.feat_len)
            feat_mat = feat_mat[:, startp:startp+self.feat_len]
        if this_feat_len < self.feat_len:
            if self.padding == 'zero':
                feat_mat = padding(feat_mat, self.feat_len)
            elif self.padding == 'repeat':
                feat_mat = repeat_padding(feat_mat, self.feat_len)
            else:
                raise ValueError('Padding should be zero or repeat!')

        return feat_mat, filename, self.tag[tag], self.label[label]

    def collate_fn(self, samples):
        return default_collate(samples)


# class ASVspoof2019(Dataset):
#     def __init__(self, access_type, path_to_features, part='train', feature='LFCC', feat_len=650, pad_chop=True, padding='zero'):
#         super(ASVspoof2019, self).__init__()
#         self.access_type = access_type
#         self.path_to_features = path_to_features
#         self.part = part
#         self.ptf = os.path.join(path_to_features, self.part)
#         self.feat_len = feat_len
#         self.feature = feature
#         self.pad_chop = pad_chop
#         self.padding = padding
#         if self.access_type == 'LA':
#             self.tag = {"-": 0, "A01": 1, "A02": 2, "A03": 3, "A04": 4, "A05": 5, "A06": 6, "A07": 7, "A08": 8, "A09": 9,
#                       "A10": 10, "A11": 11, "A12": 12, "A13": 13, "A14": 14, "A15": 15, "A16": 16, "A17": 17, "A18": 18,
#                       "A19": 19}
#         elif self.access_type == 'PA':
#             self.tag = {"-": 0, "AA": 1, "AB": 2, "AC": 3, "BA": 4, "BB": 5, "BC": 6, "CA": 7, "CB": 8, "CC": 9}
#         else:
#             raise ValueError("Access type should be LA or PA!")
#         self.label = {"spoof": 1, "bonafide": 0}
#         self.all_files = librosa.util.find_files(os.path.join(self.ptf, self.feature), ext="pt")
#
#     def __len__(self):
#         return len(self.all_files)
#
#     def __getitem__(self, idx):
#         all_info = self.all_files[idx].split(".")[0].split("_")
#         assert len(all_info) == 6
#         # with open(self.ptf + '/'+ filename + self.feature + '.pkl', 'rb') as feature_handle:
#         #     feat_mat = pickle.load(feature_handle)
#         #
#         #
#         # feat_mat = torch.from_numpy(feat_mat)
#         # this_feat_len = feat_mat.shape[1]
#         # # assert self.csv.at[idx, "feat_len"] == feat_mat.shape[1]
#         # if this_feat_len > self.feat_len:
#         #     startp = np.random.randint(this_feat_len-self.feat_len)
#         #     feat_mat = feat_mat[:, startp:startp+self.feat_len]
#         # if this_feat_len < self.feat_len:
#         #     if self.padding == 'zero':
#         #         feat_mat = padding(feat_mat, self.feat_len)
#         #     elif self.padding == 'repeat':
#         #         feat_mat = repeat_padding(feat_mat, self.feat_len)
#         #     else:
#         #         raise ValueError('Padding should be zero or repeat!')
#         #
#         # return feat_mat, self.tag[tag], self.label[label]
#
#     def collate_fn(self, samples):
#         return default_collate(samples)

class LibriGenuine(Dataset):
    def __init__(self, path_to_features, part='train', feature='LFCC', feat_len=750, pad_chop=True, padding='repeat'):
        super(LibriGenuine, self).__init__()
        self.ptf = path_to_features
        self.feat_len = feat_len
        self.feature = feature
        self.pad_chop = pad_chop
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
        featureTensor = torch.load(filepath)[:, 1:, :]
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

    def collate_fn(self, samples):
        return default_collate(samples)


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
    # path_to_database = '/data/neil/DS_10283_3336/'  # if run on GPU
    # path_to_features = '/dataNVME/neil/ASVspoof2019LAFeatures/'  # if run on GPU
    # path_to_protocol = '/data/neil/DS_10283_3336/LA/ASVspoof2019_LA_cm_protocols/'
    # training_set = ASVspoof2019("LA", path_to_database, path_to_features, path_to_protocol, genuine_only=False, pad_chop=False, feature='LFCC', feat_len=320)
    # feat_mat, audio_fn, tag, label = training_set[2999]
    # print(len(training_set))
    # print(audio_fn)
    # print(feat_mat.shape)
    # # print(cqcc.shape)
    # # print(lfcc.shape)
    # print(tag)
    # print(label)

    # samples = [training_set[26], training_set[27], training_set[28], training_set[29]]
    # out = training_set.collate_fn(samples)

    # training_set = ASVspoof2019(path_to_database, path_to_features)
    # cqcc, audio_fn, tag, label = training_set[2580]
    # print(len(training_set))
    # print(audio_fn)
    # # print(mfcc.shape)
    # print(cqcc.shape)
    # # print(lfcc.shape)
    # print(tag)
    # print(label)

    # trainDataLoader = DataLoader(training_set, batch_size=32, shuffle=True, num_workers=0, collate_fn=training_set.collate_fn)
    # feat_mat_batch, audio_fn, tags, labels = [d for d in next(iter(trainDataLoader))]
    # print(feat_mat_batch.shape)
    # # print(feat_mat_batch)
    #
    # vctk = VCTK_092(root="/data/neil/VCTK", download=False)
    # print(len(vctk))
    # waveform, sample_rate, utterance, speaker_id, utterance_id = vctk[124]
    # print(waveform.shape)
    # print(sample_rate)
    # print(utterance)
    # print(speaker_id)
    # print(utterance_id)

    # librispeech = LIBRISPEECH(root="/data/neil")
    # print(len(librispeech))
    # waveform, sample_rate, utterance, speaker_id, chapter_id, utterance_id = librispeech[164]
    # print(waveform.shape)
    # print(sample_rate)
    # print(utterance)
    # print(speaker_id)
    # print(chapter_id)
    # print(utterance_id)
    #
    # libriGen = LibriGenuine("/dataNVME/neil/libriSpeech/", feature='LFCC', feat_len=750, pad_chop=True, padding='repeat')
    # print(len(libriGen))
    # featTensor, tag, label = libriGen[123]
    # print(featTensor.shape)
    # print(tag)
    # print(label)

    asvspoof_raw = ASVspoof2019Raw("LA", "/data/neil/DS_10283_3336/", "/data/neil/DS_10283_3336/LA/ASVspoof2019_LA_cm_protocols/", part="eval")
    print(len(asvspoof_raw))
    waveform, filename, tag, label = asvspoof_raw[123]
    print(waveform.shape)
    print(filename)
    print(tag)
    print(label)
