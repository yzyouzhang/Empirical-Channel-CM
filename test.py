import logging
import os
import torch
from torch.utils.model_zoo import tqdm
import random
import numpy as np
from dataset import *
from torch.utils.data import DataLoader
import torch.nn.functional as F
import eval_metrics as em
from evaluate_tDCF_asvspoof19 import compute_eer_and_tdcf
import time
from distutils import util

## Adapted from https://github.com/pytorch/audio/tree/master/torchaudio
## https://github.com/nii-yamagishilab/project-NN-Pytorch-scripts/blob/newfunctions/

def str2bool(v):
    return bool(util.strtobool(v))

def setup_seed(random_seed, cudnn_deterministic=True):
    """ set_random_seed(random_seed, cudnn_deterministic=True)

    Set the random_seed for numpy, python, and cudnn

    input
    -----
      random_seed: integer random seed
      cudnn_deterministic: for torch.backends.cudnn.deterministic

    Note: this default configuration may result in RuntimeError
    see https://pytorch.org/docs/stable/notes/randomness.html
    """

    # # initialization
    # torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = cudnn_deterministic
        torch.backends.cudnn.benchmark = False

def init():
    parser = argparse.ArgumentParser("load model scores")
    parser.add_argument('-m', '--model_dir', type=str, help="directory for pretrained model",
                        default='/data3/neil/chan/adv1010')
    parser.add_argument("-t", "--task", type=str, help="which dataset you would like to test on",
                        required=True, default='ASVspoof2019LA', choices=["ASVspoof2019LA", "ASVspoof2015", "VCC2020"])
    parser.add_argument('-l', '--loss', help='loss for scoring', default="ocsoftmax",
                        required=False, choices=[None, "ocsoftmax", "amsoftmax", "p2sgrad"])
    parser.add_argument("--gpu", type=str, help="GPU index", default="0")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.cuda = torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")

    return args

def test_model(feat_model_path, loss_model_path, part, add_loss):
    dirname = os.path.dirname
    basename = os.path.splitext(os.path.basename(feat_model_path))[0]
    if "checkpoint" in dirname(feat_model_path):
        dir_path = dirname(dirname(feat_model_path))
    else:
        dir_path = dirname(feat_model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(feat_model_path)
    loss_model = torch.load(loss_model_path) if add_loss is not None else None
    test_set = ASVspoof2019("LA", "/dataNVME/neil/ASVspoof2019LA/", part,
                            "LFCC", feat_len=750, padding="repeat")
    testDataLoader = DataLoader(test_set, batch_size=16, shuffle=False, num_workers=0)
    model.eval()
    score_loader, idx_loader = [], []

    with open(os.path.join(dir_path, 'checkpoint_cm_score.txt'), 'w') as cm_score_file:
        for i, (lfcc, audio_fn, tags, labels) in enumerate(tqdm(testDataLoader)):
            lfcc = lfcc.transpose(2,3).to(device)
            # print(lfcc.shape)
            tags = tags.to(device)
            labels = labels.to(device)

            feats, lfcc_outputs = model(lfcc)

            score = F.softmax(lfcc_outputs)[:, 0]
            # print(score)

            if add_loss == "ocsoftmax":
                ang_isoloss, score = loss_model(feats, labels)
            elif add_loss == "amsoftmax":
                outputs, moutputs = loss_model(feats, labels)
                score = F.softmax(outputs, dim=1)[:, 0]
            else: pass

            for j in range(labels.size(0)):
                cm_score_file.write(
                    'A%02d %s %s\n' % (tags[j].data,
                                          "spoof" if labels[j].data.cpu().numpy() else "bonafide",
                                          score[j].item()))

            score_loader.append(score.detach().cpu())
            idx_loader.append(labels.detach().cpu())

    scores = torch.cat(score_loader, 0).data.cpu().numpy()
    labels = torch.cat(idx_loader, 0).data.cpu().numpy()
    eer = em.compute_eer(scores[labels == 0], scores[labels == 1])[0]
    other_eer = em.compute_eer(-scores[labels == 0], -scores[labels == 1])[0]
    eer = min(eer, other_eer)

    return eer

# def test(model_dir, add_loss):
#     model_path = os.path.join(model_dir, "anti-spoofing_feat_model.pt")
#     loss_model_path = os.path.join(model_dir, "anti-spoofing_loss_model.pt")
#     test_model(model_path, loss_model_path, "eval", add_loss)

def test_on_VCC(feat_model_path, loss_model_path, part, add_loss):
    dirname = os.path.dirname
    basename = os.path.splitext(os.path.basename(feat_model_path))[0]
    if "checkpoint" in dirname(feat_model_path):
        dir_path = dirname(dirname(feat_model_path))
    else:
        dir_path = dirname(feat_model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(feat_model_path)
    # model = torch.nn.DataParallel(model, list(range(torch.cuda.device_count())))  # for multiple GPUs
    loss_model = torch.load(loss_model_path) if add_loss is not None else None
    test_set_VCC = VCC2020("/data2/neil/VCC2020/", "LFCC", feat_len=750, padding="repeat")
    testDataLoader = DataLoader(test_set_VCC, batch_size=4, shuffle=False, num_workers=0)
    model.eval()
    score_loader, idx_loader = [], []

    with open(os.path.join(dir_path, 'checkpoint_cm_score_VCC.txt'), 'w') as cm_score_file:
        for i, (lfcc, _, tags, labels, _) in enumerate(tqdm(testDataLoader)):
            lfcc = lfcc.transpose(2,3).to(device)

            tags = tags.to(device)
            labels = labels.to(device)

            feats, lfcc_outputs = model(lfcc)

            score = F.softmax(lfcc_outputs)[:, 0]

            if add_loss == "ocsoftmax":
                ang_isoloss, score = loss_model(feats, labels)
            elif add_loss == "amsoftmax":
                outputs, moutputs = loss_model(feats, labels)
                score = F.softmax(outputs, dim=1)[:, 0]
            else: pass

            for j in range(labels.size(0)):
                cm_score_file.write(
                    'A%02d %s %s\n' % (tags[j].data,
                                          "spoof" if labels[j].data.cpu().numpy() else "bonafide",
                                          score[j].item()))

            score_loader.append(score.detach().cpu())
            idx_loader.append(labels.detach().cpu())

    scores = torch.cat(score_loader, 0).data.cpu().numpy()
    labels = torch.cat(idx_loader, 0).data.cpu().numpy()
    eer = em.compute_eer(scores[labels == 0], scores[labels == 1])[0]
    other_eer = em.compute_eer(-scores[labels == 0], -scores[labels == 1])[0]
    eer = min(eer, other_eer)

    return eer

def test_on_ASVspoof2015(feat_model_path, loss_model_path, part, add_loss):
    dirname = os.path.dirname
    basename = os.path.splitext(os.path.basename(feat_model_path))[0]
    if "checkpoint" in dirname(feat_model_path):
        dir_path = dirname(dirname(feat_model_path))
    else:
        dir_path = dirname(feat_model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(feat_model_path)
    # model = torch.nn.DataParallel(model, list(range(torch.cuda.device_count())))  # for multiple GPUs
    loss_model = torch.load(loss_model_path) if add_loss is not None else None
    test_set_2015 = ASVspoof2015("/data2/neil/ASVspoof2015/", part="eval", feature="LFCC", feat_len=750, padding="repeat")
    print(len(test_set_2015))
    testDataLoader = DataLoader(test_set_2015, batch_size=4, shuffle=False, num_workers=0)
    model.eval()
    score_loader, idx_loader = [], []

    with open(os.path.join(dir_path, 'checkpoint_cm_score_VCC.txt'), 'w') as cm_score_file:
        for i, (lfcc, audio_fn, tags, labels, _) in enumerate(tqdm(testDataLoader)):
            lfcc = lfcc.transpose(2,3).to(device)
            tags = tags.to(device)
            labels = labels.to(device)

            feats, lfcc_outputs = model(lfcc)

            score = F.softmax(lfcc_outputs)[:, 0]
            # print(score)

            if add_loss == "ocsoftmax":
                ang_isoloss, score = loss_model(feats, labels)
            elif add_loss == "amsoftmax":
                outputs, moutputs = loss_model(feats, labels)
                score = F.softmax(outputs, dim=1)[:, 0]
            else: pass

            for j in range(labels.size(0)):
                cm_score_file.write(
                    '%s A%02d %s %s\n' % (audio_fn[j], tags[j].data,
                                          "spoof" if labels[j].data.cpu().numpy() else "bonafide",
                                          score[j].item()))

            score_loader.append(score.detach().cpu())
            idx_loader.append(labels.detach().cpu())

    scores = torch.cat(score_loader, 0).data.cpu().numpy()
    labels = torch.cat(idx_loader, 0).data.cpu().numpy()
    eer = em.compute_eer(scores[labels == 0], scores[labels == 1])[0]
    other_eer = em.compute_eer(-scores[labels == 0], -scores[labels == 1])[0]
    eer = min(eer, other_eer)

    return eer

def test_individual_attacks(cm_score_file):
    # Load CM scores
    cm_data = np.genfromtxt(cm_score_file, dtype=str)
    cm_sources = cm_data[:, 1]
    cm_keys = cm_data[:, 2]
    cm_scores = cm_data[:, 3].astype(np.float)

    other_cm_scores = -cm_scores

    eer_cm_lst, min_tDCF_lst = [], []
    for attack_idx in range(0, 55):
        # Extract target, nontarget, and spoof scores from the ASV scores

        # Extract bona fide (real human) and spoof scores from the CM scores
        bona_cm = cm_scores[cm_keys == 'bonafide']
        spoof_cm = cm_scores[cm_sources == 'A%02d' % attack_idx]

        # EERs of the standalone systems and fix ASV operating point to EER threshold
        eer_cm = em.compute_eer(bona_cm, spoof_cm)[0]

        other_eer_cm = em.compute_eer(other_cm_scores[cm_keys == 'bonafide'], other_cm_scores[cm_sources == 'A%02d' % attack_idx])[0]

        eer_cm_lst.append(min(eer_cm, other_eer_cm))

    return eer_cm_lst


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    device = torch.device("cuda")

    args = init()

    model_path = os.path.join(args.model_dir, "anti-spoofing_cqcc_model.pt")
    loss_model_path = os.path.join(args.model_dir, "anti-spoofing_loss_model.pt")

    if args.task == "ASVspoof2019LA":
        eer = test_model(model_path, loss_model_path, "eval", "ocsoftmax")
    elif args.task == "ASVspoof2015":
        eer = test_on_ASVspoof2015(model_path, loss_model_path, "eval", "ocsoftmax")
    elif args.task =="VCC2020":
        eer = test_on_VCC(model_path, loss_model_path, "eval", "ocsoftmax")
    else:
        raise ValueError("Evaluation task unknown!")
    print(eer)

