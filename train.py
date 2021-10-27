import torch
import torch.nn as nn
import argparse
import os
import json
import shutil
import numpy as np
from model import *
from dataset import ASVspoof2019
from torch.utils.data import DataLoader
from evaluate_tDCF_asvspoof19 import compute_eer_and_tdcf
from loss import *
from collections import defaultdict
from tqdm import tqdm, trange
import random
from test import *
import eval_metrics as em

torch.set_default_tensor_type(torch.FloatTensor)

def initParams():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--seed', type=int, help="random number seed", default=688)

    # Data folder prepare
    parser.add_argument("-a", "--access_type", type=str, help="LA or PA", default='LA')
    parser.add_argument("-d", "--path_to_database", type=str, help="dataset path", default='/data/neil/DS_10283_3336/')
    parser.add_argument("-f", "--path_to_features", type=str, help="features path",
                        default='/data2/neil/ASVspoof2019LA/')
    parser.add_argument("-p", "--path_to_protocol", type=str, help="protocol path",
                        default='/data/neil/DS_10283_3336/LA/ASVspoof2019_LA_cm_protocols/')
    parser.add_argument("-o", "--out_fold", type=str, help="output folder", required=True, default='./models/try/')

    # Dataset prepare
    parser.add_argument("--feat", type=str, help="which feature to use", default='LFCC')
    parser.add_argument("--feat_len", type=int, help="features length", default=750)
    parser.add_argument('--padding', type=str, default='repeat', choices=['zero', 'repeat', 'silence'],
                        help="how to pad short utterance")
    parser.add_argument("--enc_dim", type=int, help="encoding dimension", default=256)

    parser.add_argument('-m', '--model', help='Model arch', default='resnet',
                        choices=['resnet', 'lcnn'])

    # Training hyperparameters
    parser.add_argument('--num_epochs', type=int, default=1000, help="Number of epochs for training")
    parser.add_argument('--batch_size', type=int, default=64, help="Mini batch size for training")
    parser.add_argument('--lr', type=float, default=0.0005, help="learning rate")
    parser.add_argument('--lr_decay', type=float, default=0.5, help="decay learning rate")
    parser.add_argument('--interval', type=int, default=100, help="interval to decay lr")

    parser.add_argument('--beta_1', type=float, default=0.9, help="bata_1 for Adam")
    parser.add_argument('--beta_2', type=float, default=0.999, help="beta_2 for Adam")
    parser.add_argument('--eps', type=float, default=1e-8, help="epsilon for Adam")
    parser.add_argument("--gpu", type=str, help="GPU index", default="1")
    parser.add_argument('--num_workers', type=int, default=0, help="number of workers")

    parser.add_argument('--add_loss', type=str, default="ocsoftmax",
                        choices=[None, 'ocsoftmax'], help="add other loss for one-class training")
    parser.add_argument('--weight_loss', type=float, default=1, help="weight for other loss")
    parser.add_argument('--r_real', type=float, default=0.9, help="r_real for ocsoftmax loss")
    parser.add_argument('--r_fake', type=float, default=0.2, help="r_fake for ocsoftmax loss")
    parser.add_argument('--alpha', type=float, default=20, help="scale factor for angular isolate loss")

    parser.add_argument('--test_only', action='store_true', help="test the trained model in case the test crash sometimes or another test method")
    parser.add_argument('--continue_training', action='store_true', help="continue training with trained model")

    parser.add_argument('--AUG', type=str2bool, nargs='?', const=True, default=False,
                        help="whether to use device_augmentation in training")
    parser.add_argument('--MT_AUG', type=str2bool, nargs='?', const=True, default=False,
                        help="whether to use device_multitask_augmentation in training")
    parser.add_argument('--ADV_AUG', type=str2bool, nargs='?', const=True, default=False,
                        help="whether to use device_adversarial_augmentation in training")
    parser.add_argument('--lambda_', type=float, default=0.05, help="lambda for gradient reversal layer")
    parser.add_argument('--lr_d', type=float, default=0.0001, help="learning rate")

    parser.add_argument('--pre_train', action='store_true', help="whether to pretrain the model")
    parser.add_argument('--test_on_eval', action='store_true',
                        help="whether to run EER on the evaluation set")

    args = parser.parse_args()

    # Change this to specify GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Set seeds
    setup_seed(args.seed)

    if args.test_only or args.continue_training:
        pass
    else:
        # Path for output data
        if not os.path.exists(args.out_fold):
            os.makedirs(args.out_fold)
        else:
            shutil.rmtree(args.out_fold)
            os.mkdir(args.out_fold)

        # Folder for intermediate results
        if not os.path.exists(os.path.join(args.out_fold, 'checkpoint')):
            os.makedirs(os.path.join(args.out_fold, 'checkpoint'))
        else:
            shutil.rmtree(os.path.join(args.out_fold, 'checkpoint'))
            os.mkdir(os.path.join(args.out_fold, 'checkpoint'))

        # Path for input data
        # assert os.path.exists(args.path_to_database)
        assert os.path.exists(args.path_to_features)

        # Save training arguments
        with open(os.path.join(args.out_fold, 'args.json'), 'w') as file:
            file.write(json.dumps(vars(args), sort_keys=True, separators=('\n', ':')))

        with open(os.path.join(args.out_fold, 'train_loss.log'), 'w') as file:
            file.write("Start recording training loss ...\n")
        with open(os.path.join(args.out_fold, 'dev_loss.log'), 'w') as file:
            file.write("Start recording validation loss ...\n")
        with open(os.path.join(args.out_fold, 'test_loss.log'), 'w') as file:
            file.write("Start recording test loss ...\n")

    args.cuda = torch.cuda.is_available()
    print('Cuda device available: ', args.cuda)
    args.device = torch.device("cuda" if args.cuda else "cpu")

    return args

def adjust_learning_rate(args, lr, optimizer, epoch_num):
    lr = lr * (args.lr_decay ** (epoch_num // args.interval))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def shuffle(feat, tags, labels):
    shuffle_index = torch.randperm(labels.shape[0])
    feat = feat[shuffle_index]
    tags = tags[shuffle_index]
    labels = labels[shuffle_index]
    # this_len = this_len[shuffle_index]
    return feat, tags, labels

def train(args):
    torch.set_default_tensor_type(torch.FloatTensor)

    # initialize model
    if args.model == 'resnet':
        node_dict = {"CQCC": 4, "LFCC": 3}
        feat_model = ResNet(node_dict[args.feat], args.enc_dim, resnet_type='18', nclasses=2).to(args.device)
    elif args.model == 'lcnn':
        feat_model = LCNN(4, args.enc_dim, nclasses=2).to(args.device)

    if args.continue_training:
        feat_model = torch.load(os.path.join(args.out_fold, 'anti-spoofing_feat_model.pt')).to(args.device)
    # feat_model = nn.DataParallel(feat_model, list(range(torch.cuda.device_count())))  # for multiple GPUs
    feat_optimizer = torch.optim.Adam(feat_model.parameters(), lr=args.lr,
                                      betas=(args.beta_1, args.beta_2), eps=args.eps, weight_decay=0.0005)

    training_set = ASVspoof2019(args.access_type, args.path_to_features, 'train',
                                args.feat, feat_len=args.feat_len, padding=args.padding)
    validation_set = ASVspoof2019(args.access_type, args.path_to_features, 'dev',
                                  args.feat, feat_len=args.feat_len, padding=args.padding)
    if args.AUG or args.MT_AUG or args.ADV_AUG:
        training_set = ASVspoof2019LA_DeviceAdversarial(path_to_features="/data2/neil/ASVspoof2019LA/",
                                                        path_to_deviced="/dataNVME/neil/ASVspoof2019LADevice",
                                                        part="train",
                                                        feature=args.feat, feat_len=args.feat_len,
                                                        padding=args.padding)
        validation_set = ASVspoof2019LA_DeviceAdversarial(path_to_features="/data2/neil/ASVspoof2019LA/",
                                                          path_to_deviced="/dataNVME/neil/ASVspoof2019LADevice",
                                                          part="dev",
                                                          feature=args.feat, feat_len=args.feat_len,
                                                          padding=args.padding)
    if args.MT_AUG or args.ADV_AUG:
        classifier = ChannelClassifier(args.enc_dim, len(training_set.devices)+1, args.lambda_, ADV=args.ADV_AUG).to(args.device)
        classifier_optimizer = torch.optim.Adam(classifier.parameters(), lr=args.lr_d,
                                                betas=(args.beta_1, args.beta_2), eps=args.eps, weight_decay=0.0005)

    trainDataLoader = DataLoader(training_set, batch_size=args.batch_size,
                                 shuffle=True, num_workers=args.num_workers, collate_fn=training_set.collate_fn)
    valDataLoader = DataLoader(validation_set, batch_size=args.batch_size,
                               shuffle=True, num_workers=args.num_workers, collate_fn=validation_set.collate_fn)

    test_set = ASVspoof2019(args.access_type, args.path_to_features, "eval", args.feat,
                            feat_len=args.feat_len, padding=args.padding)
    testDataLoader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=test_set.collate_fn)

    feat, _, _, _, _ = training_set[23]
    print("Feature shape", feat.shape)

    criterion = nn.CrossEntropyLoss()

    if args.add_loss == "ocsoftmax":
        ocsoftmax = OCSoftmax(args.enc_dim, r_real=args.r_real, r_fake=args.r_fake, alpha=args.alpha).to(args.device)
        ocsoftmax.train()
        ocsoftmax_optimzer = torch.optim.SGD(ocsoftmax.parameters(), lr=args.lr)

    early_stop_cnt = 0
    prev_loss = 1e8

    if args.add_loss is None:
        monitor_loss = 'base_loss'
    else:
        monitor_loss = args.add_loss

    for epoch_num in tqdm(range(args.num_epochs)):
        genuine_feats, ip1_loader, tag_loader, idx_loader = [], [], [], []
        feat_model.train()
        trainlossDict = defaultdict(list)
        devlossDict = defaultdict(list)
        testlossDict = defaultdict(list)
        adjust_learning_rate(args, args.lr, feat_optimizer, epoch_num)
        if args.add_loss == "ocsoftmax":
            adjust_learning_rate(args, args.lr, ocsoftmax_optimzer, epoch_num)
        if args.MT_AUG or args.ADV_AUG:
            adjust_learning_rate(args, args.lr_d, classifier_optimizer, epoch_num)
        print('\nEpoch: %d ' % (epoch_num + 1))
        correct_m, total_m, correct_c, total_c, correct_v, total_v = 0, 0, 0, 0, 0, 0

        for i, (feat, audio_fn, tags, labels, channel) in enumerate(tqdm(trainDataLoader)):
            if args.AUG or args.MT_AUG or args.ADV_AUG:
                if i > int(len(training_set) / args.batch_size / (len(training_set.devices) + 1)): break
            feat = feat.transpose(2,3).to(args.device)
            tags = tags.to(args.device)
            labels = labels.to(args.device)
            feats, feat_outputs = feat_model(feat)
            feat_loss = criterion(feat_outputs, labels)
            trainlossDict['base_loss'].append(feat_loss.item())

            if args.add_loss == None:
                feat_optimizer.zero_grad()
                feat_loss.backward()
                feat_optimizer.step()

            if args.add_loss == "ocsoftmax":
                ocsoftmaxloss, _ = ocsoftmax(feats, labels)
                feat_loss = ocsoftmaxloss * args.weight_loss
                if epoch_num > 0 and (args.MT_AUG or args.ADV_AUG):
                    channel = channel.to(args.device)
                    classifier_out = classifier(feats)
                    _, predicted = torch.max(classifier_out.data, 1)
                    total_m += channel.size(0)
                    correct_m += (predicted == channel).sum().item()
                    device_loss = criterion(classifier_out, channel)
                    feat_loss += device_loss
                    trainlossDict["adv_loss"].append(device_loss.item())
                feat_optimizer.zero_grad()
                ocsoftmax_optimzer.zero_grad()
                trainlossDict[args.add_loss].append(ocsoftmaxloss.item())
                feat_loss.backward()
                feat_optimizer.step()
                ocsoftmax_optimzer.step()

            if (args.MT_AUG or args.ADV_AUG):
                channel = channel.to(args.device)
                feats, _ = feat_model(feat)
                feats = feats.detach()
                classifier_out = classifier(feats)
                _, predicted = torch.max(classifier_out.data, 1)
                total_c += channel.size(0)
                correct_c += (predicted == channel).sum().item()
                device_loss_c = criterion(classifier_out, channel)
                classifier_optimizer.zero_grad()
                device_loss_c.backward()
                classifier_optimizer.step()

            ip1_loader.append(feats)
            idx_loader.append((labels))
            tag_loader.append((tags))


            if epoch_num > 0 and (args.MT_AUG or args.ADV_AUG):
                with open(os.path.join(args.out_fold, "train_loss.log"), "a") as log:
                    log.write(str(epoch_num) + "\t" + str(i) + "\t" +
                              str(trainlossDict["adv_loss"][-1]) + "\t" +
                              str(100 * correct_m / total_m) + "\t" +
                              str(100 * correct_c / total_c) + "\t" +
                              str(trainlossDict[monitor_loss][-1]) + "\n")
            else:
                with open(os.path.join(args.out_fold, "train_loss.log"), "a") as log:
                    log.write(str(epoch_num) + "\t" + str(i) + "\t" +
                              str(trainlossDict[monitor_loss][-1]) + "\n")


        # Val the model
        # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
        feat_model.eval()
        with torch.no_grad():
            ip1_loader, tag_loader, idx_loader, score_loader = [], [], [], []
            # with trange(2) as v:
            # with trange(len(valDataLoader)) as v:
            #     for i in v:
            for i, (feat, audio_fn, tags, labels, channel) in enumerate(tqdm(valDataLoader)):
                if args.AUG or args.MT_AUG or args.ADV_AUG:
                    if i > int(len(validation_set) / args.batch_size / (len(validation_set.devices) + 1)): break
                feat = feat.transpose(2,3).to(args.device)

                tags = tags.to(args.device)
                labels = labels.to(args.device)

                feat, tags, labels = shuffle(feat, tags, labels)

                feats, feat_outputs = feat_model(feat)

                feat_loss = criterion(feat_outputs, labels)
                score = F.softmax(feat_outputs, dim=1)[:, 0]

                ip1_loader.append(feats)
                idx_loader.append((labels))
                tag_loader.append((tags))

                if args.add_loss == "ocsoftmax":
                    ocsoftmaxloss, score = ocsoftmax(feats, labels)
                    devlossDict[args.add_loss].append(ocsoftmaxloss.item())
                    if epoch_num > 0 and (args.MT_AUG or args.ADV_AUG):
                        channel = channel.to(args.device)
                        classifier_out = classifier(feats)
                        _, predicted = torch.max(classifier_out.data, 1)
                        total_v += channel.size(0)
                        correct_v += (predicted == channel).sum().item()
                        device_loss = criterion(classifier_out, channel)
                        devlossDict["adv_loss"].append(device_loss.item())

                score_loader.append(score)

            scores = torch.cat(score_loader, 0).data.cpu().numpy()
            labels = torch.cat(idx_loader, 0).data.cpu().numpy()
            eer = em.compute_eer(scores[labels == 0], scores[labels == 1])[0]
            other_eer = em.compute_eer(-scores[labels == 0], -scores[labels == 1])[0]
            eer = min(eer, other_eer)

            if epoch_num > 0 and (args.MT_AUG or args.ADV_AUG):
                with open(os.path.join(args.out_fold, "dev_loss.log"), "a") as log:
                    log.write(str(epoch_num) + "\t"+ "\t" +
                              str(np.nanmean(devlossDict["adv_loss"])) + "\t" +
                              str(100 * correct_v / total_v) + "\t" +
                              str(np.nanmean(devlossDict[monitor_loss])) + "\t" +
                              str(eer) + "\n")
            else:
                with open(os.path.join(args.out_fold, "dev_loss.log"), "a") as log:
                    log.write(str(epoch_num) + "\t" +
                              str(np.nanmean(devlossDict[monitor_loss])) + "\t" +
                              str(eer) +"\n")
            print("Val EER: {}".format(eer))


        if args.test_on_eval:
            with torch.no_grad():
                ip1_loader, tag_loader, idx_loader, score_loader = [], [], [], []
                for i, (feat, audio_fn, tags, labels, channel) in enumerate(tqdm(testDataLoader)):
                    feat = feat.transpose(2,3).to(args.device)
                    tags = tags.to(args.device)
                    labels = labels.to(args.device)
                    feats, feat_outputs = feat_model(feat)
                    feat_loss = criterion(feat_outputs, labels)
                    score = F.softmax(feat_outputs, dim=1)[:, 0]

                    ip1_loader.append(feats)
                    idx_loader.append((labels))
                    tag_loader.append((tags))

                    if args.add_loss == "ocsoftmax":
                        ocsoftmaxloss, score = ocsoftmax(feats, labels)
                        testlossDict[args.add_loss].append(ocsoftmaxloss.item())
                    score_loader.append(score)

                scores = torch.cat(score_loader, 0).data.cpu().numpy()
                labels = torch.cat(idx_loader, 0).data.cpu().numpy()
                eer = em.compute_eer(scores[labels == 0], scores[labels == 1])[0]
                other_eer = em.compute_eer(-scores[labels == 0], -scores[labels == 1])[0]
                eer = min(eer, other_eer)

                with open(os.path.join(args.out_fold, "test_loss.log"), "a") as log:
                    log.write(str(epoch_num) + "\t" + str(np.nanmean(testlossDict[monitor_loss])) + "\t" + str(eer) + "\n")
                print("Test EER: {}".format(eer))


        valLoss = np.nanmean(devlossDict[monitor_loss])
        # if args.add_loss == "isolate":
        #     print("isolate center: ", iso_loss.center.data)
        if (epoch_num + 1) % 1 == 0:
            torch.save(feat_model, os.path.join(args.out_fold, 'checkpoint',
                                                'anti-spoofing_feat_model_%d.pt' % (epoch_num + 1)))
            if args.add_loss == "ocsoftmax":
                loss_model = ocsoftmax
                torch.save(loss_model, os.path.join(args.out_fold, 'checkpoint',
                                                    'anti-spoofing_loss_model_%d.pt' % (epoch_num + 1)))
            else:
                loss_model = None

        if valLoss < prev_loss:
            # Save the model checkpoint
            torch.save(feat_model, os.path.join(args.out_fold, 'anti-spoofing_feat_model.pt'))
            if args.add_loss == "ocsoftmax":
                loss_model = ocsoftmax
                torch.save(loss_model, os.path.join(args.out_fold, 'anti-spoofing_loss_model.pt'))
            else:
                loss_model = None
            prev_loss = valLoss
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1

        if early_stop_cnt == 500:
            with open(os.path.join(args.out_fold, 'args.json'), 'a') as res_file:
                res_file.write('\nTrained Epochs: %d\n' % (epoch_num - 499))
            break
        # if early_stop_cnt == 1:
        #     torch.save(feat_model, os.path.join(args.out_fold, 'anti-spoofing_feat_model.pt')

            # print('Dev Accuracy of the model on the val features: {} % '.format(100 * feat_correct / total))

    return feat_model, loss_model



if __name__ == "__main__":
    args = initParams()
    if not args.test_only:
        _, _ = train(args)
    # model = torch.load(os.path.join(args.out_fold, 'anti-spoofing_feat_model.pt'))
    # if args.add_loss is None:
    #     loss_model = None
    # else:
    #     loss_model = torch.load(os.path.join(args.out_fold, 'anti-spoofing_loss_model.pt'))
    # # TReer_cm, TRmin_tDCF = test(args, model, loss_model, "train")
    # # VAeer_cm, VAmin_tDCF = test(args, model, loss_model, "dev")
    # TEeer_cm, TEmin_tDCF = test(args, model, loss_model)
    # with open(os.path.join(args.out_fold, 'args.json'), 'a') as res_file:
    #     # res_file.write('\nTrain EER: %8.5f min-tDCF: %8.5f\n' % (TReer_cm, TRmin_tDCF))
    #     # res_file.write('\nVal EER: %8.5f min-tDCF: %8.5f\n' % (VAeer_cm, VAmin_tDCF))
    #     res_file.write('\nTest EER: %8.5f min-tDCF: %8.5f\n' % (TEeer_cm, TEmin_tDCF))


    # # Test a checkpoint model
    # args = initParams()
    # model = torch.load(os.path.join(args.out_fold, 'checkpoint', 'anti-spoofing_feat_model_19.pt'))
    # loss_model = torch.load(os.path.join(args.out_fold, 'checkpoint', 'anti-spoofing_loss_model_19.pt'))
    # VAeer_cm, VAmin_tDCF = test(args, model, loss_model, "dev")
