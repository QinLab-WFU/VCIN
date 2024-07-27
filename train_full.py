# --- Base packages ---
import os
from random import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from PIL import Image
# --- PyTorch packages ---
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# --- Helper Packages ---
from tqdm import tqdm

# --- Project Packages ---
from utils import save, load, train, test, data_to_device, data_concatenate
from datasets import NIHCXR, MIMIC, NLMCXR
from losses import CELoss, CELossTotal, CELossTotalEval, CELossTransfer, CELossShift
from models import CNN, MVCNN, TNN, Classifier, Generator, ClsGen, ClsGenInt
from baselines.transformer.models import LSTM_Attn, Transformer, GumbelTransformer
from baselines.rnn.models import ST
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor import Meteor
from pycocoevalcap.rouge import Rouge


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
def compute_scores(gts, res):
    """
    Performs the MS COCO evaluation using the Python 3 implementation (https://github.com/salaniz/pycocoevalcap)

    :param gts: Dictionary with the image ids and their gold captions,
    :param res: Dictionary with the image ids ant their generated captions
    :print: Evaluation score (the mean of the scores of all the instances) for each measure
    """

    # Set up scorers
    scorers = [
        (Bleu(4), ["BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L")
    ]
    eval_res = {}
    # Compute score for each metric
    for scorer, method in scorers:
        try:
            score, scores = scorer.compute_score(gts, res, verbose=0)
        except TypeError:
            score, scores = scorer.compute_score(gts, res)
        if type(method) == list:
            for sc, m in zip(score, method):
                eval_res[m] = sc
        else:
            eval_res[method] = score
    return eval_res
# --- Helper Functions ---
def find_optimal_cutoff(target, predicted):
    fpr, tpr, threshold = metrics.roc_curve(target, predicted)
    gmeans = np.sqrt(tpr * (1-fpr))
    ix = np.argmax(gmeans)
    return threshold[ix]

def infer(data_loader, model, device='cpu', threshold=None):

    model.eval()
    outputs = []
    targets = []
    history = []

    with torch.no_grad():
        prog_bar = tqdm(data_loader)
        for i, (source, target) in enumerate(prog_bar):
            source = data_to_device(source, device)
            target = data_to_device(target, device)

            # Use single input if there is no clinical history
            if threshold != None:
                output = model(image=source[0], history=source[3], threshold=threshold)
                # output = model(image=source[0], threshold=threshold)
                # output = model(image=source[0], history=source[3], label=source[2])
                # output = model(image=source[0], label=source[2])
            else:
                # output = model(source[0], source[1])
                output = model(source[0])

            hi=source[3]
            outputs.append(data_to_device(output))
            targets.append(data_to_device(target))
            history.append(data_to_device(hi))

        outputs = data_concatenate(outputs)
        targets = data_concatenate(targets)
        history = data_concatenate(history)

    
    return outputs, targets,history

def gen(model):
    txt_test_outputs, txt_test_targets,history= infer(test_loader, model, device='cuda', threshold=0.25)
    gen_outputs = txt_test_outputs[0]
    gen_targets = txt_test_targets[0]

    out_file_ref = open('outputs/x_{}_{}_{}_{}_Ref.txt'.format(DATASET_NAME, MODEL_NAME, BACKBONE_NAME, COMMENT),
                        'w')
    out_file_hyp = open('outputs/x_{}_{}_{}_{}_Hyp.txt'.format(DATASET_NAME, MODEL_NAME, BACKBONE_NAME, COMMENT),
                        'w')
    out_file_lbl = open('outputs/x_{}_{}_{}_{}_Lbl.txt'.format(DATASET_NAME, MODEL_NAME, BACKBONE_NAME, COMMENT),
                        'w')
    out_file_hi = open('outputs/x_{}_{}_{}_{}_hi.txt'.format(DATASET_NAME, MODEL_NAME, BACKBONE_NAME, COMMENT),
                        'w')

    log = dict()
    test_gts, test_res = [], []
    for i in range(len(gen_outputs)):
        candidate = ''
        for j in range(len(gen_outputs[i])):
            tok = dataset.vocab.id_to_piece(int(gen_outputs[i, j]))
            if tok == '</s>':
                break  # Manually stop generating token after </s> is reached
            elif tok == '<s>':
                continue
            elif tok == '▁':  # space
                if len(candidate) and candidate[-1] != ' ':
                    candidate += ' '
            elif tok in [',', '.', '-', ':']:  # or not tok.isalpha():
                if len(candidate) and candidate[-1] != ' ':
                    candidate += ' ' + tok + ' '
                else:
                    candidate += tok + ' '
            else:  # letter
                candidate += tok
        out_file_hyp.write(candidate + '\n')

        candi = ''

        for j in range(len(history[i])):
            tok = dataset.vocab.id_to_piece(int(history[i, j]))
            if tok == '</s>':
                break  # Manually stop generating token after </s> is reached
            elif tok == '<s>':
                continue
            elif tok == '▁':  # space
                if len(candi) and candi[-1] != ' ':
                    candi += ' '
            elif tok in [',', '.', '-', ':']:  # or not tok.isalpha():
                if len(candi) and candi[-1] != ' ':
                    candi += ' ' + tok + ' '
                else:
                    candi += tok + ' '
            else:  # letter
                candi += tok
        out_file_hi.write(candi + '\n')

        reference = ''
        reference=reference+str(i)+' '
        for j in range(len(gen_targets[i])):
            tok = dataset.vocab.id_to_piece(int(gen_targets[i, j]))
            if tok == '</s>':
                break
            elif tok == '<s>':
                continue
            elif tok == '▁':  # space
                if len(reference) and reference[-1] != ' ':
                    reference += ' '
            elif tok in [',', '.', '-', ':']:  # or not tok.isalpha():
                if len(reference) and reference[-1] != ' ':
                    reference += ' ' + tok + ' '
                else:
                    reference += tok + ' '
            else:  # letter
                reference += tok
        out_file_ref.write(reference + '\n')
        test_res.append(candidate)
        test_gts.append(reference)
    for i in tqdm(range(len(test_data))):
        target = test_data[i][1]  # caption, label
        out_file_lbl.write(' '.join(map(str, target[1])) + '\n')
    test_met = compute_scores({i: [gt] for i, gt in enumerate(test_gts)},
                              {i: [re] for i, re in enumerate(test_res)})
    log.update(**{'test_' + k: v for k, v in test_met.items()})

    print(log)
    return log



def generate_heatmap(image,weights):
    image = image.permute(1, 2, 0)
    image = image.detach().cpu().numpy()
    min_pixel_value = np.min(image)
    max_pixel_value = np.max(image)
    image = (image - min_pixel_value) / (max_pixel_value - min_pixel_value) * 255

    weights = weights.detach().cpu().numpy()
    height,width,_=image.shape

    weights = weights - np.min(weights)
    weights = weights / np.max(weights)
    weights = cv2.resize(weights, (width, height))
    weights = np.uint8(255 * weights)
    heatmap = cv2.applyColorMap(weights, cv2.COLORMAP_JET)
    result = heatmap * 0.5 + image * 0.5
    return result


# --- Hyperparameters ---
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
torch.set_num_threads(1)
torch.manual_seed(seed=123)




RELOAD = False # True / False
PHASE = 'TRAIN' # TRAIN / TEST
DATASET_NAME = 'NLMCXR' # NLMCXR / MIMIC
BACKBONE_NAME = 'DenseNet121' # ResNeSt50 / ResNet50 / DenseNet121
MODEL_NAME = 'ClsGenInt' # ClsGen / ClsGenInt


if DATASET_NAME == 'MIMIC':
    EPOCHS = 50 # Start overfitting after 20 epochs
    BATCH_SIZE = 8 if PHASE == 'TRAIN' else 64 # 128 # Fit 4 GPUs
    MILESTONES = [25] # Reduce LR by 10 after reaching milestone epochs
    
elif DATASET_NAME == 'NLMCXR':
    EPOCHS = 50 # Start overfitting after 20 epochs
    BATCH_SIZE = 8 if PHASE == 'TRAIN' else 64 # Fit 4 GPUs
    MILESTONES = [25] # Reduce LR by 10 after reaching milestone epochs
    
else:
    raise ValueError('Invalid DATASET_NAME')

if __name__ == "__main__":


    # --- Choose Inputs/Outputs
    if MODEL_NAME in ['ClsGen', 'ClsGenInt']:
        SOURCES = ['image','caption','label','history']
        TARGETS = ['caption','label']
        KW_SRC = ['image','caption','label','history']
        KW_TGT = None
        KW_OUT = None
        
    else:
        raise ValueError('Invalid BACKBONE_NAME')
        
    # --- Choose a Dataset ---
    if DATASET_NAME == 'MIMIC':
        INPUT_SIZE = (256,256)
        MAX_VIEWS = 2
        NUM_LABELS = 114
        NUM_CLASSES = 4
        
        dataset = MIMIC('./MIMIC/', INPUT_SIZE, view_pos=['AP','PA','LATERAL'], max_views=MAX_VIEWS, sources=SOURCES, targets=TARGETS)
        train_data, val_data, test_data = dataset.get_subsets(pvt=0.9, seed=55, generate_splits=True, debug_mode=False, train_phase=(PHASE == 'TRAIN'))
        
        VOCAB_SIZE = len(dataset.vocab)
        POSIT_SIZE = dataset.max_len
        COMMENT = 'MaxView{}_NumLabel{}_{}History'.format(MAX_VIEWS, NUM_LABELS, 'No' if 'history' not in SOURCES else '')
            
    elif DATASET_NAME == 'NLMCXR':
        INPUT_SIZE = (256,256)
        MAX_VIEWS = 2
        NUM_LABELS = 114
        NUM_CLASSES = 4

        dataset = NLMCXR('./NLMCXR/', INPUT_SIZE, view_pos=['AP','PA','LATERAL'], max_views=MAX_VIEWS, sources=SOURCES, targets=TARGETS)
        train_data, val_data, test_data = dataset.get_subsets(seed=55)
        
        VOCAB_SIZE = len(dataset.vocab)
        POSIT_SIZE = dataset.max_len
        COMMENT = 'MaxView{}_NumLabel{}_{}History'.format(MAX_VIEWS, NUM_LABELS, 'No' if 'history' not in SOURCES else '')
        
    else:
        raise ValueError('Invalid DATASET_NAME')
    checkpoint_path_from = 'checkpoints/{}_{}_{}_{}.pt'.format(DATASET_NAME, MODEL_NAME, BACKBONE_NAME, COMMENT)
    checkpoint_path_to = 'checkpoints/{}_{}_{}_{}.pt'.format(DATASET_NAME, MODEL_NAME, BACKBONE_NAME, COMMENT)
    # --- Choose a Backbone ---

    backbone = torch.hub.load('pytorch/vision:v0.5.0', 'densenet121', pretrained=True)
    FC_FEATURES = 1024



    LR = 3e-4 # Fastest LR
    WD = 1e-2 # Avoid overfitting with L2 regularization
    DROPOUT = 0.1 # Avoid overfitting
    NUM_EMBEDS = 256
    FWD_DIM = 256

    NUM_HEADS = 8
    NUM_LAYERS = 1

    cnn = CNN(backbone, BACKBONE_NAME)
    cnn = MVCNN(cnn)
    tnn = TNN(embed_dim=NUM_EMBEDS, num_heads=NUM_HEADS, fwd_dim=FWD_DIM, dropout=DROPOUT, num_layers=NUM_LAYERS, num_tokens=VOCAB_SIZE, num_posits=POSIT_SIZE)

    # Not enough memory to run 8 heads and 12 layers, instead 1 head is enough
    NUM_HEADS = 1
    NUM_LAYERS = 12

    cls_model = Classifier(num_topics=NUM_LABELS, num_states=NUM_CLASSES, cnn=cnn, tnn=tnn, fc_features=FC_FEATURES, embed_dim=NUM_EMBEDS, num_heads=NUM_HEADS, dropout=DROPOUT)
    gen_model = Generator(num_tokens=VOCAB_SIZE, num_posits=POSIT_SIZE, embed_dim=NUM_EMBEDS, num_heads=NUM_HEADS, fwd_dim=FWD_DIM, dropout=DROPOUT, num_layers=NUM_LAYERS)

    model = ClsGen(cls_model, gen_model, NUM_LABELS, NUM_EMBEDS)
    criterion = CELossTotal(ignore_index=3)
        

    
    # --- Main program ---
    train_loader = data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, drop_last=True)
    val_loader = data.DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)
    test_loader = data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

    model = nn.DataParallel(model).cuda()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=WD)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=MILESTONES)

    print('Total Parameters:', sum(p.numel() for p in model.parameters()))
    
    last_epoch = -1
    best_metric = 1e9


    if RELOAD:
        last_epoch, (best_metric, test_metric) = load(checkpoint_path_from, model, optimizer, scheduler) # Reload
        # last_epoch, (best_metric, test_metric) = load(checkpoint_path_from, model) # Fine-tune
        print('Reload From: {} | Last Epoch: {} | Validation Metric: {} | Test Metric: {}'.format(checkpoint_path_from, last_epoch, best_metric, test_metric))
        # plot(model)
        log=gen(model)



    if PHASE == 'TRAIN' :
        scaler = torch.cuda.amp.GradScaler()
        best_blue = 0
        for epoch in range(last_epoch + 1, EPOCHS):
            print('Epoch:', epoch)
            print(optimizer.state_dict()['param_groups'][0]['lr'])
            train_loss = train(train_loader, model, optimizer, criterion, device='cuda', kw_src=KW_SRC, kw_tgt=KW_TGT,
                               kw_out=KW_OUT, scaler=scaler)
            val_loss = test(val_loader, model, criterion, device='cuda', kw_src=KW_SRC, kw_tgt=KW_TGT, kw_out=KW_OUT,
                            return_results=False)
            # test_loss = test(test_loader, model, criterion, device='cuda', kw_src=KW_SRC, kw_tgt=KW_TGT, kw_out=KW_OUT,
            #                  return_results=False)
            test_loss, test_outputs, test_targets = test(test_loader, model, criterion, device='cuda', kw_src=KW_SRC,
                                                         kw_tgt=KW_TGT, kw_out=KW_OUT, select_outputs=[1])

            scheduler.step()

            log = gen(model)

            if log['test_BLEU_1']+log['test_BLEU_4'] > best_blue:
                best_blue = log['test_BLEU_1']+log['test_BLEU_4']
                save(checkpoint_path_to, model, optimizer, scheduler, epoch, (val_loss, test_loss))
                print('Saved To:', checkpoint_path_to)


            test_auc = []
            test_f1 = []
            test_prc = []
            test_rec = []
            test_acc = []

            threshold = 0.25
            NUM_LABELS = 14
            for i in range(NUM_LABELS):
                try:
                    test_auc.append(metrics.roc_auc_score(test_targets.cpu()[..., i], test_outputs.cpu()[..., i, 1]))
                    test_f1.append(metrics.f1_score(test_targets.cpu()[..., i], test_outputs.cpu()[..., i, 1] > threshold))
                    test_prc.append(
                        metrics.precision_score(test_targets.cpu()[..., i], test_outputs.cpu()[..., i, 1] > threshold))
                    test_rec.append(
                        metrics.recall_score(test_targets.cpu()[..., i], test_outputs.cpu()[..., i, 1] > threshold))
                    test_acc.append(
                        metrics.accuracy_score(test_targets.cpu()[..., i], test_outputs.cpu()[..., i, 1] > threshold))

                except:
                    print('An error occurs for label', i)

            test_auc = np.mean([x for x in test_auc if str(x) != 'nan'])
            test_f1 = np.mean([x for x in test_f1 if str(x) != 'nan'])
            test_prc = np.mean([x for x in test_prc if str(x) != 'nan'])
            test_rec = np.mean([x for x in test_rec if str(x) != 'nan'])
            test_acc = np.mean([x for x in test_acc if str(x) != 'nan'])

            print('Accuracy       : {}'.format(test_acc))
            print('Macro AUC      : {}'.format(test_auc))
            print('Macro F1       : {}'.format(test_f1))
            print('Macro Precision: {}'.format(test_prc))
            print('Macro Recall   : {}'.format(test_rec))
            print('Micro AUC      : {}'.format(
                metrics.roc_auc_score(test_targets.cpu()[..., :NUM_LABELS] == 1, test_outputs.cpu()[..., :NUM_LABELS, 1],
                                      average='micro')))
            print('Micro F1       : {}'.format(metrics.f1_score(test_targets.cpu()[..., :NUM_LABELS] == 1,
                                                                test_outputs.cpu()[..., :NUM_LABELS, 1] > threshold,
                                                                average='micro')))
            print('Micro Precision: {}'.format(metrics.precision_score(test_targets.cpu()[..., :NUM_LABELS] == 1,
                                                                       test_outputs.cpu()[..., :NUM_LABELS, 1] > threshold,
                                                                       average='micro')))
            print('Micro Recall   : {}'.format(metrics.recall_score(test_targets.cpu()[..., :NUM_LABELS] == 1,
                                                                    test_outputs.cpu()[..., :NUM_LABELS, 1] > threshold,
                                                                    average='micro')))

    elif PHASE == 'TEST':
        # Output the file list for inspection
        out_file_img = open('outputs/{}_{}_{}_{}_Img.txt'.format(DATASET_NAME, MODEL_NAME, BACKBONE_NAME, COMMENT), 'w')
        for i in range(len(test_data.idx_pidsid)):
            out_file_img.write(test_data.idx_pidsid[i][0] + ' ' + test_data.idx_pidsid[i][1] + '\n')

        test_loss, test_outputs, test_targets = test(test_loader, model, criterion, device='cuda', kw_src=KW_SRC, kw_tgt=KW_TGT, kw_out=KW_OUT, select_outputs=[1])

        test_auc = []
        test_f1 = []
        test_prc = []
        test_rec = []
        test_acc = []

        threshold = 0.25
        NUM_LABELS = 14
        for i in range(NUM_LABELS):
            try:
                test_auc.append(metrics.roc_auc_score(test_targets.cpu()[...,i], test_outputs.cpu()[...,i,1]))
                test_f1.append(metrics.f1_score(test_targets.cpu()[...,i], test_outputs.cpu()[...,i,1] > threshold))
                test_prc.append(metrics.precision_score(test_targets.cpu()[...,i], test_outputs.cpu()[...,i,1] > threshold))
                test_rec.append(metrics.recall_score(test_targets.cpu()[...,i], test_outputs.cpu()[...,i,1] > threshold))
                test_acc.append(metrics.accuracy_score(test_targets.cpu()[...,i], test_outputs.cpu()[...,i,1] > threshold))

            except:
                print('An error occurs for label', i)

        test_auc = np.mean([x for x in test_auc if str(x) != 'nan'])
        test_f1 = np.mean([x for x in test_f1 if str(x) != 'nan'])
        test_prc = np.mean([x for x in test_prc if str(x) != 'nan'])
        test_rec = np.mean([x for x in test_rec if str(x) != 'nan'])
        test_acc = np.mean([x for x in test_acc if str(x) != 'nan'])

        print('Accuracy       : {}'.format(test_acc))
        print('Macro AUC      : {}'.format(test_auc))
        print('Macro F1       : {}'.format(test_f1))
        print('Macro Precision: {}'.format(test_prc))
        print('Macro Recall   : {}'.format(test_rec))
        print('Micro AUC      : {}'.format(metrics.roc_auc_score(test_targets.cpu()[...,:NUM_LABELS] == 1, test_outputs.cpu()[...,:NUM_LABELS,1], average='micro')))
        print('Micro F1       : {}'.format(metrics.f1_score(test_targets.cpu()[...,:NUM_LABELS] == 1, test_outputs.cpu()[...,:NUM_LABELS,1] > threshold, average='micro')))
        print('Micro Precision: {}'.format(metrics.precision_score(test_targets.cpu()[...,:NUM_LABELS] == 1, test_outputs.cpu()[...,:NUM_LABELS,1] > threshold, average='micro')))
        print('Micro Recall   : {}'.format(metrics.recall_score(test_targets.cpu()[...,:NUM_LABELS] == 1, test_outputs.cpu()[...,:NUM_LABELS,1] > threshold, average='micro')))
