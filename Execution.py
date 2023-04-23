import datasets
import torch
import torch.nn as nn
from torch.autograd import Variable
from transformers import AutoTokenizer, AutoModel, get_scheduler, get_constant_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast,GradScaler
import tqdm
import numpy
import math
import matplotlib.pyplot as plt
from IPython import display
import random
import sys
from ModifiedDataLoader import loader
from TrainerUtils import multi_train, train, evaluation, test
import InitNetwork

import argparse

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder', type=str, default='xlm-roberta-large', help='model type(name in transformers)')
    parser.add_argument('--encoder_lr', type=float, default='1e-5', help='learning rate of encoder network')
    parser.add_argument('--head_lr', type=float, default='1e-3', help='learning rate of head network (for fuser/reasoning network)')
    parser.add_argument('--sample_count', type=int, default='1', help='counts of sampling in one batch')
    parser.add_argument('--entry_count', type=int, default='16', help='number of entries in each sampling')
    parser.add_argument('--epoch', type=int, default='7', help='batch size')
    parser.add_argument('--datasets', nargs='+', type=str, default=['cosmos_qa'], help='involved datasets')
    parser.add_argument('--warmups',type=float,default='0.16',help='proportion of warming up iters')
    parser.add_argument('--evaluation_only', type=bool, default=False, help='evaluation only')
    parser.add_argument('--weights', type=str, default='Models/model_params.pkl', help='path of trained model weights')
	
    opt = parser.parse_args()

    #print(vars(opt))

    return opt

def main(opt):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BIDIRECTION = True
    NEW_TOKENS = ['<P>','</P>','<Q>','</Q>','<O>','</O>']
    BATCH_SIZE = opt.entry_count
    ACCUMULATION_ITER = opt.sample_count
    FT_LEARNING_RATE = opt.encoder_lr
    LEARNING_RATE = opt.head_lr
    EPOCH = opt.epoch
    WARMUPS = opt.warmups
    NAME_DATASETS = opt.datasets

    print('loading model weights from huggingface')
    TOKENIZER = AutoTokenizer.from_pretrained(opt.encoder)
    ENCODER = AutoModel.from_pretrained(opt.encoder)
    HIDDEN = ENCODER.embeddings.word_embeddings.embedding_dim

    print('constructing data loaders for selected datasets')
    datasets = []
    for name in NAME_DATASETS:
        datasets.append(loader(name,BATCH_SIZE,NEW_TOKENS))

    if not opt.evaluation_only:
        CURRENT_EPOCH = 1

        print('initializing training')
        TOTAL_ITERS = int(EPOCH*sum(len(dataset.train_dataloader) for dataset in datasets)/ACCUMULATION_ITER)
        WARMING_ITERS = int(WARMUPS*TOTAL_ITERS)
        train_rec = []
        loss_rec = []
        eval_record = []

        model = InitNetwork.FCC_Network(TOKENIZER,ENCODER,BATCH_SIZE,HIDDEN,NEW_TOKENS,fuse=True,fuser_depth=3)
        model.to(DEVICE)
        params_dict = [{'params': model.encoder.parameters(), 'lr':FT_LEARNING_RATE},
        {'params': model.transformer_fuser.parameters(), 'lr':LEARNING_RATE},
        {'params': model.reasoning_net.parameters(), 'lr':LEARNING_RATE}]
        optim = torch.optim.AdamW(params_dict)
        lr_scheduler = get_scheduler('linear',
            optimizer=optim,
            num_warmup_steps=WARMING_ITERS,
            num_training_steps=WARMING_ITERS+TOTAL_ITERS)
        scaler = GradScaler()

        train_acc, train_loss = multi_train(DEVICE,model,optim,scaler,lr_scheduler,datasets,WARMING_ITERS,BIDIRECTION,CURRENT_EPOCH,ACCUMULATION_ITER)
        train_rec+=train_acc
        loss_rec+=train_loss
        warmups = []
        for dataset in datasets:
            dataset.reconstruct_loader('train')
    
            accuracy = evaluation(DEVICE,model,dataset,BIDIRECTION)
            warmups.append(accuracy)

        print('accuracy after warmup: {}'.format(dict(zip(NAME_DATASETS,warmups))))

        for i in range(EPOCH):
    
            train_acc, train_loss = multi_train(DEVICE,model,optim,scaler,lr_scheduler,datasets,TOTAL_ITERS,BIDIRECTION,CURRENT_EPOCH,ACCUMULATION_ITER)
            train_rec+=train_acc
            loss_rec+=train_loss
                  
            evals = []
            for dataset in datasets:
                dataset.reconstruct_loader('train')
            
                accuracy = evaluation(DEVICE,model,dataset,BIDIRECTION)
                evals.append(accuracy)
                
            current_evals = dict(zip(NAME_DATASETS,evals))
            eval_record.append(current_evals)
            
            torch.save(model.state_dict(), "model_params.pkl")
            
            CURRENT_EPOCH += 1

        print(eval_record)

    else:
        print('initializing testing')
        model = InitNetwork.FCC_Network(TOKENIZER,ENCODER,BATCH_SIZE,HIDDEN,NEW_TOKENS,fuse=True,fuser_depth=3)
        model.load_state_dict(torch.load(opt.weights))
        moedl.to(DEVICE)

        if 'cosmos_qa' or 'MuSeRC' in NAME_DATASETS:
            sys.exit('datasets consist of hosted datasets')

        else:
            evals = []
            for dataset in datasets:
                accuracy = test(DEVICE,model,dataset,BIDIRECTION)
                evals.append(accuracy)
                
            print(dict(zip(NAME_DATASETS,evals)))


if __name__ == '__main__':
    main(parse_opt())