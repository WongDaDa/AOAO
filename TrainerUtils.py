import random
import torch.nn as nn
from torch.cuda.amp import autocast,GradScaler
import numpy
import torch
from IPython import display

def multi_train(device,model,optim,scaler,lr_scheduler,datasets,total_iters,bi_direction,current,accumulator):
    '''Unified Tuning with smooth sampling'''
    model.encoder.train()
    accuracy_record = []
    loss_record = []
    sample_weight = []
    for sets in datasets:
        sample_weight.append(len(sets.train_dataloader))
        
    original_prob = list(n/sum(sample_weight) for n in sample_weight)
    smooth_prob = list(s**0.5 for s in original_prob)
    sampling = list(n/sum(smooth_prob) for n in smooth_prob)
    iters = 1
    t_loss = 0
    accurate = 0
    
    for i in range(total_iters*accumulator):
        sample_target = random.choices(datasets,weights=sampling)[0]
        data = sample_target.get_next('train')
        x,y,len_p,len_q,len_o,len_oi,trun_count = sample_target.call_encoder(data,model.tokenizer,bi_direction)
        
        with autocast():
            predict,loss = model.forward(x.to(device),y.to(device),len_p,len_q,len_oi,trun_count,bi_direction)
            scaler.scale(loss/accumulator).backward()
                        
            accurate += numpy.sum(torch.argmax(predict,dim=1).cpu().detach().numpy()==y.cpu().detach().numpy())
            t_loss += loss.item()/accumulator
            
            torch.cuda.empty_cache()
            
            if (i+1)%accumulator==0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
                scaler.step(optim)
                scaler.update()
                lr_scheduler.step()
                optim.zero_grad()
                
                accuracy = accurate/(accumulator*model.batch_size)
                print('Iter {} in epoch: {}'.format(iters,current))
                print('current accuracy in this iter: {}'.format(accuracy))
                print('current loss in this iter: {}'.format(t_loss))
                display.clear_output(wait=True)
                
                accuracy_record.append(accuracy)
                loss_record.append(t_loss)
                t_loss = 0
                accurate = 0
                iters += 1
        
    return accuracy_record,loss_record

def train(device,model,optim,scaler,lr_scheduler,dataset,bi_direction,current,accumulator):
    '''Separate tuning'''
    model.encoder.train()
    accuracy_record = []
    iters = 1
    t_loss = 0
    accurate = 0
    
    for i in range(total_iters*accumulator):
        data = sample_target.get_next('train')
        x,y,len_p,len_q,len_o,len_oi,trun_count = dataset.call_encoder(data,model.tokenizer,bi_direction)
        
        with autocast():
            predict,loss = model.forward(x.to(device),y.to(device),len_p,len_q,len_oi,trun_count,bi_direction)
            scaler.scale(loss/accumulator).backward()
            
            accurate += numpy.sum(torch.argmax(predict,dim=1).cpu().detach().numpy()==y.cpu().detach().numpy())
            t_loss += loss.item()/accumulator
            
            torch.cuda.empty_cache()
            
            if (i+1)%accumulator==0:
                scaler.step(optim)
                scaler.update()
                lr_scheduler.step()
                optim.zero_grad()
                
                accuracy = accurate/(accumulator*dataset.batch_size)
                print('Iter {} in epoch: {}'.format(iters,current))
                print('current accuracy in this iter: {}'.format(accuracy))
                print('current loss in this iter: {}'.format(t_loss))
                display.clear_output(wait=True)
                
                accuracy_record.append(accuracy)
                t_loss = 0
                accurate = 0
                iters += 1
        
    return accuracy_record

def evaluation(device,model,dataset,bi_direction):
    model.encoder.eval()
    correct = 0
    iters = 0
    
    for i, data in enumerate(dataset.eval_dataloader):
        iters += 1
        x,y,len_p,len_q,len_o,len_oi,trun_count = dataset.call_encoder(data,model.tokenizer,bi_direction)
        with torch.no_grad():
            with autocast():
                predict, _ = model.forward(x.to(device),y.to(device),len_p,len_q,len_oi,trun_count,bi_direction)
                correct += numpy.sum(torch.argmax(predict,dim=1).cpu().detach().numpy()==y.cpu().detach().numpy())
    
    return correct/len(dataset.eval_dataloader)

def test(device,model,dataset,bi_direction):
    model.encoder.eval()
    correct = 0
    iters = 0
    
    for i, data in enumerate(dataset.test_dataloader):
        iters += 1
        x,y,len_p,len_q,len_o,len_oi,trun_count = dataset.call_encoder(data,model.tokenizer,bi_direction)
        with torch.no_grad():
            with autocast():
                predict, _ = model.forward(x.to(device),y.to(device),len_p,len_q,len_oi,trun_count,bi_direction)
                correct += numpy.sum(torch.argmax(predict,dim=1).cpu().detach().numpy()==y.cpu().detach().numpy())
    
    return correct/len(dataset.test_dataloader)