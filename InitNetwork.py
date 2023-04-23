import torch
import torch.nn as nn
from torch.cuda.amp import autocast

class FCC_Network(nn.Module):
    
    def __init__(self,tokenizer,encoder,batch_size,hidden_size,new_tokens,fuse=False,fuser_depth=3):
        super(FCC_Network,self).__init__()
        self.hidden_size = hidden_size
        self.fuse = fuse
        self.fuser_depth = fuser_depth
        self.new_tokens = new_tokens
        self.batch_size = batch_size
        
        self.loss_fn = nn.CrossEntropyLoss()
        self.tokenizer = self.modify_tokenizer(tokenizer,new_tokens)
        self.encoder = self.modify_embeddings(encoder)
        self.reasoning_net = nn.Sequential(nn.Linear(self.hidden_size,self.hidden_size),
                                           nn.ReLU(),
                                           nn.Linear(self.hidden_size,self.hidden_size),
                                           nn.ReLU(),
                                           nn.Linear(self.hidden_size,1,bias=False))
        
        self.transformer_fuser = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=self.hidden_size, nhead=16,dropout=0), num_layers=self.fuser_depth)
        
    def modify_tokenizer(self,tokenizer,new_tokens):
        tokenizer.add_tokens(new_tokens)
        return tokenizer
        
    def modify_embeddings(self,encoder):
        encoder.resize_token_embeddings(len(self.tokenizer))
        return encoder

    @autocast()
    def forward(self,x,y,p_length,q_length,opt_length,trun_count,bi_direction):
        
        embeddings = self.encoder(**x).last_hidden_state
        
        pool_p_head = embeddings[:,0:1,:].contiguous().view(trun_count,-1,1,self.hidden_size)
        p_features = torch.mean(pool_p_head,dim=0)
        if bi_direction:
            pool_p_tail = embeddings[:,p_length-1:p_length,:].contiguous().view(trun_count,-1,1,self.hidden_size)
            p_tail_feature = torch.mean(pool_p_tail,dim=0)
            p_features = torch.mean(torch.cat((p_features,p_tail_feature),dim=1),dim=1,keepdim=True)
        
        pool_q_head = embeddings[:,p_length:p_length+1,:].contiguous().view(trun_count,-1,1,self.hidden_size)
        q_features = torch.mean(pool_q_head,dim=0)
        if bi_direction:
            pool_q_tail = embeddings[:,p_length+q_length-1:p_length+q_length,:].contiguous().view(trun_count,-1,1,self.hidden_size)
            q_tail_feature = torch.mean(pool_q_tail,dim=0)
            q_features = torch.mean(torch.cat((q_features,q_tail_feature),dim=1),dim=1,keepdim=True)
        
        features = torch.add(p_features,q_features).repeat(1,len(opt_length)-1,1).permute(1,0,2)
        for i in range(len(opt_length)-1):
            pool_o_head = embeddings[:,p_length+q_length+sum(opt_length[0:i+1]):p_length+q_length+sum(opt_length[0:i+1])+1,:].contiguous().view(trun_count,-1,1,self.hidden_size)
            o_features = torch.mean(pool_o_head,dim=0)
            if bi_direction:
                pool_o_tail = embeddings[:,p_length+q_length+sum(opt_length[0:i+2])-1:p_length+q_length+sum(opt_length[0:i+2]),:].contiguous().view(trun_count,-1,1,self.hidden_size)
                o_tail_feature = torch.mean(pool_o_tail,dim=0)
                o_features = torch.mean(torch.cat((o_features,o_tail_feature),dim=1),dim=1,keepdim=True)
            
            features[i] = features[i].add(o_features.squeeze(1))
            
        result = torch.zeros(features.size(dim=1),features.size(dim=0))
        
        if self.fuse:
            fused_features = self.transformer_fuser(features)
            final = torch.add(features,fused_features)
            for i in range(features.size(dim=0)):
                result[:,i] = self.reasoning_net(final[i]).squeeze(-1)
        
        else:
            for i in range(features.size(dim=0)):
                result[:,i] = self.reasoning_net(features[i]).squeeze(-1)
            
        loss = self.loss_fn(result.float().cuda(),y.long())
        
            
        return result,loss

class Baseline_Network(nn.Module):
    
    def __init__(self,tokenizer,encoder,hidden_size):
        super(Baseline_Network,self).__init__()
        self.hidden_size = hidden_size
        self.loss_fn = nn.CrossEntropyLoss()
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.reasoning_net = nn.Sequential(nn.Linear(self.hidden_size,self.hidden_size),
                                           nn.ReLU(),
                                           nn.Linear(self.hidden_size,self.hidden_size),
                                           nn.ReLU(),
                                           nn.Linear(self.hidden_size,1,bias=False)
                                           )

    @autocast()
    def forward(self,x,y,trun_count,opt_num):
        
        cls_embeddings = self.encoder(**x).last_hidden_state[:,0,:]
        cls_embeddings = cls_embeddings.view(trun_count,opt_num,-1,1024)
        cls_embeddings = torch.mean(cls_embeddings,dim=0)
        result = torch.zeros(cls_embeddings.size(dim=1),cls_embeddings.size(dim=0))
        for i in range(opt_num):
            result[:,i] = self.reasoning_net(cls_embeddings[i,:,:]).squeeze(-1)
            
        loss = self.loss_fn(result.cuda(),y.long())
        
        return result,loss