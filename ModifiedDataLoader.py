import datasets
from torch.utils.data import Dataset, DataLoader
from transformers import tokenization_utils_base
import math
import torch
import pandas as pd
import numpy
import random

class pack_dataset():
    def __init__(self,dataset):
        self.dataset = dataset

    def __getitem__(self,idx):
        assert idx<len(self.dataset)
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)

class loader():

    def __init__(self,dataset_name,batch_size,new_tokens):
        self.batch_size = batch_size
        self.name = dataset_name
        self.new_tokens = new_tokens
        
        '''Modify data collection sources of datasets'''
        if self.name == 'race':
            self.call_encoder = self.race_pre_processing
            self.dataset = datasets.load_dataset(dataset_name,'all',split=['train','validation','test'])
            self.opt_num = 4
        elif self.name == 'cosmos_qa':
            self.call_encoder = self.cosmos_pre_processing
            self.dataset = datasets.load_dataset(dataset_name,split=['train','validation','test'])
            self.opt_num = 4
        elif self.name == 'dream':
            self.call_encoder = self.dream_pre_processing
            self.dataset = datasets.load_dataset(dataset_name,split=['train','validation','test'])
            for i in range(len(self.dataset)):
                self.dataset[i] = self.dataset[i].map(lambda batch: {"dialogue": ' '.join(batch['dialogue'])})
            self.opt_num = 3
        elif self.name == 'sagnikrayc/mctest':
            self.call_encoder = self.mctest_pre_processing
            self.dataset = datasets.load_dataset(dataset_name,split=['train','validation','test'])
            for i in range(len(self.dataset)):
                self.dataset[i] = self.dataset[i].map(lambda batch: {"answer_options": batch['answer_options'].values()})
            self.opt_num = 4
        elif self.name == 'c3':
            self.call_encoder = self.local_pre_processing
            self.dataset = [datasets.Dataset.from_pandas(self.construct_c3_dataframe(['Datasets/C3/c3-d-train.json','Datasets/C3/c3-m-train.json']),split='train'),
                            datasets.Dataset.from_pandas(self.construct_c3_dataframe(['Datasets/C3/c3-d-dev.json','Datasets/C3/c3-m-dev.json']),split='validation'),
                            datasets.Dataset.from_pandas(self.construct_c3_dataframe(['Datasets/C3/c3-d-test.json','Datasets/C3/c3-m-test.json']),split='test')]
            self.opt_num = 4
        elif self.name == 'c3_d':
            self.call_encoder = self.local_pre_processing
            self.dataset = [datasets.Dataset.from_pandas(self.construct_c3_dataframe(['Datasets/C3/c3-d-train.json']),split='train'),
                            datasets.Dataset.from_pandas(self.construct_c3_dataframe(['Datasets/C3/c3-d-dev.json']),split='validation'),
                            datasets.Dataset.from_pandas(self.construct_c3_dataframe(['Datasets/C3/c3-d-test.json']),split='test')]
            self.opt_num = 4
        elif self.name == 'c3_m':
            self.call_encoder = self.local_pre_processing
            self.dataset = [datasets.Dataset.from_pandas(self.construct_c3_dataframe(['Datasets/C3/c3-m-train.json']),split='train'),
                            datasets.Dataset.from_pandas(self.construct_c3_dataframe(['Datasets/C3/c3-m-dev.json']),split='validation'),
                            datasets.Dataset.from_pandas(self.construct_c3_dataframe(['Datasets/C3/c3-m-test.json']),split='test')]
            self.opt_num = 4
        elif self.name == 'swequad':
            self.call_encoder = self.local_pre_processing
            self.dataset = [datasets.Dataset.from_pandas(self.construct_swequad_dataframe('Datasets/SweQUAD/training.json'),split='train'),
                            datasets.Dataset.from_pandas(self.construct_swequad_dataframe('Datasets/SweQUAD/dev.json'),split='validation'),
                            datasets.Dataset.from_pandas(self.construct_swequad_dataframe('Datasets/SweQUAD/test.json'),split='test')]
            self.opt_num = 4
        elif self.name == 'muserc':
            self.call_encoder = self.local_pre_processing
            self.dataset = [datasets.Dataset.from_pandas(self.construct_muserc_dataframe('Datasets/MuSeRC/train.jsonl'),split='train'),
                            datasets.Dataset.from_pandas(self.construct_muserc_dataframe('Datasets/MuSeRC/val.jsonl'),split='validation'),
                            datasets.Dataset.from_pandas(self.construct_muserc_dataframe('Datasets/MuSeRC/val.jsonl'),split='test')]
            self.opt_num = 4
            
        
        self.train_dataloader = DataLoader(pack_dataset(self.dataset[0]),batch_size=self.batch_size,shuffle=True,collate_fn=None,drop_last=True)
        self.eval_dataloader = DataLoader(pack_dataset(self.dataset[1]),batch_size=1,shuffle=False,collate_fn=None,drop_last=False)
        self.test_dataloader = DataLoader(pack_dataset(self.dataset[2]),batch_size=1,shuffle=False,collate_fn=None,drop_last=False)
    
    def construct_c3_dataframe(self,pth):
        components = []
        for p in pth:
            raw = pd.read_json(p)
            raw_p = list([' '.join(p)] for p in raw[0])
            count = list(len(raw[1][i]) for i in range(len(raw)))
            passage = pd.Series(list(p for ps in list(raw_p[i]*count[i] for i in range(len(raw))) for p in ps))
            content = pd.json_normalize(pd.Series(list(item for sub in raw[1] for item in sub)))
            c3 = pd.concat([passage,content],axis=1)
            c3.columns = ['article','question','options','answer']
            for i in range(len(c3)):
                if len(c3['options'][i])<4:
                    c3['options'][i]+=['<pad>']*(4-len(c3['options'][i]))
            components.append(c3)
            
        return pd.concat(components,axis=0)
    
    def construct_swequad_dataframe(self,pth):
        raw = pd.json_normalize(pd.read_json(pth)['data'])
        passage = raw['context'].tolist()
        question = raw['question'].tolist()
        options = list(numpy.unique(list(choice['text'] for choice in options)).tolist() for options in raw['choices'])
        answers = list(numpy.unique(list(choice['text'] for choice in options if choice['type']=='Correct answer')).tolist() for options in raw['choices'])
        multi_index = list(i for i in range(len(answers)) if len(answers[i])>1)
        for index in reversed(multi_index):
            for ans in answers[index]:
                passage.insert(index,passage[index])
                question.insert(index,question[index])
                options[index].remove(ans)

            passage.pop(index)
            question.pop(index)

        for index in reversed(multi_index):
            for ans in reversed(answers[index]):
                if len(options[index])>3:
                    options.insert(index+1,random.sample(options[index],3)+[ans])
                else:
                    options.insert(index+1,options[index]+[ans])
                random.shuffle(options[index+1])
            options.pop(index)

        answers = list(answers[i][j] for i in range(len(answers)) for j in range(len(answers[i])))
        swequad = pd.DataFrame(list(zip(passage,question,options,answers)))
        swequad.columns = ['article','question','options','answer']
        for i in range(len(swequad['options'])):
            if len(swequad['options'][i])<4:
                swequad['options'][i]+=['<pad>']*(4-len(swequad['options'][i]))
            elif len(swequad['options'][i])>4:
                swequad['options'][i].remove(swequad['answer'][i])
                swequad['options'][i] = random.sample(swequad['options'][i],3)+[swequad['answer'][i]]
                random.shuffle(swequad['options'][i])

        return swequad
    
    def construct_muserc_dataframe(self,pth):
        raw = pd.json_normalize(pd.read_json(pth,lines=True)['passage'])
        passage = raw['text'].tolist()
        count = list(len(raw['questions'][i]) for i in range(len(raw)))
        passage = list(p for ps in list([passage[i]]*count[i] for i in range(len(passage))) for p in ps)
        pack = pd.json_normalize(pd.Series(list(item for sub in raw['questions'] for item in sub)))
        question = pack['question'].tolist()
        options = list(numpy.unique(list(choice['text'] for choice in options)).tolist() for options in pack['answers'])
        answers = list(numpy.unique(list(choice['text'] for choice in options if choice['label']==1)).tolist() for options in pack['answers'])
        multi_index = list(i for i in range(len(answers)) if len(answers[i])>1)
        for index in reversed(multi_index):
            for ans in answers[index]:
                passage.insert(index,passage[index])
                question.insert(index,question[index])
                options[index].remove(ans)

            passage.pop(index)
            question.pop(index)

        for index in reversed(multi_index):
            for ans in reversed(answers[index]):
                if len(options[index])>3:
                    options.insert(index+1,random.sample(options[index],3)+[ans])
                else:
                    options.insert(index+1,options[index]+[ans])
                random.shuffle(options[index+1])
            options.pop(index)

        answers = list(answers[i][j] for i in range(len(answers)) for j in range(len(answers[i])))
        muserc = pd.DataFrame(list(zip(passage,question,options,answers)))
        muserc.columns = ['article','question','options','answer']
        for i in range(len(muserc['options'])):
            if len(muserc['options'][i])<4:
                muserc['options'][i]+=['<pad>']*(4-len(muserc['options'][i]))
            elif len(muserc['options'][i])>4:
                muserc['options'][i].remove(muserc['answer'][i])
                muserc['options'][i] = random.sample(muserc['options'][i],3)+[muserc['answer'][i]]
                random.shuffle(muserc['options'][i])
        return muserc
    
    def reconstruct_loader(self,split):
        if split=='train':
            self.train_dataloader = DataLoader(pack_dataset(self.dataset[0]),batch_size=self.batch_size,shuffle=True,collate_fn=None,drop_last=True)
        elif split=='eval':
            self.eval_dataloader = DataLoader(pack_dataset(self.dataset[1]),batch_size=1,shuffle=False,collate_fn=None,drop_last=False)
        elif split=='test':
            self.test_dataloader = DataLoader(pack_dataset(self.dataset[2]),batch_size=1,shuffle=False,collate_fn=None,drop_last=False)
        
    def get_next(self,split):
        if split=='train':
            return next(iter(self.train_dataloader))
        elif split=='eval':
            return next(iter(self.eval_dataloader))
        elif split=='test':
            return next(iter(self.test_dataloader))
        
    def validate_loader(self,loader,tokenizer,bi_direction):
        for i, data in enumerate(loader):
            x,_,p_l,q_l,o_l,oi_l,trun_count = self.call_encoder(data,tokenizer,bi_direction=bi_direction)
            assert torch.equal(x['input_ids'][:,0:1],self.make_special(x['input_ids'].size(dim=0),tokenizer(self.new_tokens[0],padding=False, return_tensors="pt",add_special_tokens=False)['input_ids']))
            assert torch.equal(x['token_type_ids'][:,0:1],self.make_special(x['token_type_ids'].size(dim=0),0))
            if bi_direction:
                assert torch.equal(x['input_ids'][:,p_l-1:p_l],self.make_special(x['input_ids'].size(dim=0),tokenizer(self.new_tokens[1],padding=False, return_tensors="pt",add_special_tokens=False)['input_ids']))
                assert torch.equal(x['token_type_ids'][:,p_l-1:p_l],self.make_special(x['token_type_ids'].size(dim=0),0))
            assert torch.equal(x['input_ids'][:,p_l:p_l+1],self.make_special(x['input_ids'].size(dim=0),tokenizer(self.new_tokens[2],padding=False, return_tensors="pt",add_special_tokens=False)['input_ids']))
            assert torch.equal(x['token_type_ids'][:,p_l:p_l+1],self.make_special(x['token_type_ids'].size(dim=0),0))
            if bi_direction:
                assert torch.equal(x['input_ids'][:,p_l+q_l-1:p_l+q_l],self.make_special(x['input_ids'].size(dim=0),tokenizer(self.new_tokens[3],padding=False, return_tensors="pt",add_special_tokens=False)['input_ids']))
                assert torch.equal(x['token_type_ids'][:,p_l+q_l-1:p_l+q_l],self.make_special(x['token_type_ids'].size(dim=0),0))
            for i in range(len(oi_l)-1):
                assert torch.equal(x['input_ids'][:,p_l+q_l+sum(oi_l[0:i+1]):p_l+q_l+sum(oi_l[0:i+1])+1],self.make_special(x['input_ids'].size(dim=0),tokenizer(self.new_tokens[4],padding=False, return_tensors="pt",add_special_tokens=False)['input_ids']))
                assert torch.equal(x['token_type_ids'][:,p_l+q_l+sum(oi_l[0:i+1]):p_l+q_l+sum(oi_l[0:i+1])+1],self.make_special(x['token_type_ids'].size(dim=0),0))
                if bi_direction:
                    assert torch.equal(x['input_ids'][:,p_l+q_l+sum(oi_l[0:i+2])-1:p_l+q_l+sum(oi_l[0:i+2])],self.make_special(x['input_ids'].size(dim=0),tokenizer(self.new_tokens[5],padding=False, return_tensors="pt",add_special_tokens=False)['input_ids']))
                    assert torch.equal(x['token_type_ids'][:,p_l+q_l+sum(oi_l[0:i+2])-1:p_l+q_l+sum(oi_l[0:i+2])],self.make_special(x['token_type_ids'].size(dim=0),0))
            
        return True
    
    def make_special(self,length,special):
        tokens = torch.ones(length,1,dtype=torch.int64)
        tokens[:] = special
        return tokens
    
    def make_type(self,type_id,shape):
        types = torch.ones(shape,dtype=torch.int64)
        types[:] = type_id
        return types
    
    def truncate_passage(self,p_tokens,p_att,q,q_type,q_m,o,o_type,o_m,p_length,q_length,o_length,tokenizer,bi_direction,overlap=0.5):
        
        if p_length+q_length+o_length+2 > 512:
            
            if bi_direction:
                max_length = 512-(2+q_length+o_length)
            else:
                max_length = 512-(1+q_length+o_length)
                
            trun_length = math.ceil(max_length*(1-overlap))
            trun_count = math.floor(p_length/trun_length)

            trun_p_tokens = torch.ones(trun_count,p_tokens.size(dim=0),max_length,dtype=torch.int64)
            trun_p_m = torch.zeros(trun_count,p_tokens.size(dim=0),max_length,dtype=torch.int64)
                
            for i in range(trun_count):
                chunk = p_tokens[:,int(i*trun_length):int(i*trun_length)+max_length]
                chunk_att = p_att[:,int(i*trun_length):int(i*trun_length)+max_length]
                chunk_length = chunk.size(dim=-1)
                trun_p_tokens[i,:,0:chunk_length] = chunk
                trun_p_m[i,:,0:chunk_length] = chunk_att
            
            p_head_lab = tokenizer(self.new_tokens[0],padding=False, return_tensors="pt",add_special_tokens=False)
            p_head = self.make_special(p_tokens.size(dim=0),p_head_lab['input_ids']).unsqueeze(0).repeat(trun_count,1,1)
            p_head_att = self.make_special(p_tokens.size(dim=0),p_head_lab['attention_mask']).unsqueeze(0).repeat(trun_count,1,1)
            if bi_direction:
                p_tail_lab = tokenizer(self.new_tokens[1],padding=False, return_tensors="pt",add_special_tokens=False)
                p_tail = self.make_special(p_tokens.size(dim=0),p_tail_lab['input_ids']).unsqueeze(0).repeat(trun_count,1,1)
                p_tail_att = self.make_special(p_tokens.size(dim=0),p_tail_lab['attention_mask']).unsqueeze(0).repeat(trun_count,1,1)
                p = torch.cat((p_head,trun_p_tokens,p_tail),dim=-1)
                p_m = torch.cat((p_head_att,trun_p_m,p_tail_att),dim=-1)
            else:
                p = torch.cat((p_head,trun_p_tokens),dim=-1)
                p_m = torch.cat((p_head_att,trun_p_m),dim=-1)
                
            p_length = p.size(dim=-1)
            p_type = self.make_type(0,p.size())
            #p_type[:,0] = 0

            input_ids = torch.cat((p,q.unsqueeze(0).repeat(trun_count,1,1),o.unsqueeze(0).repeat(trun_count,1,1)),dim=-1)
            attention_mask = torch.cat((p_m,q_m.unsqueeze(0).repeat(trun_count,1,1),o_m.unsqueeze(0).repeat(trun_count,1,1)),dim=-1)
            type_ids = torch.cat((p_type,q_type.unsqueeze(0).repeat(trun_count,1,1),o_type.unsqueeze(0).repeat(trun_count,1,1)),dim=-1)
            

        else:
            trun_count = 1
            p_head_lab = tokenizer(self.new_tokens[0],padding=False, return_tensors="pt",add_special_tokens=False)
            p_head = self.make_special(p_tokens.size(dim=0),p_head_lab['input_ids'])
            p_head_att = self.make_special(p_tokens.size(dim=0),p_head_lab['attention_mask'])
            if bi_direction:
                p_tail_lab = tokenizer(self.new_tokens[1],padding=False, return_tensors="pt",add_special_tokens=False)
                p_tail = self.make_special(p_tokens.size(dim=0),p_tail_lab['input_ids'])
                p_tail_att = self.make_special(p_tokens.size(dim=0),p_tail_lab['attention_mask'])
                p = torch.cat((p_head,p_tokens,p_tail),dim=-1)
                p_m = torch.cat((p_head_att,p_att,p_tail_att),dim=-1)
            else:
                p = torch.cat((p_head,p_tokens),dim=-1)
                p_m = torch.cat((p_head_att,p_att),dim=-1)
                
            p_length = p.size(dim=-1)
            p_type = self.make_type(0,p.size())
            #p_type[:,0] = 0

            input_ids = torch.cat((p,q,o),dim=-1)
            attention_mask = torch.cat((p_m,q_m,o_m),dim=-1)
            type_ids = torch.cat((p_type,q_type,o_type),dim=-1)
            
        #x = {'input_ids':input_ids.view(-1,input_ids.size(dim=-1)),'token_type_ids':type_ids.view(-1,type_ids.size(dim=-1)),'attention_mask':attention_mask.view(-1,attention_mask.size(dim=-1))}
        x = {'input_ids':input_ids.view(-1,input_ids.size(dim=-1)),'attention_mask':attention_mask.view(-1,attention_mask.size(dim=-1))}
        #x = {'input_ids':input_ids.view(-1,input_ids.size(dim=-1))}
            
        return tokenization_utils_base.BatchEncoding(x),p_length,trun_count
    
    def make_question(self,q_tokens,q_att,tokenizer,bi_direction):
        q_head_lab = tokenizer(self.new_tokens[2],padding=False, return_tensors="pt",add_special_tokens=False)
        q_head = self.make_special(q_tokens.size(dim=0),q_head_lab['input_ids'])
        q_head_att = self.make_special(q_tokens.size(dim=0),q_head_lab['attention_mask'])
        if bi_direction:
            q_tail_lab = tokenizer(self.new_tokens[3],padding=False, return_tensors="pt",add_special_tokens=False)
            q_tail = self.make_special(q_tokens.size(dim=0),q_tail_lab['input_ids'])
            q_tail_att = self.make_special(q_tokens.size(dim=0),q_tail_lab['attention_mask'])
            q = torch.cat((q_head,q_tokens,q_tail),dim=-1)
            q_m = torch.cat((q_head_att,q_att,q_tail_att),dim=-1)
        else:
            q = torch.cat((q_head,q_tokens),dim=-1)
            q_m = torch.cat((q_head_att,q_att),dim=-1)
        
        q_length = q.size(dim=-1)
        q_type = self.make_type(1,q.size())
        #q_type[:,0] = 0
        
        return q,q_m,q_type,q_length
    
    def make_options(self,oi_tokens,oi_att,tokenizer,bi_direction):
        o_head_lab = tokenizer(self.new_tokens[4],padding=False, return_tensors="pt",add_special_tokens=False)
        o_head = self.make_special(oi_tokens[0].size(dim=0),o_head_lab['input_ids'])
        o_head_att = self.make_special(oi_tokens[0].size(dim=0),o_head_lab['attention_mask'])
        for i in range(len(oi_tokens)):
            if bi_direction:
                o_tail_lab = tokenizer(self.new_tokens[5],padding=False, return_tensors="pt",add_special_tokens=False)
                o_tail = self.make_special(oi_tokens[i].size(dim=0),o_tail_lab['input_ids'])
                o_tail_att = self.make_special(oi_tokens[i].size(dim=0),o_tail_lab['attention_mask'])
                oi_tokens[i] = torch.cat((o_head,oi_tokens[i],o_tail),dim=-1)
                oi_att[i] = torch.cat((o_head_att,oi_att[i],o_tail_att),dim=-1)
            else:
                oi_tokens[i] = torch.cat((o_head,oi_tokens[i]),dim=-1)
                oi_att[i] = torch.cat((o_head_att,oi_att[i]),dim=-1)

        o = torch.cat(oi_tokens,dim=-1)
        o_m = torch.cat(oi_att,dim=-1)
        oi_lengths = [0] + list(o_i.size(dim=-1) for o_i in oi_tokens)
        o_length = sum(oi_lengths)
        o_type = self.make_type(2,o.size())
        #for i in range(len(oi_lengths)-1):
        #    o_type[:,sum(oi_lengths[0:i+1])] = 0
        
        return o,o_m,o_type,o_length,oi_lengths
        

    def race_pre_processing(self,data,tokenizer,bi_direction=False):

        p_lab = tokenizer(data['article'],padding=True, return_tensors="pt",add_special_tokens=False)
        p_tokens = p_lab['input_ids']
        p_att = p_lab['attention_mask']
        p_length = p_tokens.size(dim=-1)

        q_lab = tokenizer(data['question'],padding=True, return_tensors="pt",add_special_tokens=False)
        q_tokens = q_lab['input_ids']
        q_att = q_lab['attention_mask']
        
        oi_tokens = list(tokenizer(opt,padding=True, return_tensors="pt",add_special_tokens=False)['input_ids'] for opt in list(map(list,data['options'])))
        oi_att = list(tokenizer(opt,padding=True, return_tensors="pt",add_special_tokens=False)['attention_mask'] for opt in list(map(list,data['options'])))
        
        q,q_m,q_type,q_length = self.make_question(q_tokens,q_att,tokenizer,bi_direction)
        
        o,o_m,o_type,o_length,oi_lengths = self.make_options(oi_tokens,oi_att,tokenizer,bi_direction)

        x, p_length, trun_count = self.truncate_passage(p_tokens,p_att,q,q_type,q_m,o,o_type,o_m,p_length,q_length,o_length,tokenizer,bi_direction)
            
        shift_dict = {"A":0,"B":1,"C":2,"D":3}
        y = torch.Tensor(list(shift_dict[item] for item in data['answer']))

        return x,y,p_length,q_length,o_length,oi_lengths,trun_count
    
    def cosmos_pre_processing(self,data,tokenizer,bi_direction=False):
        
        p_lab = tokenizer(data['context'],padding=True, return_tensors="pt",add_special_tokens=False)
        p_tokens = p_lab['input_ids']
        p_att = p_lab['attention_mask']
        p_length = p_tokens.size(dim=-1)
        
        q_lab = tokenizer(data['question'],padding=True, return_tensors="pt",add_special_tokens=False)
        q_tokens = q_lab['input_ids']
        q_att = q_lab['attention_mask']
        
        ans_0 = tokenizer(data['answer0'],padding=True, return_tensors="pt",add_special_tokens=False)
        ans_1 = tokenizer(data['answer1'],padding=True, return_tensors="pt",add_special_tokens=False)
        ans_2 = tokenizer(data['answer2'],padding=True, return_tensors="pt",add_special_tokens=False)
        ans_3 = tokenizer(data['answer3'],padding=True, return_tensors="pt",add_special_tokens=False)
        oi_tokens = [ans_0['input_ids'],ans_1['input_ids'],ans_2['input_ids'],ans_3['input_ids']]
        oi_att = [ans_0['attention_mask'],ans_1['attention_mask'],ans_2['attention_mask'],ans_3['attention_mask']]
        
        q,q_m,q_type,q_length = self.make_question(q_tokens,q_att,tokenizer,bi_direction)
        
        o,o_m,o_type,o_length,oi_lengths = self.make_options(oi_tokens,oi_att,tokenizer,bi_direction)
        
        x, p_length, trun_count = self.truncate_passage(p_tokens,p_att,q,q_type,q_m,o,o_type,o_m,p_length,q_length,o_length,tokenizer,bi_direction)
        
        y = data['label']
        
        return x,y,p_length,q_length,o_length,oi_lengths,trun_count
    
    def dream_pre_processing(self,data,tokenizer,bi_direction=False):
        
        p_lab = tokenizer(data['dialogue'],padding=True, return_tensors="pt",add_special_tokens=False)
        p_tokens = p_lab['input_ids']
        p_att = p_lab['attention_mask']
        p_length = p_tokens.size(dim=-1)
        
        q_lab = tokenizer(data['question'],padding=True, return_tensors="pt",add_special_tokens=False)
        q_tokens = q_lab['input_ids']
        q_att = q_lab['attention_mask']
        
        oi_tokens = list(tokenizer(opt,padding=True, return_tensors="pt",add_special_tokens=False)['input_ids'] for opt in list(map(list,data['choice'])))
        oi_att = list(tokenizer(opt,padding=True, return_tensors="pt",add_special_tokens=False)['attention_mask'] for opt in list(map(list,data['choice'])))
        
        q,q_m,q_type,q_length = self.make_question(q_tokens,q_att,tokenizer,bi_direction)
        
        o,o_m,o_type,o_length,oi_lengths = self.make_options(oi_tokens,oi_att,tokenizer,bi_direction)
        
        x, p_length, trun_count = self.truncate_passage(p_tokens,p_att,q,q_type,q_m,o,o_type,o_m,p_length,q_length,o_length,tokenizer,bi_direction)
            
        y = torch.Tensor(list(list([data['choice'][i][j] for i in range(len(data['choice']))] for j in range(len(data['choice'][0])))[i].index(data['answer'][i]) for i in range(len(data['choice'][0]))))
        
        return x,y,p_length,q_length,o_length,oi_lengths,trun_count
    
    def mctest_pre_processing(self,data,tokenizer,bi_direction=False):
        
        p_lab = tokenizer(data['story'],padding=True, return_tensors="pt",add_special_tokens=False)
        p_tokens = p_lab['input_ids']
        p_att = p_lab['attention_mask']
        p_length = p_tokens.size(dim=-1)
        
        q_lab = tokenizer(data['question'],padding=True, return_tensors="pt",add_special_tokens=False)
        q_tokens = q_lab['input_ids']
        q_att = q_lab['attention_mask']
        
        oi_tokens = list(tokenizer(opt,padding=True, return_tensors="pt",add_special_tokens=False)['input_ids'] for opt in list(map(list,data['answer_options'])))
        oi_att = list(tokenizer(opt,padding=True, return_tensors="pt",add_special_tokens=False)['attention_mask'] for opt in list(map(list,data['answer_options'])))
        
        q,q_m,q_type,q_length = self.make_question(q_tokens,q_att,tokenizer,bi_direction)
        
        o,o_m,o_type,o_length,oi_lengths = self.make_options(oi_tokens,oi_att,tokenizer,bi_direction)
        
        x, p_length, trun_count = self.truncate_passage(p_tokens,p_att,q,q_type,q_m,o,o_type,o_m,p_length,q_length,o_length,tokenizer,bi_direction)
        
        shift_dict = {"A":0,"B":1,"C":2,"D":3}
        y = torch.Tensor(list(shift_dict[item] for item in data['answer']))
        
        return x,y,p_length,q_length,o_length,oi_lengths,trun_count
    
    def qa4mre_pre_processing(self,data,tokenizer,bi_direction=False):
        
        p_lab = tokenizer(data['document_str'],padding=True, return_tensors="pt",add_special_tokens=False)
        p_tokens = p_lab['input_ids']
        p_att = p_lab['attention_mask']
        p_length = p_tokens.size(dim=-1)
        
        q_lab = tokenizer(data['question_str'],padding=True, return_tensors="pt",add_special_tokens=False)
        q_tokens = q_lab['input_ids']
        q_att = q_lab['attention_mask']
        
        oi_tokens = list(tokenizer(opt,padding=True, return_tensors="pt",add_special_tokens=False)['input_ids'] for opt in list(map(list,data['answer_options']['answer_str'])))
        oi_att = list(tokenizer(opt,padding=True, return_tensors="pt",add_special_tokens=False)['attention_mask'] for opt in list(map(list,data['answer_options']['answer_str'])))
        
        q,q_m,q_type,q_length = self.make_question(q_tokens,q_att,tokenizer,bi_direction)
        
        o,o_m,o_type,o_length,oi_lengths = self.make_options(oi_tokens,oi_att,tokenizer,bi_direction)
        
        x, p_length, trun_count = self.truncate_passage(p_tokens,p_att,q,q_type,q_m,o,o_type,o_m,p_length,q_length,o_length,tokenizer,bi_direction)
        
        y = torch.Tensor(list(int(ans) for ans in data['correct_answer_id']))
        
        return x,y,p_length,q_length,o_length,oi_lengths,trun_count
    
    def local_pre_processing(self,data,tokenizer,bi_direction=False):
        
        p_lab = tokenizer(data['article'],padding=True, return_tensors="pt",add_special_tokens=False)
        p_tokens = p_lab['input_ids']
        p_att = p_lab['attention_mask']
        p_length = p_tokens.size(dim=-1)
        
        q_lab = tokenizer(data['question'],padding=True, return_tensors="pt",add_special_tokens=False)
        q_tokens = q_lab['input_ids']
        q_att = q_lab['attention_mask']
        
        oi_tokens = list(tokenizer(opt,padding=True, return_tensors="pt",add_special_tokens=False)['input_ids'] for opt in list(map(list,data['options'])))
        oi_att = list(tokenizer(opt,padding=True, return_tensors="pt",add_special_tokens=False)['attention_mask'] for opt in list(map(list,data['options'])))
        
        q,q_m,q_type,q_length = self.make_question(q_tokens,q_att,tokenizer,bi_direction)
        
        o,o_m,o_type,o_length,oi_lengths = self.make_options(oi_tokens,oi_att,tokenizer,bi_direction)
        
        x, p_length, trun_count = self.truncate_passage(p_tokens,p_att,q,q_type,q_m,o,o_type,o_m,p_length,q_length,o_length,tokenizer,bi_direction)
        
        y = torch.Tensor(list(list([data['options'][i][j] for i in range(len(data['options']))] for j in range(len(data['options'][0])))[i].index(data['answer'][i]) for i in range(len(data['options'][0]))))
        
        return x,y,p_length,q_length,o_length,oi_lengths,trun_count