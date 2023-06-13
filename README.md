# Full-Context-Componets
Code for Project: Attend All Options At Once: Full Context Input For Multi-Choice Reading Comprehension

To run the full model with full context components input, execute 'Execute.py' with the following args:   

'--encoder': string, 'model type(name in Transformers library/HuggingFace)'   
'--encoder_lr': float, 'learning rate of encoder network'   
'--head_lr': float, 'learning rate of the head network including fuser & discriminator network'   
'--sample_count': integer, 'counts of sampling in one batch'   
'--entry_count': integer, 'number of samples in each sampling'   
'--epoch': integer, 'number of training epochs'   
'--datasets': string/list of string, 'involved datasets'   
'--warmups': float, 'proportion of warming up iters'   
'--evaluation_only': bool, 'evaluation only'   
'--weights': string, 'path of trained model weights'   
   
*code for baseline & no fuser model is still under reconstruction.   
*new datasets are not supported yet, datasets involved in our experiments: 'race', 'cosmos_qa', 'dream', 'c3', 'swequad' and 'muserc'.
