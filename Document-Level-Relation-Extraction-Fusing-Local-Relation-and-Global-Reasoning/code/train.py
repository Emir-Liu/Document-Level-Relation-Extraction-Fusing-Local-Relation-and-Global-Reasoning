from Config import Config
from preope import read_file
from model import Model

import time
import math

import torch
from torch.utils.data import DataLoader

from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

from evaluation import *

import random
import numpy as np
config = Config()

if config.use_wandb == True:
    import wandb

if torch.cuda.is_available() and config.use_gpu == True :
    from apex import amp
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')
config.device = device

# 读取关系列表
rel2id = json.load(open(config.rel2id_path, 'r'))
id2rel = {value: key for key, value in rel2id.items()}

def collate_fn(batch):
    max_len = max([len(item["input_ids"]) for item in batch])

    input_ids = [item['input_ids'] + [0] * (max_len - len(item['input_ids'])) for item in batch]
    input_masks = [[1] * len(item["input_ids"]) + [0] * (max_len - len(item["input_ids"])) for item in batch]
    labels = [torch.tensor(item['labels']).to(config.device).view(-1,config.num_class) for item in batch]
    entity_pos = [item['entity_pos'] for item in batch]

    input_masks = torch.tensor(input_masks,dtype=torch.long).to(config.device)
    input_ids = torch.tensor(input_ids, dtype=torch.long).to(config.device)
    #labels = torch.tensor(labels, dtype=torch.long).to(config.device)
    titles = [item['title'] for item in batch]
    return {'batch_input_ids':input_ids,
            'batch_input_masks':input_masks,
            'batch_labels':labels,
            'batch_entity_pos':entity_pos,
            'batch_titles':titles}

best_score = -1

if __name__ == '__main__':

    seed = 66
    random.seed(66)
    np.random.seed(66)
    torch.manual_seed(66)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(66)

    if config.use_wandb == True:
        wandb.init(project="test")

    t = time.strftime("%a %b %d %H:%M:%S %Y", time.localtime())
    print('{} start and init: '.format(t))

    bert_config = AutoConfig.from_pretrained(config.bert_path)
    bert_tokenizer = AutoTokenizer.from_pretrained(config.bert_path)
    config.cls_token_id = bert_tokenizer.cls_token_id
    config.sep_token_id = bert_tokenizer.sep_token_id
    config.bert_config = bert_config

    # 读取训练数据
    train_dataset = read_file(config.train_path,bert_tokenizer,rel2id)
    dev_dataset = read_file(config.dev_path, bert_tokenizer, rel2id)
    #print('train_dataset: {}'.format(train_dataset))

    # 将训练数据切片
    train_dataloader = DataLoader(train_dataset,batch_size=config.train_batch_size,collate_fn=collate_fn,drop_last=True)
    dev_dataloader = DataLoader(dev_dataset,batch_size=config.dev_batch_size,collate_fn=collate_fn,drop_last=True)
    '''
    for item in train_dataloader:
        print('item:{}'.format(item))
    '''
    # 初始化模型
    model = Model(config)
    model.to(config.device)
    model.zero_grad()

    # 修改模型的学习速率
    new_layer = ['extractor', 'build_relation','inference','concept','result_linear']
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in new_layer)], },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in new_layer)], "lr": config.lr},
    ]

    total_steps = int(len(train_dataloader) * config.num_epoch )
    warmup_steps = int(total_steps * config.warmup_ratio)
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.bert_lr, eps=config.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,num_training_steps=total_steps)

    if torch.cuda.is_available() and config.use_gpu == True :
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1", verbosity=0)

    print('named parameters {}'.format(model.parameters))
    print('total parameters:', sum([np.prod(list(p.size())) for p in model.parameters() if p.requires_grad]))

    num_steps = 0
    t = time.strftime("%a %b %d %H:%M:%S %Y", time.localtime())
    print('{} start training: '.format(t))

    model.train()
    best_score = -1
    for epoch in range(config.num_epoch):
        for item in train_dataloader:
            num_steps += 1
            #print('item:{}'.format(item))
            #print('batch :{}'.format(item))
            #print('batch batch_input_ids :{}'.format(batch['batch_input_ids']))
            loss,output = model(item)
            #print('output:{}'.format(output))
            loss = loss/config.train_batch_size

            #print('loss:{}'.format(loss.shape))
            if torch.cuda.is_available() and config.use_gpu == True:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                if config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), config.max_grad_norm)
            else:
                loss.backward()
            optimizer.step()
            scheduler.step()
            model.zero_grad()

            t = time.strftime("%a %b %d %H:%M:%S %Y", time.localtime())
            #print('{} loss: {}'.format(t,loss))
            if config.use_wandb == True:
                wandb.log({"loss": loss}, step=num_steps)


        # 下面是对训练的模型进行评价
        preds = []
        model.eval()
        title = []
        for item in dev_dataloader:
            #print('dev item : {}'.format(item))
            title.append(item['batch_titles'])
            with torch.no_grad():
                pred = model(item)
                #print('pred : {}'.format(pred))
                #pred = pred.cpu().numpy()
                #print('pred :{}'.format(pred))
                #pred[np.isnan(pred)] = 0
                preds.append(pred)
        #preds = np.concatenate(preds, axis=0).astype(np.float32)
        #print('preds:{}'.format(preds))
        # 下面是直接显示预测的内容
        ans = []
        #print('preds0 : {}'.format(preds[0]))
        for idx,item in enumerate(preds):
            #print('item : {}'.format(item))
            for idx_data, data in enumerate(item):
                #print('idx : {}'.format(idx))
                #print('preds : {}'.format(preds[idx]))
                #print('data:{}'.format(data))
                num_entity = int(math.sqrt(data.shape[0]))
                #print('num_entity:{}'.format(num_entity))
                #print('item : {} \n {}'.format(preds[idx],preds[idx].shape))
                tmp_pred = data.view((num_entity,num_entity,-1))
                #print('tmp_pred:{}'.format(tmp_pred.shape))
                for h in range(num_entity):
                    for t in range(num_entity):
                        rel = tmp_pred[h][t]
                        for tmp in range(len(rel)):
                            if(rel[tmp]==1 and tmp != 0 and h!=t):
                                ans.append(
                                {
                                    'title': title[idx][idx_data],
                                    'h_idx': h,
                                    't_idx': t,
                                    'r': id2rel[tmp],
                                })

        #print('ans : {}'.format(ans))
        #best_f1 = 0
        #best_f1_ign = 0
        if len(ans) > 0:
            best_f1, best_f1_ign,re_p,re_r,re_p_i = official_evaluate(ans, config.data_path,config.train_file,config.dev_file)
        else:
            best_f1 = 0
            best_f1_ign = 0
            re_p = 0
            re_r = 0
            re_p_i = 0
        tag = "dev"
        output = {
            tag + "_F1": best_f1 * 100,
            tag + "_F1_ign": best_f1_ign * 100,
            tag + "re_p" : re_p,
            tag + "re_c" : re_r,
            tag + "re_p_i" : re_p_i
        }
        if config.use_wandb == True:
            wandb.log(output, step=num_steps)
        t = time.strftime("%a %b %d %H:%M:%S %Y", time.localtime())
        print('{} out_put:{}'.format(t,output))
        if best_f1 > best_score:
            with open(config.result_path,'w') as f:
                json.dump(ans,f)
            torch.save(model.state_dict(), config.ckpt_path)
            #print('save result and model')
        model.train()
    print('finish')

