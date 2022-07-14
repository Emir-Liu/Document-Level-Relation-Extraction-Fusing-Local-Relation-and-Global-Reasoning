import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel
from opt_einsum import contract

from long_seq import process_long_input


class Model(nn.Module):
    def __init__(self,config):
        super().__init__()

        self.config = config
        self.bert_model = AutoModel.from_pretrained(config.bert_path)

        # 实体表示和局部表示结合
        self.head_extractor = nn.Linear(2*config.hidden_bert_size,config.hidden_ent_size)
        self.tail_extractor = nn.Linear(2*config.hidden_bert_size,config.hidden_ent_size)

        # 构建实体之间的关系
        if self.config.use_group_bilinear == False:
            self.build_relation = nn.Bilinear(config.hidden_ent_size,config.hidden_ent_size,config.hidden_rel_size)
        elif self.config.use_group_bilinear == True:
            self.build_relation = nn.Linear(config.hidden_ent_size * config.block_size, config.hidden_rel_size)

        # 推理过程中的内容
        if config.use_inf == True:
            self.inference = nn.Linear(config.hidden_rel_size*config.hidden_rel_size,config.hidden_inf_size)

        # 概念过程中的内容
        if config.use_con == True:
            self.concept = nn.Linear((config.hidden_inf_size+config.hidden_rel_size)*(config.hidden_inf_size+config.hidden_rel_size),config.hidden_con_size)

        # 从关系到最后的分类
        if self.config.use_rel == False:
            self.config.hidden_rel_size = 0
        if self.config.use_inf == False:
            self.config.hidden_inf_size = 0
        if self.config.use_con == False:
            self.config.hidden_con_size = 0
        self.dim_out = self.config.hidden_rel_size + self.config.hidden_inf_size + self.config.hidden_con_size
        if self.dim_out != self.config.num_class:
            self.result_linear = nn.Linear(self.dim_out,config.num_class)

        # 自适应
        self.loss_fnt = ATLoss()

    def forward(self,batch):

        # 这个是用来输出结果的
        total_output = []
        offset = 1
        #print('batch data:{}'.format(batch))

        # 标签
        labels = batch['batch_labels']
        # 增加attention机制
        input_ids = batch['batch_input_ids']
        input_masks = batch['batch_input_masks']

        bert_config = self.config.bert_config
        #print('config : {}'.format(bert_config))
        start_tokens = [self.config.cls_token_id]
        end_tokens = [self.config.sep_token_id]
        #print('input_ids:{}'.format(input_ids))
        #print('input_masks:{}'.format(input_masks))
        bert_sequence, bert_attention = process_long_input(self.bert_model, input_ids, input_masks, start_tokens, end_tokens)
        #print('bert_sequence:{}'.format(bert_sequence))
        #print('bert_attention:{}'.format(bert_attention))
        entity_pos = batch['batch_entity_pos']
        hs,ts,rs = [],[],[]
        batch_output = []
        #print('batch:{}'.format(batch))
        for i in range(len(entity_pos)):
            entity_embs, entity_atts = [], []
            #print('entity pos[i]:{}'.format(entity_pos[i]))
            num_entity = len(entity_pos[i])
            hs,ts,rs = [],[],[]
            #print('num_entity:{}'.format(num_entity))
            for idx_entity,entity in enumerate(entity_pos[i]):

                #print('entity:{}'.format(entity))
                if len(entity) > 1:
                    tmp_emb ,tmp_att = [],[]
                    for start, end in entity:
                        tmp_emb.append(bert_sequence[i,start + offset])
                        tmp_att.append(bert_attention[i, :, start + offset])

                    entity_emb = torch.logsumexp(torch.stack(tmp_emb, dim=0), dim=0)
                    entity_att = torch.stack(tmp_att, dim=0).mean(0)
                else:
                    start,end = entity[0]
                    entity_emb = bert_sequence[i,start + offset]
                    entity_att = bert_attention[i, :, start + offset]
                entity_embs.append(entity_emb)
                entity_atts.append(entity_att)
            entity_embs = torch.stack(entity_embs, dim=0)  # [n_e, d]
            entity_atts = torch.stack(entity_atts, dim=0)  # [n_e, h, seq_len]
            #print('entity_atts:{}'.format(entity_atts.shape))
            #print('entity embs {}'.format(entity_embs.size()))
            #print('entity atts {}'.format(entity_atts.size()))

            ht_i = []
            for h in range(num_entity):
                for t in range(num_entity):
                    ht_i.append((h,t))
            ht_i = torch.LongTensor(ht_i).to(bert_sequence.device)
            hs = torch.index_select(entity_embs, 0, ht_i[:, 0])
            ts = torch.index_select(entity_embs, 0, ht_i[:, 1])

            h_att = torch.index_select(entity_atts, 0, ht_i[:, 0])
            t_att = torch.index_select(entity_atts, 0, ht_i[:, 1])
            #print('h_att:{}'.format(h_att.shape))
            #print('t_att:{}'.format(t_att.shape))
            ht_att = (h_att * t_att).mean(1)
            #print('ht_att :{}'.format(ht_att.shape))

            ht_att = ht_att / (ht_att.sum(1, keepdim=True) + 1e-5)
            rs = contract("ld,rl->rd", bert_sequence[i], ht_att)
            #print('hs ts rs {} {} {}'.format(hs.size(), ts.size(), rs.size()))

            hs = torch.tanh(self.head_extractor(torch.cat([hs,rs],dim=1)))
            ts = torch.tanh(self.tail_extractor(torch.cat([ts,rs],dim=1)))
            #print('hs ts {} {}'.format(hs.size(),ts.size()))

            hs = hs.view(num_entity,num_entity,-1)
            ts = ts.view(num_entity, num_entity, -1)
            rs = rs.view(num_entity, num_entity, -1)
            #print('hs ts rs {} {} {}'.format(hs.size(),ts.size(),rs.size()))

            if self.config.use_group_bilinear == False:
                rel_vec = self.build_relation(hs,ts)
            elif self.config.use_group_bilinear == True:
                b1 = hs.view(-1, self.config.hidden_ent_size // self.config.block_size, self.config.block_size)
                b2 = ts.view(-1, self.config.hidden_ent_size // self.config.block_size, self.config.block_size)
                bl = (b1.unsqueeze(3) * b2.unsqueeze(2)).view(-1, self.config.hidden_ent_size * self.config.block_size)
                rel_vec = torch.tanh(self.build_relation(bl))
                print('rel vec :{}'.format(rel_vec.shape))
                #rel_vec = rel_vec.view(num_entity,num_entity,-1)
            #print('rel vec :{}'.format(rel_vec.shape))

            if self.config.use_inf == True:
                # 这里是每个元素进行相乘操作，可以直接用相乘然后线性。
                #print('rel_vec {}'.format(rel_vec.size()))
                # 上面是实体对之间的关系
                '''
                #下面的可以代替上面的Bilinear内容
                next_rel_vec = torch.matmul(rel_vec.unsqueeze(-1),rel_vec.unsqueeze(-2))
                print('next rel vec {}'.format(next_rel_vec.shape))
                next_rel_vec = torch.tanh(self.logic_bilinear(next_rel_vec))
                print('next rel vec {}'.format(next_rel_vec.shape))
                '''

                # 下面是推理的过程有两个实现的方法
                # 下面是推理过程
                # 下面的部分需要大量的内存空间未压缩为17G，关系推理的矩阵实现方法
                # num_entity rel_vec

                tmp_rel_vec = rel_vec.view(num_entity,num_entity,-1)
                #next_rel_vec = torch.zeros((tmp_rel_vec.size()))

                ht = tmp_rel_vec.unsqueeze(1)
                tt = tmp_rel_vec.unsqueeze(0)
                #print('ht tt {} {}'.format(ht.shape,tt.shape))

                htt = ht.unsqueeze(-1)
                ttt = tt.unsqueeze(-2)
                #print('htt ttt {} {}'.format(htt.shape,ttt.shape))

                inf_vec = torch.matmul(htt,ttt)
                inf_vec = torch.sum(inf_vec,2)
                inf_vec = inf_vec.view(num_entity,num_entity,-1)
                inf_vec = torch.tanh(self.inference(inf_vec))
                inf_vec = inf_vec.view(-1,self.config.hidden_inf_size)
                print('inf vec:{}'.format(inf_vec.shape))

                if self.config.use_con == True:
                    inf_rel_vec = torch.cat([inf_vec,rel_vec],dim = -1)
                    #print('inf_rel_vec:{}'.format(inf_rel_vec.shape))
                    tmp_inf_rel_vec = inf_rel_vec.view(num_entity,num_entity,-1)
                    #print('tmp_inf_rel_vec:{}'.format(tmp_inf_rel_vec.shape))
                    next_rel_vec = torch.zeros((tmp_rel_vec.size()))

                    con_ht = tmp_inf_rel_vec.unsqueeze(1)
                    con_tt = tmp_inf_rel_vec.unsqueeze(0)
                    #print('ht tt {} {}'.format(con_ht.shape,con_tt.shape))

                    con_htt = con_ht.unsqueeze(-1)
                    con_ttt = con_tt.unsqueeze(-2)
                    #print('htt ttt {} {}'.format(con_htt.shape,con_ttt.shape))

                    con_vec = torch.matmul(con_htt,con_ttt)
                    con_vec = torch.sum(con_vec,2)
                    con_vec = con_vec.view(num_entity,num_entity,-1)
                    #print('con_vec -1:{}'.format(con_vec.shape))
                    con_vec = torch.tanh(self.concept(con_vec))
                    con_vec = con_vec.view(-1,self.config.hidden_con_size)
                    print('con vec:{}'.format(con_vec.shape))

            if self.config.use_rel == True and self.config.use_inf == True and self.config.use_con == True:
                next_vec = torch.cat([con_vec,inf_vec,rel_vec],dim = -1)
            elif self.config.use_rel == True and self.config.use_inf == True:
                next_vec = torch.cat([inf_vec,rel_vec],dim = -1)
            elif self.config.use_inf == True:
                next_vec = inf_vec
            elif self.config.use_rel == True:
                next_vec = rel_vec
            else:
                print('error')
            #print('next_rel_vec : {}'.format(next_rel_vec.size()))

            #
            print('next vec:{}'.format(next_vec.shape))

            if next_vec.shape[1] == self.config.num_class:
                out = next_vec
            else:
                out = self.result_linear(next_vec)
            print('out vec:{}'.format(out.shape))

            output = self.loss_fnt.get_label(logits=out,num_labels=self.config.num_class)
            #print('output:{}'.format(output))
            #print('output grad:{}'.format(output.requires_grad))

            # 下面的batch_output是用来计算误差的
            batch_output.append(out)
            total_output.append(output)
        #print('batch output:{}'.format(batch_output))
        if self.training:
            #print('labels:{}'.format(labels))
            new_labels = torch.cat(labels,0)
            new_out = torch.cat(batch_output,0)
            #print('new_labels:{} \n {}'.format(new_labels,new_labels.shape))
            #print('new_out:{} \n {}'.format(new_out,new_out.shape))
            loss = self.loss_fnt(new_out.float(), new_labels.float())

            return loss,total_output

        #print("output model:{}".format(output))
        return total_output


class ATLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, labels):
        # TH label
        #print('logits : {}'.format(logits))
        #print('labels : {}'.format(labels))
        th_label = torch.zeros_like(labels, dtype=torch.float).to(labels)
        #print('th lable {}'.format(th_label.shape))
        th_label[:, 0] = 1.0
        labels[:, 0] = 0.0

        p_mask = labels + th_label
        n_mask = 1 - labels

        # Rank positive classes to TH
        #print('logits :{}'.format(logits.shape))
        #print('p_mask: {}'.format(p_mask.shape))
        #print('th_lable : {}'.format(th_label))
        logit1 = logits - (1 - p_mask) * 1e30
        loss1 = -(F.log_softmax(logit1, dim=-1) * labels).sum(1)

        # Rank TH to negative classes
        logit2 = logits - (1 - n_mask) * 1e30
        loss2 = -(F.log_softmax(logit2, dim=-1) * th_label).sum(1)

        # Sum two parts
        loss = loss1 + loss2
        loss = loss.mean()
        return loss

    def get_label(self, logits, num_labels=-1):
        th_logit = logits[:, 0].unsqueeze(1)
        output = torch.zeros_like(logits).to(logits)
        mask = (logits > th_logit)
        if num_labels > 0:
            #print('logits:{}'.format(logits.shape))
            top_v, _ = torch.topk(logits, num_labels, dim=1)
            top_v = top_v[:, -1]
            mask = (logits >= top_v.unsqueeze(1)) & mask
        output[mask] = 1.0
        output[:, 0] = (output.sum(1) == 0.).to(logits)
        return output
