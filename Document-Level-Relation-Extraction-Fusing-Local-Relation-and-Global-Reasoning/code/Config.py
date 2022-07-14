import os.path

class Config:
    def __init__(self):
        # 输入文件的路径配
        data_type = 'docred'# 'minidata' 'jf' 'docred'
        if data_type  == 'minidata':
            # 下面是Docred部分小数据
            self.data_path = '../dataset/minidata'
            self.rel2id_file = 'rel2id.json'
            self.train_file = 'train_annotated.json' # 训练神经网络
            self.dev_file = 'train_annotated.json'   # 根据F1挑选模型
            self.test_file = 'train_annotated.json'  # 输出预测的结果和F1
            self.bert_path = '../plm/bert-base-cased'
            self.num_class = 97 # 最终关系的类别数
            # 训练参数设置
            self.train_batch_size = 1
            self.dev_batch_size = 1
            self.num_epoch = 2
        elif data_type == 'docred':
            self.data_path = '../dataset/docred'
            self.rel2id_file = 'rel2id.json'
            self.train_file = 'train_annotated.json' # 训练神经网络
            self.dev_file = 'dev.json'   # 根据F1挑选模型
            self.test_file = 'test.json'  # 输出预测的结果和F1
            self.bert_path = '../plm/bert-base-cased'
            self.num_class = 97 # 最终关系的类别数
            # 训练参数设置
            self.train_batch_size = 3
            self.dev_batch_size = 6
            self.num_epoch = 200

        # 输出文件路径配置
        #self.output_path = '../768+768+97+97+rel+inf+group_bilinear_64_200e'
        self.output_path = '../test_out'
        self.result_file = 'result.json'
        self.ckpt_file = 'model.ckpt'

        # 文件的完整路径表示
        self.rel2id_path = os.path.join(self.data_path,self.rel2id_file)
        self.train_path = os.path.join(self.data_path,self.train_file)
        self.dev_path = os.path.join(self.data_path,self.dev_file)

        self.result_path = os.path.join(self.output_path,self.result_file)
        self.ckpt_path = os.path.join(self.output_path,self.ckpt_file)


        self.bert_lr = 5e-5 # bert模型部分的学习速率，参考原始模型
        self.lr = 1e-4 # 推理网络的学习速率
        self.adam_epsilon = 1e-6
        self.warmup_ratio = 0.06

        # 网络参数设置
        self.hidden_bert_size = 768 #bert隐藏层的维度
        self.hidden_ent_size = 768 # 实体维度
        self.hidden_rel_size = 97 # 实体对间关系维度
        self.hidden_inf_size = 10 # 实体对间推理维度
        self.hidden_con_size = 5 # 实体对间关系概念维度

        self.max_grad_norm = 1.0
        self.use_gpu = True
        self.use_wandb = True

        # 下面的内容是关于在分类中是否使用关系或者推理或者概念信息的标记
        # 概念信息无法单独存在，因为维度太小了
        self.use_rel = True
        self.use_inf = False
        self.use_con = False
        # 下面的内容是关于group bilinear的内容
        self.use_group_bilinear = True
        self.block_size = 128
