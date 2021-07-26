import utils
import config
import logging
import numpy as np
import re

import torch
from torch.utils.data import DataLoader
import gensim
from train import train, test, translate,reall,reall2
from data_loader import MTDataset
from utils import english_tokenizer_load,chinese_tokenizer_load
from model import make_model, LabelSmoothing
from text2vec import Similarity

class NoamOpt:
    """Optim wrapper that implements rate."""

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        """Update parameters and rate"""
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        """Implement `lrate` above"""
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))


def get_std_opt(model):
    """for batch_size 32, 5530 steps for one epoch, 2 epoch for warm-up"""
    return NoamOpt(model.src_embed[0].d_model, 1, 10000,
                   torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


def run():
    utils.set_logger(config.log_path)

    train_dataset = MTDataset(config.train_data_path)
    dev_dataset = MTDataset(config.dev_data_path)
    test_dataset = MTDataset(config.test_data_path)

    logging.info("-------- Dataset Build! --------")
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=config.batch_size,
                                  collate_fn=train_dataset.collate_fn)
    dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=config.batch_size,
                                collate_fn=dev_dataset.collate_fn)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=config.batch_size,
                                 collate_fn=test_dataset.collate_fn)

    logging.info("-------- Get Dataloader! --------")
    # 初始化模型
    model = make_model(config.src_vocab_size, config.tgt_vocab_size, config.n_layers,
                       config.d_model, config.d_ff, config.n_heads, config.dropout)
    model_par = torch.nn.DataParallel(model)
    # 训练
    if config.use_smoothing:
        criterion = LabelSmoothing(size=config.tgt_vocab_size, padding_idx=config.padding_idx, smoothing=0.1)
        criterion.cuda()
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0, reduction='sum')
    if config.use_noamopt:
        optimizer = get_std_opt(model)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    train(train_dataloader, dev_dataloader, model, model_par, criterion, optimizer)
    test(test_dataloader, model, criterion)


def check_opt():
    """check learning rate changes"""
    import numpy as np
    import matplotlib.pyplot as plt
    model = make_model(config.src_vocab_size, config.tgt_vocab_size, config.n_layers,
                       config.d_model, config.d_ff, config.n_heads, config.dropout)
    opt = get_std_opt(model)
    # Three settings of the lrate hyperparameters.
    opts = [opt,
            NoamOpt(512, 1, 20000, None),
            NoamOpt(256, 1, 10000, None)]
    plt.plot(np.arange(1, 50000), [[opt.rate(i) for opt in opts] for i in range(1, 50000)])
    plt.legend(["512:10000", "512:20000", "256:10000"])
    plt.show()


def one_sentence_translate(sent, beam_search=True):
    # 初始化模型
    model = make_model(config.src_vocab_size, config.tgt_vocab_size, config.n_layers,
                       config.d_model, config.d_ff, config.n_heads, config.dropout)
#     BOS = english_tokenizer_load().bos_id()  # 2
#     EOS = english_tokenizer_load().eos_id()  # 3
#     src_tokens = [[BOS] + english_tokenizer_load().EncodeAsIds(sent) + [EOS]]
    BOS = chinese_tokenizer_load().bos_id()  # 2
    EOS = chinese_tokenizer_load().eos_id()  # 3
    src_tokens = [[BOS] + chinese_tokenizer_load().EncodeAsIds(sent) + [EOS]]
    batch_input = torch.LongTensor(np.array(src_tokens)).to(config.device)
    translate(batch_input, model, use_beam=beam_search)


def translate_example():
    """单句翻译示例"""
    sent = "近期的政策对策很明确：把最低工资提升 " \
           "到足以一个全职工人及其家庭免于贫困的水平 " \
           "扩大对无子女劳动者的工资所得税减免."
    # tgt:。
    one_sentence_translate(sent, beam_search=True)
    
    
def real_example():
    paragraph = "生活对我们任何人来说都不容易！我们必须努力，最重要的是我们必须相信自己。 \
我们必须相信，我们每个人都能够做得很好，而且，当我们发现这是什么时，我们必须努力工作，直到我们成功。"
 
    sentences = re.split('(。|！|\!|\.|？|\?)',paragraph)         # 保留分割符

    new_sents = []
    for i in range(int(len(sentences)/2)):
        sent = sentences[2*i] + sentences[2*i+1]
        new_sents.append(sent)
    
    
def model_score_test():
    utils.set_logger(config.log_path)
    train_dataset = MTDataset(config.train_data_path)
    dev_dataset = MTDataset(config.dev_data_path)
    test_dataset = MTDataset(config.test_data_path)

    logging.info("-------- Dataset Build! --------")
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=config.batch_size,
                                  collate_fn=train_dataset.collate_fn)
    dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=config.batch_size,
                                collate_fn=dev_dataset.collate_fn)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=config.batch_size,
                                 collate_fn=test_dataset.collate_fn)

    logging.info("-------- Get Dataloader! --------")
    # 初始化模型
    model = make_model(config.src_vocab_size, config.tgt_vocab_size, config.n_layers,
                       config.d_model, config.d_ff, config.n_heads, config.dropout)
    model_par = torch.nn.DataParallel(model)
    # 训练
    if config.use_smoothing:
        criterion = LabelSmoothing(size=config.tgt_vocab_size, padding_idx=config.padding_idx, smoothing=0.1)
        criterion.cuda()
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0, reduction='sum')
    if config.use_noamopt:
        optimizer = get_std_opt(model)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    reall(model)
    pass


def sentence_split(str_centence):
    list_ret1 = list()
    list_ret2 = list()
#     list_ret3 = list()
    for s_str in str_centence.split('.'):
        list_ret1.append(s_str)
#     for s_str in list_ret1:
#         for s_str2 in s_str.split(','):
#             list_ret2.append(s_str2)
    for s_str in list_ret1:
        for s_str2 in s_str.split('!'):
            list_ret2.append(s_str2)
    return list_ret2

def sentence_split_cn(str_centence):
    list_ret1 = list()
    list_ret2 = list()
#     list_ret3 = list()
    for s_str in str_centence.split('。'):
        list_ret1.append(s_str)
#     for s_str in list_ret1:
#         for s_str2 in s_str.split('，'):
#             list_ret2.append(s_str2)
    for s_str in list_ret1:
        for s_str2 in s_str.split('！'):
            list_ret2.append(s_str2)
    return list_ret2

def sentence_split_3(str_centence):
    list_ret = list()
#     list_ret2 = list()
#     list_ret3 = list()
    for s_str in str_centence.split('，'):
        list_ret.append(s_str)
    return list_ret

def model_score_test2():
    utils.set_logger(config.log_path)
    train_dataset = MTDataset(config.train_data_path)
    dev_dataset = MTDataset(config.dev_data_path)
    test_dataset = MTDataset(config.test_data_path)

    logging.info("-------- Dataset Build! --------")
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=config.batch_size,
                                  collate_fn=train_dataset.collate_fn)
    dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=config.batch_size,
                                collate_fn=dev_dataset.collate_fn)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=config.batch_size,
                                 collate_fn=test_dataset.collate_fn)

    logging.info("-------- Get Dataloader! --------")
    # 初始化模型
    model = make_model(config.src_vocab_size, config.tgt_vocab_size, config.n_layers,
                       config.d_model, config.d_ff, config.n_heads, config.dropout)
    model_par = torch.nn.DataParallel(model)
    # 训练
    if config.use_smoothing:
        criterion = LabelSmoothing(size=config.tgt_vocab_size, padding_idx=config.padding_idx, smoothing=0.1)
        criterion.cuda()
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0, reduction='sum')
    if config.use_noamopt:
        optimizer = get_std_opt(model)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    paragraph = "生活对我们任何人来说都不容易！我们必须努力，最重要的是我们必须相信自己。我们必须相信我们每个人都能够做得很好，而且当我们发现这是什么时，我们必须努力工作直到我们成功。或者说这样做我们才能变强，才能够打败敌人。打败敌人才有美好的未来，才能更好的建设祖国。当然，我们也不能只注重一方面。我们还要时常关心身边的人，帮助有困难的人。只有这样社会才能变得更加和谐，国家风气才能变得更好。综上所述，我们要自信善良。这是做人的学问，是一辈子的事情。"
    paragraph2 = "任何人的生活对我们来说都不容易！我们一定要努力，我们相信自己是最重要的。一定要相信我们任何人都可以做的不错，而且在它被我们发现时，直到我们成功都要坚持努力工作。也可以认为我们要变强只能这样做，打败敌人需要这样。光明的未来在击败敌人后，国家也能建造的更好。但是只看到一方面也是不行的。身边的人需要我们时常关心，有困难的人需要我们帮助。更和谐的社会，更好的国家风气都取决于这个。总而言之，自信善良被我们需要。做人的知识就在这里，一辈子都要学。"

    trg_0='Life is not easy for any of us! We have to work hard ,above all we have to believe in ourselves.We have to believe that each of us can do well, and when we find out what this is, we have to work hard until we succeed.Or only in this way can we become stronger, then defeat the enemy. Only by defeating the enemy can we have a bright future ,and build our country better.Of course, we can not just focus on one aspect. We should always care about the people around us and help those in trouble. Only in this way can the society become more harmonious and the national atmosphere become better. To sum up, we should be confident and kind. This is the knowledge of life, and it is a lifetime thing.'
    
    #sentences = re.split('(。|！|\!|\.|？|\?|，|,)',paragraph)         # 保留分割符
    print(trg_0)
#     sentences_trgs= re.split('(.|！|\!|\.|？|\?|,)',trg_0)
#     print(sentences_trgs)
    
    trgs=sentence_split(trg_0)
    print(trgs)
    new_sents = sentence_split_cn(paragraph)
    new_sents2 = sentence_split_cn(paragraph2)
   # sim = Similarity()
    model_g = gensim.models.KeyedVectors.load_word2vec_format('light_Tencent_AILab_ChineseEmbedding.bin', binary=True)

#     test=model_g.n_similarity('今天上班遇到一个帅气的男孩', '今天上班遇到一个漂亮的女孩')
#     print(test)
#     for i in range(int(len(sentences)/2)):
#         sent = sentences[2*i] + sentences[2*i+1]
#         new_sents.append(sent) 
#     print(new_sents)
    num_e=0
    num_c=0
    num_g=0
    for i,j,k in zip(new_sents,trgs,new_sents2):
        print(i,j)
        res=''
        score=0
        if i!='' and j!='':
            res,score=reall2(i,j,model)
            if score<20:
                print('origin:',i)
                print('result:',res+' ',score)
                num_e=num_e+1
            else:
                #res,score=reall2(k,j,model)
                print('超出阈值！')
                t1=sentence_split_3(i)
                t2=sentence_split_3(k)
                num=0
                al=0
                for m,n in zip(t1,t2):
                    if m!='' and n!='':
                        simla=model_g.n_similarity(m, n)
                        al=al+simla
                        num=num+1;
                print('origin:',i)
                print('result:',k)       
               # simla=model_g.n_similarity(i, k)
                num_c=num_c+1
                print('中文相似度：',al/num)
                if(simla>0.93):
                    num_g=num_g+1
    print('整个段落中成功输出的句子数量（英文）：',num_e)
    print('整个段落中成功输出的句子数量（中文）：',num_c)
    print('高于阈值的句子数量：',num_g)

if __name__ == "__main__":
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    import warnings
    warnings.filterwarnings('ignore')
#     run()
#     translate_example()
#     modle_score_test()
    model_score_test2()