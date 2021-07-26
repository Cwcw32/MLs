import torch
import torch.nn as nn
from torch.autograd import Variable

import logging
import sacrebleu
from tqdm import tqdm
import numpy as np

import config
from beam_decoder import beam_search
from model import batch_greedy_decode,make_model
from utils import chinese_tokenizer_load,english_tokenizer_load


def one_sentence_translate(sent, i,beam_search=True):
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
   # trg=[]
    res = "The near-term policy remedies are clear: raise the minimum wage to a level that will keep a " \
           "fully employed worker and his or her family out of poverty, and extend the earned-income tax credit " \
           "to childless workers."
    trg=[res]
#    print(trg)
    translate2(batch_input,trg, model,i, use_beam=beam_search)
    

def translate_example(i):
    """单句翻译示例"""
    sent = "近期的政策很明确：把最低工资提升 " \
           "来让一个全职工人及其家庭免于贫困的水平 " \
           "从而扩大对无子女劳动者的工资所得税减免."
    # tgt:。
    one_sentence_translate(sent, i,beam_search=True)
    

def reall(model):
    for epoch in range(1, config.epoch_num + 1):
        #model.load_state_dict(torch.load(config.model_path+'.pth'))
        print(epoch,':')
        translate_example(epoch) 

        
        
def one_sentence_translate2(sent,trg_0, i,beam_search=True):
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
   # trg=[]
#     res = "The near-term policy remedies are clear: raise the minimum wage to a level that will keep a " \
#            "fully employed worker and his or her family out of poverty, and extend the earned-income tax credit " \
#            "to childless workers."
    trg=[trg_0]
#    print(trg)
    result,score=translate2(batch_input,trg, model,i, use_beam=beam_search)
    return result,score # 翻译后的字符串和分数
    
    
def translate_example2(sent,trg_0,i):
    """单句翻译示例
    sent:要翻译的句子
    trg_0:自己的英文
    i：模型选择罢了
    """
    # tgt:。
    res,score=one_sentence_translate2(sent,trg_0, i,beam_search=True)
    return res,score # 翻译后的字符串和分数
    


def reall2(sent,trg_0,model):
    res_temp=''
    score_temp=0
    res=''
    score=0
    print(sent)
    for epoch in range(1, config.epoch_num + 1):
     #   model.load_state_dict(torch.load(config.model_path+'.pth'))
        #print(epoch,':')
        res_temp,score_temp=translate_example2(sent,trg_0,epoch) 
        if(score_temp>score):
            res=res_temp
            score=score_temp
   # print(res+' ',score)
    return res,score  # 翻译后的字符串和分数
        
    
def run_epoch(data, model, loss_compute):
    total_tokens = 0.
    total_loss = 0.

    for batch in tqdm(data):
        out = model(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)

        total_loss += loss
        total_tokens += batch.ntokens
    return total_loss / total_tokens


def train(train_data, dev_data, model, model_par, criterion, optimizer):
    """训练并保存模型"""
    # 初始化模型在dev集上的最优Loss为一个较大值
    best_bleu_score = 0.0
    early_stop = config.early_stop
    for epoch in range(1, config.epoch_num + 1):
        # 模型训练
        model.train()
        train_loss = run_epoch(train_data, model_par,
                               MultiGPULossCompute(model.generator, criterion, config.device_id, optimizer))
        logging.info("Epoch: {}, loss: {}".format(epoch, train_loss))
        # 模型验证
        model.eval()

  #      model.load_state_dict(torch.load(config.model_path+'.pth'))
        torch.save(model.state_dict(), config.model_path+'%i'%epoch+'.pth')
        
      #  model.load_state_dict(torch.load(config.model_path))
        translate_example(epoch)
#         dev_loss = run_epoch(dev_data, model_par,
#                              MultiGPULossCompute(model.generator, criterion, config.device_id, None))
#         bleu_score = evaluate(dev_data, model)
#         logging.info('Epoch: {}, Dev loss: {}, Bleu Score: {}'.format(epoch, dev_loss, bleu_score))

#         # 如果当前epoch的模型在dev集上的loss优于之前记录的最优loss则保存当前模型，并更新最优loss值
#         if bleu_score > best_bleu_score:
#             torch.save(model.state_dict(), config.model_path)
#             best_bleu_score = bleu_score
#             early_stop = config.early_stop
#             logging.info("-------- Save Best Model! --------")
#         else:
#             early_stop -= 1
#             logging.info("Early Stop Left: {}".format(early_stop))
#         if early_stop == 0:
#             logging.info("-------- Early Stop! --------")
#             break


class LossCompute:
    """简单的计算损失和进行参数反向传播更新训练的函数"""

    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            if config.use_noamopt:
                self.opt.optimizer.zero_grad()
            else:
                self.opt.zero_grad()
        return loss.data.item() * norm.float()


class MultiGPULossCompute:
    """A multi-gpu loss compute and train function."""

    def __init__(self, generator, criterion, devices, opt=None, chunk_size=5):
        # Send out to different gpus.
        self.generator = generator
        self.criterion = nn.parallel.replicate(criterion, devices=devices)
        self.opt = opt
        self.devices = devices
        self.chunk_size = chunk_size

    def __call__(self, out, targets, normalize):
        total = 0.0
        generator = nn.parallel.replicate(self.generator, devices=self.devices)
        out_scatter = nn.parallel.scatter(out, target_gpus=self.devices)
        out_grad = [[] for _ in out_scatter]
        targets = nn.parallel.scatter(targets, target_gpus=self.devices)

        # Divide generating into chunks.
        chunk_size = self.chunk_size
        for i in range(0, out_scatter[0].size(1), chunk_size):
            # Predict distributions
            out_column = [[Variable(o[:, i:i + chunk_size].data,
                                    requires_grad=self.opt is not None)]
                          for o in out_scatter]
            gen = nn.parallel.parallel_apply(generator, out_column)

            # Compute loss.
            y = [(g.contiguous().view(-1, g.size(-1)),
                  t[:, i:i + chunk_size].contiguous().view(-1))
                 for g, t in zip(gen, targets)]
            loss = nn.parallel.parallel_apply(self.criterion, y)

            # Sum and normalize loss
            l_ = nn.parallel.gather(loss, target_device=self.devices[0])
            l_ = l_.sum() / normalize
            total += l_.data

            # Backprop loss to output of transformer
            if self.opt is not None:
                l_.backward()
                for j, l in enumerate(loss):
                    out_grad[j].append(out_column[j][0].grad.data.clone())

        # Backprop all loss through transformer.
        if self.opt is not None:
            out_grad = [Variable(torch.cat(og, dim=1)) for og in out_grad]
            o1 = out
            o2 = nn.parallel.gather(out_grad,
                                    target_device=self.devices[0])
            o1.backward(gradient=o2)
            self.opt.step()
            if config.use_noamopt:
                self.opt.optimizer.zero_grad()
            else:
                self.opt.zero_grad()
        return total * normalize


def evaluate(data, model, mode='dev', use_beam=True):
    """在data上用训练好的模型进行预测，打印模型翻译结果"""
    sp_chn = english_tokenizer_load()
    trg = []
    res = []
    with torch.no_grad():
        # 在data的英文数据长度上遍历下标
        for batch in tqdm(data):
            # 对应的中文句子
            cn_sent = batch.trg_text
            src = batch.src
            src_mask = (src != 0).unsqueeze(-2)
            if use_beam:
                decode_result, _ = beam_search(model, src, src_mask, config.max_len,
                                               config.padding_idx, config.bos_idx, config.eos_idx,
                                               config.beam_size, config.device)
            else:
                decode_result = batch_greedy_decode(model, src, src_mask,
                                                    max_len=config.max_len)
            decode_result = [h[0] for h in decode_result]
            translation = [sp_chn.decode_ids(_s) for _s in decode_result]
            trg.extend(cn_sent)
            res.extend(translation)
    if mode == 'test':
        with open(config.output_path, "w") as fp:
            for i in range(len(trg)):
                line = "idx:" + str(i) + trg[i] + '|||' + res[i] + '\n'
                fp.write(line)
    trg = [trg]
    bleu = sacrebleu.corpus_bleu(res, trg)
    return float(bleu.score)


def test(data, model, criterion):
    with torch.no_grad():
        # 加载模型
        model.load_state_dict(torch.load(config.model_path+'20.pth'))
        model_par = torch.nn.DataParallel(model)
        model.eval()
        # 开始预测
        test_loss = run_epoch(data, model_par,
                              MultiGPULossCompute(model.generator, criterion, config.device_id, None))
        bleu_score = evaluate(data, model, 'test')
        logging.info('Test loss: {},  Bleu Score: {}'.format(test_loss, bleu_score))


def translate(src, model, use_beam=True):
    """用训练好的模型进行预测单句，打印模型翻译结果"""
    sp_chn = english_tokenizer_load()
    sp_e = chinese_tokenizer_load()

    with torch.no_grad():
        model.load_state_dict(torch.load(config.model_path))
        model.eval()
        src_mask = (src != 0).unsqueeze(-2)
        if use_beam:
            decode_result, _ = beam_search(model, src, src_mask, config.max_len,
                                           config.padding_idx, config.bos_idx, config.eos_idx,
                                           config.beam_size, config.device)
            decode_result = [h[0] for h in decode_result]
        else:
            decode_result = batch_greedy_decode(model, src, src_mask, max_len=config.max_len)
        translation = [sp_chn.decode_ids(_s) for _s in decode_result]
        print(translation[0])
#         res=[sp_e.decode_ids(_s) for _s in src]
#         bleu = sacrebleu.corpus_bleu(src, translation[0])
#         print(bleu.score)

def translate2(src, trg,model,i, use_beam=True):
    """用训练好的模型进行预测单句，打印模型翻译结果"""
    sp_chn = english_tokenizer_load()
    sp_e = chinese_tokenizer_load()

    with torch.no_grad():
        model.load_state_dict(torch.load(config.model_path+'%i'%i+'.pth'))
        model.eval()
        src_mask = (src != 0).unsqueeze(-2)
        if use_beam:
            decode_result, _ = beam_search(model, src, src_mask, config.max_len,
                                           config.padding_idx, config.bos_idx, config.eos_idx,
                                           config.beam_size, config.device)
            decode_result = [h[0] for h in decode_result]
        else:
            decode_result = batch_greedy_decode(model, src, src_mask, max_len=config.max_len)
        translation = [sp_chn.decode_ids(_s) for _s in decode_result]
      #  print(translation[0])
      #  print(trg)
        bleu = sacrebleu.corpus_bleu(trg, translation[0])
        print(i,' ',bleu.score,' ',translation[0])
        
    return translation[0],bleu.score