# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import torch.optim as optim
from model import Lstm, VanillaRNN
from data_loader import DataLoader, load_dataset
from config import Config
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F

class TrainModel(object):
    def __init__(self):
        self.config = Config()
        self.device = self.config.DEVICE

    def train(self, data_loader, valid_loader, model, optimizer, criterion, word_to_id):
        """训练模型"""
        last_prep = 10000
        epoch_loss = [] 
        for epoch in range(self.config.EPOCH):
            model.train()
            t_bar = tqdm(data_loader,total=len(data_loader))
            total_loss = 0
            for x, y in t_bar:
                # 1.处理数据
                # x: tensor(torch.float32) [batch_size, seq_len]
                # y: tensor(torch.float32) [batch_size, seq_len]
                x = torch.tensor(x, dtype=torch.long).to(self.device)
                y = torch.tensor(y, dtype=torch.long).to(self.device)
                y = y.view(self.config.BATCH_SIZE * self.config.SEQ_LEN)

                optimizer.zero_grad()

                # 初始化hidden为(c0, h0): ((layer_num， batch_size, hidden_dim)，(layer_num， batch_size, hidden_dim)）
                # hidden = model.init_hidden(self.config.layer_num, x.size()[1])

                # 2.前向计算
                # print(input.size(), hidden[0].size(), target.size())
                output, _, _ = model(x)
                loss = F.nll_loss(output, y)  # output:(max_len*batch_size,vocab_size), target:(max_len*batch_size)
                total_loss += loss
                # 反向计算梯度
                loss.backward()

                # 权重更新
                optimizer.step()

                t_bar.set_description('epoch: %d,  loss: %f' % (epoch, loss.data))
            epoch_loss.append((total_loss/self.config.BATCH_SIZE).cpu().data)
            if epoch % 1 == 0:
                # 保存模型
                model.eval()
                t_bar = tqdm(valid_loader)
                tol = 0
                cor = 0
                prep = 0
                valid_loss = 0
                for x, y in t_bar:
                    x = torch.tensor(x, dtype=torch.long).to(self.device)
                    y = torch.tensor(y, dtype=torch.long).to(self.device)
                    y = y.view(self.config.BATCH_SIZE * self.config.SEQ_LEN)
                    output, _, _ = model(x)
                    prep += F.cross_entropy(output,y)
                    # sm = nn.Softmax(dim = 1)
                    loss = F.nll_loss(output, y)  # output:(max_len*batch_size,vocab_size), target:(max_len*batch_size)
                    valid_loss += loss
                    # output = torch.argmax(output,dim = 1)
                    # cnt = (output == y).sum().item()
                    # cor += cnt
                    # tol += self.config.BATCH_SIZE * self.config.SEQ_LEN
                prep = torch.exp(prep / self.config.BATCH_SIZE)
                valid_loss = valid_loss / self.config.BATCH_SIZE
                # accuracy = cor/tol
                print("epoch:%d,valid_loss:%f,valid:preplexity: %f" % (epoch, valid_loss, prep))
                if(prep < last_prep):
                    last_prep = prep
                    torch.save(model.state_dict(), '%s_%s.pth' % (self.config.MODEL_PATH, epoch))
        plt.xlabel("epoch")#x轴上的名字
        plt.ylabel("loss")#y轴上的名字
        plt.plot(range(self.config.EPOCH),epoch_loss)
        plt.savefig('./training_loss.jpg')

    def run(self):
        # 1 获取数据
        train_data, valid_data, test_data, word_to_id = load_dataset(train_dir=self.config.TRAIN_DATA_PATH,
                                                         valid_dir=self.config.VALID_DATA_PATH,
                                                         test_dir=self.config.TEST_DATA_PATH,
                                                         vocab_dir=self.config.VOCAB_PATH)
        vocab_size = len(word_to_id)
        print('样本数：%d' % len(train_data))
        print('词典大小： %d' % vocab_size)

        # 2 设置dataloader
        train_data_loader = DataLoader(data=train_data, batch_size=self.config.BATCH_SIZE,
                                       shuffle=False, seq_len=self.config.SEQ_LEN)
        valid_data_loader = DataLoader(data=valid_data, batch_size=self.config.BATCH_SIZE,
                                       shuffle=False, seq_len=self.config.SEQ_LEN)
        test_data_loader = DataLoader(data=test_data, batch_size=self.config.BATCH_SIZE,
                                       shuffle=False, seq_len=self.config.SEQ_LEN)
        # 3 创建模型
        if self.config.MODEL == 'lstm':
            model = Lstm(word2ix=word_to_id)
        else:
            model = VanillaRNN()
        model.to(self.device)

        # 4 创建优化器
        optimizer = optim.Adam(model.parameters(), lr=self.config.LR)

        # 5 创建损失函数,使用与log_softmax的输出
        # input: (N, C) where C = number of classes, or (N, C, d1, d2, ..., dk) with K≥1 for K-dimensional loss.
        # target: (N) where each value is 0≤targets[i]≤C−1
        criterion = nn.NLLLoss()

        # 6.训练
        self.train(train_data_loader, valid_data_loader, model, optimizer, criterion, word_to_id)
        self.test(test_data_loader, model, criterion)

    def generate_head_test(self, model, head_sentence, word_to_ix, ix_to_word):
        """生成藏头诗"""
        poetry = []
        head_char_len = len(head_sentence)  # 要生成的句子的数量
        sentence_len = 0  # 当前句子的数量
        pre_char = '<START>'  # 前一个已经生成的字

        # 准备第一步要输入的数据
        input = (torch.Tensor([word_to_ix['<START>']]).view(1, 1).long()).to(self.device)
        hidden = model.init_hidden(self.config.layer_num, 1)

        for i in range(self.config.max_gen_len):
            # 前向计算出概率最大的当前词
            output, hidden = model(input, hidden)
            top_index = output.data[0].topk(1)[1][0].item()
            char = ix_to_word[top_index]

            # 句首的字用藏头字代替
            if pre_char in ['。', '！', '<START>']:
                if sentence_len == head_char_len:
                    break
                else:
                    char = head_sentence[sentence_len]
                    sentence_len += 1
                    input = (input.data.new([word_to_ix[char]])).view(1, 1)
            else:
                input = (input.data.new([top_index])).view(1, 1)

            poetry.append(char)
            pre_char = char

        return poetry
    def test(self, test_data_loader, model, criterion):
        model.eval()
        t_bar = tqdm(test_data_loader)
        tol = 0
        cor = 0
        prep = 0
        test_loss = 0
        for x, y in t_bar:
            x = torch.tensor(x, dtype=torch.long).to(self.device)
            y = torch.tensor(y, dtype=torch.long).to(self.device)
            y = y.view(self.config.BATCH_SIZE * self.config.SEQ_LEN)
            output, _, _ = model(x)
            cross_entropy_loss = nn.CrossEntropyLoss()
            prep += torch.exp(cross_entropy_loss(output,y))
            sm = nn.Softmax(dim = 1)
            loss = criterion(torch.log(sm(output)+1e-10), y)  # output:(max_len*batch_size,vocab_size), target:(max_len*batch_size)
            test_loss += loss
            # output = torch.argmax(output,dim = 1)
            # cnt = (output == y).sum().item()
            # cor += cnt
            # tol += self.config.BATCH_SIZE * self.config.SEQ_LEN
        prep = prep / self.config.BATCH_SIZE
        test_loss = test_loss / self.config.BATCH_SIZE
        # accuracy = cor/tol
        print("Test loss: %f, preplexity: %f" % (test_loss, prep.data))

if __name__ == '__main__':
    obj = TrainModel()
    obj.run()
