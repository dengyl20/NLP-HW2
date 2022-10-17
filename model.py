import torch
import torch.nn as nn
from config import Config
from data_loader import get_numpy_word_embed


class Lstm(nn.Module):
    def __init__(self, word2ix):
        super(Lstm, self).__init__()
        self.config = Config()
        # 定义LSTM层
        # input: (batch, seq_len, input_size=embedding_dim)
        # output: (batch, seq_len, num_directions * proj_size)
        # embedding层
        numpy_embed = get_numpy_word_embed(word2ix, file_path=self.config.GLOVE_PATH, embed_dim=self.config.INPUT_SIZE)
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(numpy_embed), freeze=False).to(self.config.DEVICE)
        self.rnn = nn.LSTM(
            input_size=self.config.INPUT_SIZE,
            hidden_size=self.config.HIDDEN_SIZE,
            # proj_size=self.config.OUTPUT_SIZE,
            num_layers=self.config.LAYERS,
            dropout=self.config.DROP_RATE,
            bidirectional=self.config.LSTM_BI,
            batch_first=True  # 如果为True，输入输出数据格式是(batch, seq_len, feature)
            # 为False，输入输出数据格式是(seq_len, batch, feature)，
        )
        if self.config.LSTM_BI:
            self.linear = nn.Linear(self.config.HIDDEN_SIZE * 2, self.config.OUTPUT_SIZE).to(self.config.DEVICE)
        else:
            self.linear = nn.Linear(self.config.HIDDEN_SIZE, self.config.OUTPUT_SIZE).to(self.config.DEVICE)

    def forward(self, x, _h_s=None, _h_c=None):  # x是输入数据集
        # embedding层
        x = self.embedding(x)
        if _h_s is None and _h_c is None:
            r_out, (h_s, h_c) = self.rnn(x)  # 如果不导入h_s和h_c，默认每次都进行0初始化
        else:
            r_out, (h_s, h_c) = self.rnn(x, (_h_s, _h_c))
        #  h_s和h_c表示每一个隐层的上一时间点输出值和输入细胞状态
        # h_s和h_c的格式均是(num_layers * num_directions, batch, HIDDEN_SIZE)
        # 如果是双向LSTM，num_directions是2，单向是1
        # output = self.hidden_out(r_out)
        output = self.linear(r_out).view(self.config.BATCH_SIZE * self.config.SEQ_LEN, self.config.OUTPUT_SIZE)
        return output, h_s, h_c


class VanillaRNN(nn.Module):
    def __init__(self):
        super(VanillaRNN, self).__init__()
        config = Config()
        self.rnn = nn.RNN(
            input_size=config.INPUT_SIZE,
            hidden_size=config.HIDDEN_SIZE,
            num_layers=config.LAYERS,
            dropout=config.DROP_RATE,
            bidirectional=config.LSTM_BI,
            batch_first=True  # 如果为True，输入输出数据格式是(batch, seq_len, feature)
            # 为False，输入输出数据格式是(seq_len, batch, feature)，
        )
        self.hidden_out = nn.Linear(config.HIDDEN_SIZE, config.OUTPUT_SIZE)  # 最后一个时序的输出接一个全连接层

    def forward(self, x, _h_s):  # x是输入数据集
        r_out, h_s = self.rnn(x, _h_s)  # 如果不导入h_s和h_c，默认每次都进行0初始化
        #  h_s和h_c表示每一个隐层的上一时间点输出值和输入细胞状态
        # h_s和h_c的格式均是(num_layers * num_directions, batch, HIDDEN_SIZE)
        # 如果是双向LSTM，num_directions是2，单向是1
        output = self.hidden_out(r_out)
        return output, h_s

