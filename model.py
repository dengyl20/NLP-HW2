import torch.nn as nn
from config import Config


class Lstm(nn.Module):
    def __init__(self):
        super(Lstm, self).__init__()
        config = Config()
        self.rnn = nn.LSTM(
            input_size = config.INPUT_SIZE, 
            hidden_size = config.HIDDEN_SIZE, 
            num_layers = config.LAYERS,
            dropout = config.DROP_RATE,
            bidirectional = config.LSTM_BI,
            batch_first = True    # 如果为True，输入输出数据格式是(batch, seq_len, feature)
                                  # 为False，输入输出数据格式是(seq_len, batch, feature)，
        )
        self.hidden_out = nn.Linear(config.HIDDEN_SIZE, config.OUTPUT_SIZE)  # 最后一个时序的输出接一个全连接层
        self.h_s = None
        self.h_c = None
        
    def forward(self, x):    # x是输入数据集
        r_out, (h_s, h_c) = self.rnn(x)   # 如果不导入h_s和h_c，默认每次都进行0初始化
                                          #  h_s和h_c表示每一个隐层的上一时间点输出值和输入细胞状态
                                          # h_s和h_c的格式均是(num_layers * num_directions, batch, HIDDEN_SIZE)
                                          # 如果是双向LSTM，num_directions是2，单向是1
        output = self.hidden_out(r_out)
        return output
