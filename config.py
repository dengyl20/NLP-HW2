import torch


class Config(object):
    TRAIN_DATA_PATH = "data/ptb.train.txt"
    TEST_DATA_PATH = "data/ptb.test.txt"
    VOCAB_PATH = "data/vocab.txt"
    BATCH_SIZE = 64
    SEQ_LEN = 3
    VOCAB_SIZE = 10000
    INPUT_SIZE = 300
    HIDDEN_SIZE = 200
    LAYERS = 1
    DROP_RATE = 0
    LSTM_BI = True
    OUTPUT_SIZE = 10000
    LR = 0.001
    EPOCH = 30
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    MODEL = "lstm"
    GLOVE_PATH = "data/glove.6B.300d.txt"
    MODEL_PATH = "model/lstm.m"
