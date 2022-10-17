class Config(object):
    TRAIN_DATA_PATH = "data/train.txt"
    TEST_DATA_PATH = "data/test.txt"
    VOCAB_PATH = "data/vocab.txt"
    BATCH_SIZE = 64
    SEQ_LEN = 3
    VOCAB_SIZE = 10000
    INPUT_SIZE = 300
    HIDDEN_SIZE = 200
    LAYERS = 1
    DROP_RATE = 0
    LSTM_BI = True
    OUTPUT_SIZE = 9998
