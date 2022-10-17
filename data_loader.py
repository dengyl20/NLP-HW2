# Dataset and DataLoader
from collections import Counter

import numpy as np
from numpy import ndarray
from tqdm import tqdm

from config import Config


def read_file(filename: str) -> list:
    """Read the file data"""
    content = []
    with open(filename, 'r') as f:
        print("Reading file: {}".format(filename))
        for line in tqdm(f.readlines()):
            try:
                line_content = line.strip().split(' ')
                if line_content:
                    content.extend(line_content)
            except:
                pass
    return content


def build_vocab(train_dir: str, vocab_dir: str, vocab_size: int = 10000) -> None:
    """Build vocabulary based on the training set and store it"""
    data_train = read_file(train_dir)

    counter = Counter(data_train)
    count_pairs = counter.most_common(vocab_size - 1)  # Most common words
    words, _ = list(zip(*count_pairs))
    # Add a <PAD> to make all text pad to the same length
    words = ['<PAD>'] + list(words)
    print(words)

    with open(vocab_dir, 'w') as f:
        f.write('\n'.join(words) + '\n')


def read_vocab(vocab_dir: str) -> list:
    """Read vocabulary"""
    with open(vocab_dir, 'r') as f:
        words = [_.strip() for _ in f.readlines()]
    return words


def load_dataset(train_dir: str, test_dir: str, vocab_dir: str) -> tuple:
    """Load the dataset"""
    # Read the vocabulary
    words = read_vocab(vocab_dir)
    # Build a dictionary to map words to indices
    word_to_id = dict(zip(words, range(len(words))))

    # Read the training set
    train_data = read_file(train_dir)
    train_data = [word_to_id[x] for x in train_data]

    # Read the test set
    test_data = read_file(test_dir)
    test_data = [word_to_id[x] for x in test_data]

    return train_data, test_data, word_to_id


# > This class loads data from a file and returns a list of lists
class DataLoader(object):
    """Data Loader"""

    def __init__(self, data: list, batch_num: int = 1000, batch_size: int = 64, seq_len: int = 3,
            shuffle: bool = False):
        """Initialize the DataLoader"""
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.batch_num = batch_num
        self.data = data
        self.shuffle = shuffle
        # The number of steps in each batch
        self.seq_len = seq_len
        self.epoch_size = ((len(data) // batch_size) - 1) // seq_len
        self.i = 0
        # Reshape the data
        self.data = self._reshape()

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        """Get the next batch"""
        if self.i == self.epoch_size:
            self.i = 0
            raise StopIteration

        x = self.data[self.i * self.batch_size: (self.i + 1) * self.batch_size,
            self.i * self.seq_len: (self.i + 1) * self.seq_len]
        y = self.data[self.i * self.batch_size: (self.i + 1) * self.batch_size,
            self.i * self.seq_len + 1: (self.i + 1) * self.seq_len + 1]
        self.i += 1
        return x, y

    def _reshape(self) -> ndarray:
        new_data = np.zeros((self.batch_num, self.seq_len + 1))
        for item in range(self.batch_num):
            new_data[item, :] = np.array(self.data[item * self.seq_len: (item + 1) * self.seq_len + 1])
        if self.shuffle:
            np.random.shuffle(new_data)
        return new_data


if __name__ == '__main__':
    config = Config()
    contents = read_file(config.TRAIN_DATA_PATH)
    print(len(contents))

    print(len(set(contents)))
    for i in range(100):
        print(contents[i], end=' ')
    # save the vocabulary
    build_vocab(train_dir=config.TRAIN_DATA_PATH, vocab_dir=config.VOCAB_PATH, vocab_size=config.VOCAB_SIZE)
