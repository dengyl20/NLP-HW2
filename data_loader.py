# Dataset and DataLoader
from collections import Counter

from tqdm import tqdm


def read_file(filename: str) -> list:
    """Read the file data"""
    contents = []
    with open(filename, 'r') as f:
        print("Reading file: {}".format(filename))
        for line in tqdm(f.readlines()):
            try:
                line_content = line.strip().split(' ')
                if line_content:
                    contents.extend(line_content)
            except:
                pass
    return contents


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


if __name__ == '__main__':
    contents = read_file('data/ptb.train.txt')
    print(len(contents))

    print(len(set(contents)))
    for i in range(100):
        print(contents[i], end=' ')
    # save the vocabulary
    build_vocab(train_dir='data/ptb.train.txt', vocab_dir='data/vocab.txt')
