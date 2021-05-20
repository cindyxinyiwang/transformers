import argparse


def get_vocab_bpe_to_ngram(train_file, vocab_file, max_size=100000):
    vocab = {}
    print("Creating char ngram vocab from {} with max_size {}...".format(train_file, max_size))
    with open(train_file, 'r') as myfile:
        for line in myfile:
            toks = line.split()
            w = toks[0]
            if w.startswith("▁"):
                w = w[1:]
            if w:
                vocab[w] = 1
    print("Writing vocab to {} with size {}...".format(vocab_file, max_size))
    vocab_file = open(vocab_file, 'w')
    for w, c in vocab.items():
        vocab_file.write("{}\t{}\n".format(w, c))
    vocab_file.close()


def get_vocab_char_ngram(train_file, vocab_file, n=4, max_size=100000):
    vocab = {}
    print("Creating char ngram vocab from {} with max_size {}...".format(train_file, max_size))
    with open(train_file, 'r') as myfile:
        for line in myfile:
            toks = line.split()
            for w in toks:
                for i in range(len(w)):
                    for j in range(i+1, min(i+n, len(w))+1):
                        char = w[i:j]
                        if char not in vocab:
                            vocab[char] = 0
                        vocab[char] += 1
    max_size = min(max_size, len(vocab))
    vocab = sorted(vocab.items(), key=lambda kv: kv[1], reverse=True)[:max_size]
    print("Writing vocab to {} with size {}...".format(vocab_file, max_size))
    vocab_file = open(vocab_file, 'w')
    for w, c in vocab:
        vocab_file.write("{}\t{}\n".format(w, c))
    vocab_file.close()


def get_vocab_word(train_file, vocab_file, max_size=100000):
    vocab = {}
    print("Creating word vocab from {} with max_size {}...".format(train_file, max_size))
    with open(train_file, 'r') as myfile:
        for line in myfile:
            toks = line.split()
            for t in toks:
                if t not in vocab:
                    vocab[t] = 0
                vocab[t] += 1
    max_size = min(max_size, len(vocab))
    vocab = sorted(vocab.items(), key=lambda kv: kv[1], reverse=True)[:max_size]
    print("Writing vocab to {} with size {}...".format(vocab_file, max_size))
    vocab_file = open(vocab_file, 'w')
    for w, c in vocab:
        vocab_file.write("{}\t{}\n".format(w, c))
    vocab_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_file", default=None, type=str, required=True)
    parser.add_argument("--vocab_file", default=None, type=str, required=True)
    parser.add_argument("--vocab_type", default=None, type=str, help="[word|char_ngram]")
    parser.add_argument("--max_size", default=100000, type=int)
    parser.add_argument("--n", default=4, type=int)

    args = parser.parse_args()

    if args.vocab_type == "word":
        get_vocab_word(args.train_file, args.vocab_file, max_size=args.max_size)
    elif args.vocab_type == "char_ngram":
        get_vocab_char_ngram(args.train_file, args.vocab_file, args.n, max_size=args.max_size)
    else:
        get_vocab_bpe_to_ngram(args.train_file, args.vocab_file, max_size=args.max_size)