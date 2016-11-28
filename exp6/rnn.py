import itertools
import csv
import nltk
import numpy as np

from theano import shared, function
import theano
import theano.tensor as T
import datetime

vocabulary = 8000
unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"

'''
    receive the sentences from the file
'''
def get_sentences():
    print "reading csv file ..."
    with open('reddit-comments-2015-08.csv', 'rb') as f:
        reader = csv.reader(f, skipinitialspace=True)
        reader.next()
        sentences = itertools.chain(*[nltk.sent_tokenize(x[0].decode('utf-8').lower()) for x in reader])
        sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in sentences]
    print "finish"
    return sentences, [nltk.word_tokenize(sent) for sent in sentences]

def preprocess(sentences, tokenized):
    print "tokenized ..."
    word_freq = nltk.FreqDist(itertools.chain(*tokenized))
    vocab = word_freq.most_common(vocabulary - 1)
    idx2word = [x[0] for x in vocab]
    idx2word.append(unknown_token)
    word2idx = dict([(w, i) for i, w in enumerate(idx2word)])

    for i, sent in enumerate(tokenized):
        tokenized[i] = [w if w in word2idx else unknown_token for w in sent]

    X_train = np.asarray([[word2idx[w] for w in sent[:-1]] for sent in tokenized])
    Y_train = np.asarray([[word2idx[w] for w in sent[1:]] for sent in tokenized])
    print "finish tokenized"
    return X_train, Y_train

def softmax(x):
    xt = np.exp(x - np.max(x))
    return xt / np.sum(xt)

class RNNNumpy:
    def __init__(self, word_dim, hidden_dim=100, bptt_truncate=4):
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate

        self.U = np.random.uniform(- np.sqrt(1. / word_dim), np.sqrt(1. / word_dim), (hidden_dim, word_dim))
        self.V = np.random.uniform(- np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (word_dim, hidden_dim))
        self.W = np.random.uniform(- np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (hidden_dim, hidden_dim))

    def forward_propagation(self, x):
        T = len(x)
        s = np.zeros((T + 1, self.hidden_dim))
        o = np.zeros((T, self.word_dim))
        for t in np.arange(T):
            # because we use the ont-hot, don't need multiply method
            s[t] = np.tanh(self.U[:, x[t]] + self.W.dot(s[t - 1]))
            o[t] = softmax(self.V.dot(s[t]))
        return [o, s]

    def predict(self, x):
        o, s = self.forward_propagation(x)
        return np.argmax(o, axis=1)
    
    def calculate_total_loss(self, x, y):
        loss = 0.
        for i in np.arange(len(y)):
            o, s = self.forward_propagation(x[i])
            correct_word_predictions = o[np.arange(len(y[i])), y[i]]
            loss += -1. * np.sum(np.log(correct_word_predictions))
        return loss
    
    def calculate_loss(self, x, y):
        n = np.sum((len(yi) for yi in y))
        return self.calculate_total_loss(x, y) / n

    def bptt(self, x, y):
        T = len(y)
        o, s = self.forward_propagation(x)

        dLdU = np.zeros(self.U.shape)
        dLdV = np.zeros(self.V.shape)
        dLdW = np.zeros(self.W.shape)
        delta_o = o
        delta_o[np.arange(len(y)), y] -= 1

        for t in np.arange(T)[::-1]:
            dLdV += np.outer(delta_o[t], s[t].T)
            delta_t = self.V.T.dot(delta_o[t]) * (1 - (s[t] ** 2))
            for bptt_step in np.arange(max(0, t - self.bptt_truncate), t + 1)[::-1]:
                dLdW += np.outer(delta_t, s[bptt_step - 1])
                dLdU[:, x[bptt_step]] += delta_t
                delta_t = self.W.T.dot(delta_t) * (1 - s[bptt_step] ** 2)
        return [dLdU, dLdV, dLdW]
    
    def gradient_check(self, x, y, h=0.001, error_threshold=0.01):
        bptt_gradient = self.bptt(x, y)

    def sgd_step(self, x, y, learning_rate):
        dLdU, dLdV, dLdW = self.bptt(x, y)
        self.U -= learning_rate * dLdU
        self.V -= learning_rate * dLdV
        self.W -= learning_rate * dLdW

def train_with_sgd(model, x_train, y_train, learning_rate=0.005, nepoch=100, evaluate_loss_after=5):
    losses = []
    sample_seen = 0
    for epoch in range(nepoch):
        if (epoch % evaluate_loss_after) == 0:
            loss = model.calculate_loss(x_train, y_train)
            losses.append((sample_seen, loss))
            time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print "%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, sample_seen, epoch, loss)
            if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                learning_rate = learning_rate * .5
                print "Setting learning rate to %f" % learning_rate
        for i in range(len(y_train)):
            model.sgd_step(x_train[i], y_train[i], learning_rate)
            sample_seen += 1

def main():
    sentences, tokenized = get_sentences()
    x, y = preprocess(sentences, tokenized)
    model = RNNNumpy(vocabulary)
    train_with_sgd(model, x[:10], y[:10])
    pword = model.predict(x[10])
    print pword

if __name__ == '__main__':
    main()
