# Chase Brown
# CSCI Deep Learning Program 4
# RNN

import itertools
import numpy as np
import nltk
import rnn
import os
import matplotlib.pyplot as plt
# break up txt documents into sentences and concatenate them
sentences = []
for file in os.listdir('stories'):
    with open('stories/'+file, encoding='utf8') as f:
        lines = f.read()
        s = nltk.sent_tokenize(lines)
        for i in range(len(s)):
            s[i] = s[i].replace('\n', ' ').lower()
        for j in range(len(s)):
            s[j] = ['%s' % (s[j])]
        sentences = sentences + s
# create the vocab ACSII 0-256
characters = list(map(chr, range(0, 256)))
# Tokenize the sentences into words
toke_sent = []
for i in range(len(sentences)):
    toke_sent.append(list(sentences[i][0]))
# Sequences of length 50
toke_n = []
for i in range(len(toke_sent)):
    if len(toke_sent[i]) == 50:
        toke_n.append(toke_sent[i])

char_freq = nltk.FreqDist(itertools.chain(*toke_sent))
print(char_freq)

vocab = char_freq.most_common(len(characters))
index_to_char = [x[0] for x in characters]
char_to_index = dict([(w, i) for i, w in enumerate(index_to_char)])

print("Using vocabulary size %d." % len(index_to_char))
print("The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1]))
print(char_freq.most_common(10))
f = open('report.doc', 'w')
# Create the training data
XTrain = np.asarray([[char_to_index[w] for w in sent[:-1]] for sent in toke_n])
yTrain = np.asarray([[char_to_index[w] for w in sent[1:]] for sent in toke_n])
# Print a training data example
x_example, y_example = XTrain[42], yTrain[42]
print("x:\n%s\n%s" % (" ".join([index_to_char[x] for x in x_example]), x_example))
print("\ny:\n%s\n%s" % (" ".join([index_to_char[x] for x in y_example]), y_example))
# ++++++++++++++++++++++++Gradient Check++++++++++++++++++++++++++++++++++++++++++++
np.random.seed(10)
grad_check_vocab_size = 100
model = rnn.RNNVanilla(grad_check_vocab_size, 10, bptt_truncate=1000)
model.gradient_check([31, 40, 20, 43], [40, 20, 43, 50], f)
# ========================rnn vanilla 100 hidden units==============================
model = rnn.RNNVanilla(len(index_to_char))
# Limit to 1000 examples to save time
print("Expected Loss for random predictions: %f" % np.log(len(index_to_char)))
print("Actual loss: %f" % model.calculate_loss(XTrain[:1000], yTrain[:1000]))
f.write("Expected Loss for random predictions: %f\n" % np.log(len(index_to_char)))
f.write("Actual loss: %f\n" % model.calculate_loss(XTrain[:1000], yTrain[:1000]))

loss = []
# Train
L = rnn.runit(f, XTrain, yTrain, index_to_char, 'rnnVanilla 100 Hidden Units\n', '100hidden', 100)
loss.append(L)
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ========================rnn vanilla 50 hidden units==================================================================
L = rnn.runit(f, XTrain, yTrain, index_to_char, 'rnnVanilla 50 Hidden Units\n', '50hidden', 50)
loss.append(L)
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ========================rnn vanilla 200 hidden units=================================================================
L = rnn.runit(f, XTrain, yTrain, index_to_char, 'rnnVanilla 200 Hidden Units\n', '200hidden', 200)
loss.append(L)
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Collect sequences of length 25 for half sequence testing
toke_n = []
for i in range(len(toke_sent)):
    if len(toke_sent[i]) == 25:
        toke_n.append(toke_sent[i])
# Create the training data for the new sequence
XTrain = np.asarray([[char_to_index[w] for w in sent[:-1]] for sent in toke_n])
yTrain = np.asarray([[char_to_index[w] for w in sent[1:]] for sent in toke_n])
# ========================rnn vanilla 100 hidden layers/half sequence=================
L = rnn.runit(f, XTrain, yTrain, index_to_char, 'rnnVanilla Half Sequence\n', 'HalfSeq', 100)
loss.append(L)
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Sequences of length 100
toke_n = []
for i in range(len(toke_sent)):
    if len(toke_sent[i]) == 100:
        toke_n.append(toke_sent[i])
# Create the training data
XTrain = np.asarray([[char_to_index[w] for w in sent[:-1]] for sent in toke_n])
yTrain = np.asarray([[char_to_index[w] for w in sent[1:]] for sent in toke_n])
# ========================rnn vanilla 100 hidden layers/half sequence=================
L, epoch = rnn.runit(f, XTrain, yTrain, index_to_char, 'rnnVanilla Double Sequence\n', 'DoubleSeq', 100)
loss.append(L)
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# plt.plot(epoch, loss[0], label='100Hidden')
# plt.plot(epoch, loss[1], label='50Hidden')
# plt.plot(epoch, loss[2], label='200Hidden')
# plt.plot(epoch, loss[3], label='HalfSeq')
# plt.plot(epoch, loss[4], label='DoubleSeq')
# leg = plt.legend(loc='best', ncol=2, mode="expand", shadow=True, fancybox=True)
# leg.get_frame().set_alpha(0.5)
# plt.show()
f.close()
print('break')