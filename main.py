# Chase Brown
# CSCI Deep Learning Program 4
# RNN

import itertools
import numpy as np
import nltk
import rnn
import os

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

# Create the training data
XTrain = np.asarray([[char_to_index[w] for w in sent[:-1]] for sent in toke_n])
yTrain = np.asarray([[char_to_index[w] for w in sent[1:]] for sent in toke_n])
# Print a training data example
x_example, y_example = XTrain[42], yTrain[42]
print("x:\n%s\n%s" % (" ".join([index_to_char[x] for x in x_example]), x_example))
print("\ny:\n%s\n%s" % (" ".join([index_to_char[x] for x in y_example]), y_example))
np.random.seed(10)
# ========================rnn vanilla 100 hidden units==============================
model = rnn.RNNVanilla(len(index_to_char))
# Limit to 1000 examples to save time
print("Expected Loss for random predictions: %f" % np.log(len(index_to_char)))
print("Actual loss: %f" % model.calculate_loss(XTrain[:1000], yTrain[:1000]))
f = open('report.doc', 'w')
f.write("Expected Loss for random predictions: %f\n" % np.log(len(index_to_char)))
f.write("Actual loss: %f\n" % model.calculate_loss(XTrain[:1000], yTrain[:1000]))
# Train on a small subset of the data to see what happens
losses, strings = rnn.train_with_sgd(model, XTrain, yTrain, index_to_char, learning_rate=0.005,
                            nepoch=101, evaluate_loss_after=1)

f.write('rnnVaninlla 100 Hidden Units:\n')
ep = [20, 40, 60, 80, 100]
x_example, y_example = XTrain[42], yTrain[42]
f.write("x:\n%s\n" % (" ".join([index_to_char[x] for x in x_example])))
f.write("\ny:\n%s\n" % (" ".join([index_to_char[x] for x in y_example])))

for i in range(len(strings)):
    print("index_to_word>")
    print('%s' % " ".join([index_to_char[x] for x in strings[i]]))
    f.write('epoch %d : ' % ep[i])
    f.write('%s\n' % " ".join([index_to_char[x] for x in strings[i]]))
loss = []
epoch = []
for i in range(len(losses)):
    loss.append(losses[i][1])
    epoch.append(i)
# Plot the epoch.loss
rnn.lossvsepoch(epoch, loss, '100hidden')
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ========================rnn vanilla 50 hidden units==============================
# Train on a small subset of the data to see what happens
model = rnn.RNNVanilla(len(index_to_char), hidden_dim=50)
losses1, strings1 = rnn.train_with_sgd(model, XTrain, yTrain, index_to_char, learning_rate=0.005,
                            nepoch=101, evaluate_loss_after=1)

f.write('rnnVaninlla 50 Hidden Units:\n')
x_example, y_example = XTrain[42], yTrain[42]
f.write("x:\n%s\n" % (" ".join([index_to_char[x] for x in x_example])))
f.write("\ny:\n%s\n" % (" ".join([index_to_char[x] for x in y_example])))

for i in range(len(strings1)):
    print("index_to_word>")
    print('%s' % " ".join([index_to_char[x] for x in strings1[i]]))
    f.write('epoch %d : ' % ep[i])
    f.write('%s\n' % " ".join([index_to_char[x] for x in strings1[i]]))
loss = []
epoch = []
for i in range(len(losses1)):
    loss.append(losses1[i][1])
    epoch.append(i)
# Plot the epoch.loss
rnn.lossvsepoch(epoch, loss, '50Units')
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ========================rnn vanilla 200 hidden units==============================
model = rnn.RNNVanilla(len(index_to_char), hidden_dim=200)
losses2, strings2 = rnn.train_with_sgd(model, XTrain, yTrain, index_to_char, learning_rate=0.005,
                            nepoch=101, evaluate_loss_after=1)

f.write('rnnVaninlla 200 Hidden Units:\n')
x_example, y_example = XTrain[42], yTrain[42]
f.write("x:\n%s\n" % (" ".join([index_to_char[x] for x in x_example])))
f.write("\ny:\n%s\n" % (" ".join([index_to_char[x] for x in y_example])))

for i in range(len(strings2)):
    print("index_to_word>")
    print('%s' % " ".join([index_to_char[x] for x in strings2[i]]))
    f.write('epoch %d : ' % ep[i])
    f.write('%s\n' % " ".join([index_to_char[x] for x in strings2[i]]))
loss = []
epoch = []
for i in range(len(losses2)):
    loss.append(losses2[i][1])
    epoch.append(i)
# Plot the epoch.loss
rnn.lossvsepoch(epoch, loss, '200Units')
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Sequences of length 50
toke_n = []
for i in range(len(toke_sent)):
    if len(toke_sent[i]) == 25:
        toke_n.append(toke_sent[i])
# Create the training data
XTrain = np.asarray([[char_to_index[w] for w in sent[:-1]] for sent in toke_n])
yTrain = np.asarray([[char_to_index[w] for w in sent[1:]] for sent in toke_n])
# ========================rnn vanilla 100 hidden layers/half sequence=================
model = rnn.RNNVanilla(len(index_to_char))
# Limit to 1000 examples to save time
f.write("Expected Loss for random predictions: %f\n" % np.log(len(index_to_char)))
f.write("Actual loss: %f\n" % model.calculate_loss(XTrain[:1000], yTrain[:1000]))
# Train on a small subset of the data to see what happens
losses3, strings3 = rnn.train_with_sgd(model, XTrain, yTrain, index_to_char, learning_rate=0.005,
                            nepoch=101, evaluate_loss_after=1)

f.write('rnnVaninlla Half Sequence:\n')
ep = [20, 40, 60, 80, 100]
x_example, y_example = XTrain[42], yTrain[42]
f.write("x:\n%s\n" % (" ".join([index_to_char[x] for x in x_example])))
f.write("\ny:\n%s\n" % (" ".join([index_to_char[x] for x in y_example])))

for i in range(len(strings3)):
    print("index_to_word>")
    print('%s' % " ".join([index_to_char[x] for x in strings3[i]]))
    f.write('epoch %d : ' % ep[i])
    f.write('%s\n' % " ".join([index_to_char[x] for x in strings3[i]]))
loss = []
epoch = []
for i in range(len(losses3)):
    loss.append(losses3[i][1])
    epoch.append(i)
# Plot the epoch.loss
rnn.lossvsepoch(epoch, loss, 'Halfseq')
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
model = rnn.RNNVanilla(len(index_to_char))
# Limit to 1000 examples to save time
f.write("Expected Loss for random predictions: %f\n" % np.log(len(index_to_char)))
f.write("Actual loss: %f\n" % model.calculate_loss(XTrain[:1000], yTrain[:1000]))
# Train on a small subset of the data to see what happens
losses4, strings4 = rnn.train_with_sgd(model, XTrain, yTrain, index_to_char, learning_rate=0.005,
                            nepoch=101, evaluate_loss_after=1)

f.write('rnnVaninlla Half Sequence:\n')
ep = [20, 40, 60, 80, 100]
x_example, y_example = XTrain[42], yTrain[42]
f.write("x:\n%s\n" % (" ".join([index_to_char[x] for x in x_example])))
f.write("\ny:\n%s\n" % (" ".join([index_to_char[x] for x in y_example])))

for i in range(len(strings3)):
    print("index_to_word>")
    print('%s' % " ".join([index_to_char[x] for x in strings4[i]]))
    f.write('epoch %d : ' % ep[i])
    f.write('%s\n' % " ".join([index_to_char[x] for x in strings4[i]]))
loss = []
epoch = []
for i in range(len(losses4)):
    loss.append(losses4[i][1])
    epoch.append(i)
# Plot the epoch.loss
rnn.lossvsepoch(epoch, loss, 'Doubleseq')
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
f.close()
print('break')