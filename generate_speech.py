import random
import sys
import numpy as np
import os
from keras.models import load_model


def sample(preds, temperature=1.0):
    """Perform Temperature Sampling"""
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    # Softmax of predictions
    preds = exp_preds / np.sum(exp_preds)
    # Sample a single characters, with probabilities defined in `preds`
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


corpora_dir = "trump_speeches-master/data"

# Read all file paths in corpora directory
file_list = []
for root, _, files in os.walk(corpora_dir):
    for filename in files:
        file_list.append(os.path.join(root, filename))

print("Read ", len(file_list), " files...")

# Extract text from all documents
docs = []

for files in file_list:
    with open(files, 'r') as fin:
        try:
            str_form = fin.read().lower().replace('\n', '')
            docs.append(str_form)
        except UnicodeDecodeError:
            # Some sentences have wierd characters. Ignore them for now
            pass
# Combine them all into a string of text
text = ' '.join(docs)


chars = sorted(list(set(text)))
print('Total Number of Unique Characters:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars)) # Character to index
indices_char = dict((i, c) for i, c in enumerate(chars)) # Index to Character




maxlen = 51  # Number of characters considered
step = 3  # Stide of our window
sentences = []
next_chars = []

"""Function invoked at end of each epoch. Prints generated text"""
print()

start_index = random.randint(0, len(text) - maxlen - 1)
for diversity in [0.2, 0.5, 1.0, 1.2]:
    print('----- Diversity:', diversity)

generated = ''
sentence = text[start_index: start_index + maxlen]
# sentence = 'they are bringing drugs. they are bringing crime. t'
generated += sentence
print('----- Generating with seed: "' + sentence + '"')
sys.stdout.write(generated)

model = load_model('trump_speech.h5')


for i in range(400):
    x_pred = np.zeros((1, maxlen, len(chars)))
    for t, char in enumerate(sentence):
        x_pred[0, t, char_indices[char]] = 1.

    preds = model.predict(x_pred, verbose=0)[0]
    # Generate next character
    next_index = sample(preds, diversity)
    next_char = indices_char[next_index]

    # Append character to generated sequence
    generated += next_char
    sentence = sentence[1:] + next_char

    sys.stdout.write(next_char)
    sys.stdout.flush()
print()