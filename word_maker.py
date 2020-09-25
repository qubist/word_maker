# Some code copied from my submission for Lab2
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from nltk.tokenize import *
from nltk.util import bigrams, trigrams
import codecs
from collections import defaultdict
import random
filename = 'illiad.txt'
f = codecs.open(filename, encoding='utf8')

lines = f.readlines()
all_text = ' '.join(lines).lower()

# tokenize each sentence and only include letters (regex word chacacters)
tokenizer = RegexpTokenizer('[a-z]+')
tokens = tokenizer.tokenize(all_text)

# print(tokens)

# add sentence boundaries to tokenized sentences
bounded_words = ['$' + token + '###############' for token in tokens]
# print(bounded_words)

# set containing every letter in the corpus, will be useful later
letters = set([])
for word in bounded_words:
    for letter in word:
        letters.add(letter)

# list containing every letter in the corpus, will be useful later
letters = []
for word in bounded_words:
    for letter in word:
        letters += letter

def generate_ngrams(n):
    ngrams = defaultdict(lambda: 0)

    for word in bounded_words:
        for i, token in enumerate(word): # enumerate to have index of current token
            # if it's not the last token
            if token != '#':
                # increment this ngram in the ngrams dictionary
                ngram = []
                for j in range(n):
                    # add each letter n times to make the ngram
                    ngram += [word[i+j]]
                ngram = tuple(ngram)
                ngrams[ngram] += 1
    return ngrams

# returns a dictionary of ngrams of size n
# with probabilities
def generate_prob_dict(n):
    print('PROB DICT BEING GENERATED!')
    ngrams = generate_ngrams(n)
    nminus1grams = generate_ngrams(n-1)
    # takes an ngram in the form of an n-tuple
    # returns that bigram followed by its probability
    # based on the corpus
    def ngram_prob(ngram):
        # get the first n-1 words from the ngram
        first_words = ngram[0:n-1]
        first_words_count = nminus1grams[first_words]
        ngram_count = ngrams[ngram]
        # Equation of bigram probability (3.11 in J&M)
        return (ngram_count / first_words_count)

    return {ngram : ngram_prob(ngram) for ngram in ngrams}

def starting_ngram(n, ngrams):
    options = [(ngram, prob) for ngram, prob in ngrams.items() if ngram[0] == '$']
    ngram = random.choices([n for (n,p) in options], [p for (n,p) in options], k = 1)[0]
    return ''.join(ngram)

def generate_next_letter(last_nminus1_letters, ngrams, n):
    # get all bigrams starting with last n-1 letters along with their probabilities
    options = [(ngram, prob) for ngram, prob in ngrams.items() if ngram[:n-1] == tuple(last_nminus1_letters)]
    # return a random one weighted by its probability
    return random.choices([b[-1] for (b,p) in options], [p for (b,p) in options], k = 1)[0]

# recursive function to make a word.
def make_word(n, word_so_far = '', ngrams = None):
    # detect if function is being called without passed-along ngrams dict,
    # and make one
    if ngrams is None:
        return make_word(n, word_so_far, generate_prob_dict(n)) # trying to only generate a dict once per word, max
    # detect if function is being called without starting letters and start
    # it off for us, using a random starting ngram's letters
    if word_so_far == '':
        return make_word(n, starting_ngram(n, ngrams), ngrams)

    last_nminus1_letters = word_so_far[-(n-1):]
    # if last letter added was EOW #
    if word_so_far[-1] == '#':
        # Ding! Word is done!
        return word_so_far
    else:
        improved_word = word_so_far + generate_next_letter(last_nminus1_letters, ngrams, n)
        return make_word(n, improved_word, ngrams)

def prettify(word):
    # remove trailing and leading #s and $
    return word.strip("$#")

if __name__ == '__main__':
    tests = 20
    n = 3
    # generate prob dict only once and pass it into each word generation
    ngrams = generate_prob_dict(n)
    print([starting_ngram(n, ngrams) for i in range(100)])
    print(f'Generating {tests} test words from file "{filename}" using {n}-gram model:')
    for i in range(tests):
        print(prettify(make_word(n, ngrams = ngrams)))
