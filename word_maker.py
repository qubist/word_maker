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
tokenizer = RegexpTokenizer('\w+')
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

# using default dictionary so that default count is 0
# meaning I can just add without having to have a condition
# to check whether item isn't in the dict and add it
unigrams = defaultdict(lambda: 0)

for letter in letters:
    unigrams[letter] += 1

def display_unigrams(unigrams):
    for unigram, count in unigrams.items():
        print(f'{unigram}\t{count}')

# display_unigrams(unigrams)

bigrams = defaultdict(lambda: 0)

for word in bounded_words:
    for i, token in enumerate(word): # enumerate to have index of current token
        # if it's not the last token
        if token != '#':
            # increment the bigram of this token and the next one in
            # the bigrams dictionary
            bigram = f'{word[i]} {word[i+1]}'
            bigrams[bigram] += 1

# takes a bigram in the form '{token} {token}'
# (string, tokens separated by a space)
# returns that bigram followed by its probability
# based on the corpus
def bigram_prob(bigram):
    # get the first word from the bigram
    first_word = bigram.split()[0]
    first_word_count = unigrams[first_word]
    bigram_count = bigrams[bigram]
    # Equation of bigram probability (3.11 in J&M)
    return (bigram_count / first_word_count)

bigrams_with_probs = {bigram : bigram_prob(bigram) for bigram in bigrams}

# print(bigrams_with_probs)

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
    ngrams = generate_ngrams(n)
    print(ngrams)
    nminus1grams = generate_ngrams(n-1)
    # takes an ngram in the form of an n-tuple
    # returns that bigram followed by its probability
    # based on the corpus
    def ngram_prob(ngram):
        print(f'getting bigram probability for: {ngram}')
        # get the first n-1 words from the ngram
        first_words = ngram[0:n-1]
        first_words_count = nminus1grams[first_words]
        ngram_count = ngrams[ngram]
        # Equation of bigram probability (3.11 in J&M)
        return (ngram_count / first_words_count)

    return {ngram : ngram_prob(ngram) for ngram in ngrams}

def generate_next_letter(last_letter):
    # get all bigrams starting with last letter along with their probabilities
    options = [(bigram, prob) for bigram, prob in bigrams_with_probs.items() if bigram[0] == last_letter]
    # return a random one weighted by its probability
    return random.choices([b[-1] for (b,p) in options], [p for (b,p) in options], k = 1)[0]

# recursive function to make a word.
def make_word(word_so_far):
    last_letter = word_so_far[-1]
    if last_letter == '#':
        # Ding! Word is done!
        return word_so_far
    else:
        return make_word(word_so_far + generate_next_letter(last_letter))

def prettify(word):
    # remove trailing and leading #s and $
    return word.strip("$#")

tests = 50
print(f'Generating {tests} test words from file "{filename}":')
for i in range(tests):
    print(prettify(make_word('$')))

# TODO: Extend it to compare bigram and trigram model, perhaps generalize it to n-gram
