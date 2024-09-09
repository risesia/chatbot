import nltk
import Sastrawi
import numpy as np
# nltk.download('punkt')
# nltk.download('punkt_tab')
from nltk.stem.porter import PorterStemmer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

stemmer = PorterStemmer()

indonesian_stemmer = StemmerFactory().create_stemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def indonesian_stem(word):
    return indonesian_stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w, in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0

    return bag

# sentence = ["hello", "how", "are", "you"]
# words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
# bog = bag_of_words(sentence, words)
# print(bog)

# a = "See you later, thanks for visiting!"
# print(a)
# a = tokenize(a)
# print(a)

# words = ["Organize", "Organizes", "Organizing"]
# stemmed_words = [stem(w) for w in words]
# print(stemmed_words)

# b = "Saya adalah manusia yang merakyat, kepada Republik Indonesia."
# b =  tokenize(b)
# print(b)

# kata = ["Berkobar", "Kobaran", "Berkobar-kobar"]
# indonesian_stemmed_words = [indonesian_stem(k) for k in kata]
# print(indonesian_stemmed_words)