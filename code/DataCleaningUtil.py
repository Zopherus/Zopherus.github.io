from collections import defaultdict
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn

def clean(abstract_list):
    from nltk.corpus import stopwords

    tag_map = defaultdict(lambda : wn.NOUN)
    tag_map['J'] = wn.ADJ
    tag_map['V'] = wn.VERB
    tag_map['R'] = wn.ADV
    stopwords = set(stopwords.words("english"))
    cleaned_abstract_list = []
    for abstract in abstract_list:
        words = word_tokenize(abstract)
        content = []
        word_Lemmatized = WordNetLemmatizer()
        for word, tag in pos_tag(words):
            if word.isalpha() and word not in stopwords:
                word = word_Lemmatized.lemmatize(word, tag_map[tag[0]])
                content.append(word)
        cleaned_abstract_list.append(content)
    return cleaned_abstract_list
