"""
Andy Qin
CSE 163 AB

This is the processor module, which takes a directory of blog xml files,
cleans it into a dataframe, applies natural language processing on each
entry's text, and then exports it for further analysis in the analyzer
module.

Running main will process the data in the given directory, 'directory', and
export the dataframe as 'cleaned_data.pickle' and print the dataframe.
Note that large directories may take a substatial amount of time to process.
"""

import time
import os
import spacy
import pandas as pd
from lxml import html
import pickle
from collections import Counter


def xmls_to_df(directory):
    """
    Takes a directory of xml blog files and processes them into a pandas
    dataframe indexed by their unique ids, with each entry representing a
    single author's blog.
    The dataframe contains information for the blog author's gender, age,
    self-assigned category, number of posts, and a single field for the text
    of all their posts (blogpost dates are stripped).'
    """
    t1 = time.time()
    data_dict = {'id': [], 'gender': [], 'age': [], 'category': [],
                 'post count': [], 'text': []}
    for f in os.listdir(directory):
        info = f.split('.')
        data_dict['id'].append(int(info[0]))
        data_dict['gender'].append(info[1])
        data_dict['age'].append(int(info[2]))
        data_dict['category'].append(info[3])

        root = html.parse(directory + f).getroot()
        posts = root.findall('.//post')
        text = ''
        for post in posts:
            text += post.text.strip() + ' '
        data_dict['post count'].append(len(posts))
        data_dict['text'].append(text)

    df = pd.DataFrame(data_dict)
    df = df.set_index('id')
    t2 = time.time()
    print('Reading XMLs Time: ' + str(t2 - t1))
    return df


def basic_info(df):
    """
    Takes a pandas dataframe of processed xml blogs and adds basic info
    about word counts, character counts, and per post averages.
    Modifies the original dataframe.
    """
    t1 = time.time()
    df['char count'] = df['text'].str.len()
    df['word count'] = df['text'].str.split().str.len()
    df['avg words'] = df['word count'] / df['post count']
    t2 = time.time()
    print('Process Basic Time: ' + str(t2 - t1))
    return df


def add_lemma_pos_info(df):
    """
    Takes a pandas dataframe and applies natural language processing to the
    body of each consolidated blog to add new columns for lemma counts and
    part of speech ratios. Removes original text after analysis is complete.
    Returns the modified dataframe.
    """
    nlp = spacy.load('en_core_web_sm')
    df['lemma counts'], df['pos counts'] = zip(*df['text'].apply(count_lang,
                                                                 args=(nlp,)))
    export_pickle(df)
    df['nounr'], df['propnr'], df['verbr'], df['adjr'], df['advr'] =\
        zip(*df['pos counts'].apply(unpack_pos_freq))
    df = df.drop(['text'], axis=1)
    return df


def count_lang(text, nlp):
    """
    Takes a string body of text applies natural language processing to
    remove all stop words, whitespace, and puncutation, and then count all
    unique lemmas and major parts of speech (adjectives, nouns, proper nouns,
                                             verbs, and adverbs)
    Returns a tuple of dicts of processed token and part of speech counts
    """
    t1 = time.time()
    doc = nlp(text)
    lemmas = []
    pos = []
    for token in doc:
        if (token.is_stop is False
                and token.is_punct is False
                and token.is_space is False
                and token.like_num is False
                and (token.text in nlp.vocab)
                and len(token) > 2):
            lemmas.append(token.lemma_.lower())
            pos.append(token.pos_)
    lemma_freq = Counter(lemmas)
    pos_freq = Counter(pos)
    t2 = time.time()
    print('Process Language Time: ' + str(t2 - t1))
    return (lemma_freq, pos_freq)


def unpack_pos_freq(c):
    """
    Takes a blog's part of speech counts and packs their values as
    frequency ratios for seperate unpacking into different columns.
    Returns a tuple of part of speech ratios based on the given counts
    """
    total = sum(c.values())
    if total == 0:
        total = 1
    values = (c['NOUN'], c['PROPN'], c['VERB'], c['ADJ'], c['ADV'])
    freqs = tuple(v / total for v in values)
    return (freqs)


def export_pickle(df):
    """
    Takes a pandas dataframe and exports it as a pickle file for later use
    """
    with open('cleaned_data.pickle', 'wb') as f:
        pickle.dump(df, f)


def main():
    t1 = time.time()
    directory = 'data\\two_hundred_test\\'
    df = xmls_to_df(directory)
    df = basic_info(df)
    df = add_lemma_pos_info(df)
    print(df)
    export_pickle(df)

    t2 = time.time()
    print('Done! Data saved as cleaned_data.pickle')
    print('Total Time: ' + str(t2 - t1))


if __name__ == '__main__':
    main()