"""
Andy Qin
CSE 163 AB

This is the analyzer module, which takes a cleaned dataframe of blog
information and applies various analyses and visualization of the blog data,
including the creation of polarized gender word frequency lists.
It also contains functionality to train a machine learning algorithm to
determine a blog author's gender.

Running the main function will save a variety of data visualizations as png
plots as well as the creation of polarized word frequency scores which are
used to train a machine learning model, which is immediately tested.
"""
import pandas as pd
import pickle
from collections import Counter

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def import_pickle(file_path):
    """
    Takes a cleaned pickle file of processed blogs and unpacks it, returning
    it as a pandas dataframe.
    """
    with open(file_path, 'rb') as f:
        return(pickle.load(f))


def drop_words(c, to_drop):
    """
    Takes a Counter and a list of text tuples and removes all occurrances of
    the text tuple from the Counter
    Returns the modified Counter
    """
    for word in to_drop:
        if word[0] in c:
            c.pop(word[0])
    return c


def tuple_list_to_df(t_list):
    """
    Takes a list of tuples (from a Counter object) and converts it into a
    pandas dataframe, returning the dataframe
    """
    temp = {}
    for tup in t_list[:10]:
        word, count = tup
        temp[word] = [count]
    df = pd.DataFrame.from_dict(temp)
    return df


def group_gender_counts(df):
    """
    Takes a pandas dataframe of blogs and groups the lemma counts by gender,
    totalling all word counts from all blogs of each gender.
    Returns a new dataframe with consolidated counts and no other columns.
    """
    df = df.groupby('gender')['lemma counts'].sum()
    return df


def get_word_freq(grouped_df, group_name):
    """
    Takes a consolidated dataframe with Counters that are grouped and a group
    name and calculates the frequency of the most common word in
    comparison the total number of words in the counter object, multiplied by
    1000.
    Returns a new Counter with the words as keys and frequencies as values.
    """
    word_total = sum(grouped_df[group_name].values(), 0.0)
    word_freq = Counter(dict(grouped_df[group_name].most_common(5000)))
    for word in word_freq:
        word_freq[word] = 1000 * word_freq[word] / word_total
    return word_freq


def freq_score(counts, score_list):
    """
    Takes a Counter of words in a blog and a score list of word frequncies
    and calcultes a weighted score for the similarity based on the number of
    matching words both lists multiplied by their frequency.
    """
    score = 0
    for word in score_list:
        if word[0] in counts:
            score += word[1] * counts[word[0]]
    return score


def gender_word_score(grouped_df, raw_df):
    """
    Takes a pandas dataframe and the same dataframe grouped by gender with
    the lemma counts summed and computes word frequencies for each gender.
    Uses these word frequnces to calculate a 'lang score' for each blog
    entry in the main dataframe that indicates possible gender based on
    commonly used words and their frequency of the blogs.
    Returns a tuple of word frequency dataframes for males and females.
    Modifies the original dataframe.
    """
    m_freq = get_word_freq(grouped_df, 'male')
    f_freq = get_word_freq(grouped_df, 'female')
    total = len(m_freq + f_freq)
    common_words = (f_freq + m_freq).most_common(int(total / 25))

    f_freq = drop_words(f_freq, common_words)
    m_freq = drop_words(m_freq, common_words)
    f_common = (f_freq - m_freq).most_common(200)
    m_common = (m_freq - f_freq).most_common(200)

    raw_df['mlang score'] =\
        raw_df['lemma counts'].apply(freq_score, args=[m_common])
    raw_df['flang score'] =\
        raw_df['lemma counts'].apply(freq_score, args=[f_common])
    raw_df['lang score'] = raw_df['mlang score'] - raw_df['flang score']

    m_df = tuple_list_to_df(m_common)
    f_df = tuple_list_to_df(f_common)

    return (m_df, f_df)


def visualize_word_freq(m_df, f_df):
    """
    Takes two pandas dataframes of relative word frequencies for male and
    females and visualizes their most frequent words, excluding words that
    are found commonly in both male and females blogs.
    Saves the figure as 'word_results.png'
    """
    sns.set(font_scale=1.1)
    fig, [ax1, ax2] = plt.subplots(2, figsize=(14, 12))
    sns.catplot(kind='bar', color='b', ax=ax1, data=m_df)
    sns.catplot(kind='bar', color='r', ax=ax2, data=f_df)
    ax1.set_title('Polarizing Male Words')
    ax1.set_ylabel('Relative Frequency per 1000 Words')
    ax2.set_title('Polarizing Female Words')
    ax2.set_ylabel('Relative Frequency per 1000 Words')
    fig.savefig('word_results.png')


def visualize_pos(df):
    """
    Takes a cleaned dataframe of processed blogs and visualizes part of
    speech distribution broken down by gender.
    Saves the figure as 'pos_results.png'
    """
    cols = ['gender', 'nounr', 'propnr', 'verbr', 'adjr', 'advr']
    df = df.groupby('gender')[cols].mean()
    df['gender'] = df.index
    df = df.melt('gender', var_name='pos', value_name='ratio')

    sns.set(font_scale=1.2)
    fig, ax = plt.subplots(1, figsize=(10, 5))
    sns.catplot(kind='bar', ax=ax, x='pos', y='ratio', hue='gender', data=df)
    ax.set_title('Part of Speech Ratios')
    ax.set_ylabel('Ratio')
    ax.set_xlabel('Part of Speech')
    fig.savefig('pos_results.png')


def visualize_categories(df):
    """
    Takes a cleaned dataframe of processed blogs and visualizes the gender
    distribution across different self-dfined blog categories. Cleans up
    category names for presentation as well.
    Saves figure as 'category_results.png'
    """
    df = df.loc[:, ['category', 'gender']]
    df = pd.get_dummies(df, columns=['gender'])
    df = df.groupby('category').sum()
    df = df.drop(['indUnk', 'Student'], axis=0)
    df['category'] = df.index

    df = df.replace(to_replace='Communications-Media', value='Media')
    df = df.replace(to_replace='LawEnforcement-Security', value='Security')
    df = df.replace(to_replace='Museums-Libraries', value='Libraries')
    df = df.replace(to_replace='Sports-Recreation', value='Sports')
    df = df.replace(to_replace='BusinessServices', value='Services')
    df = df.replace(to_replace='Telecommunications', value='Telecoms')
    df = df.replace(to_replace='Transportation', value='Transport')
    df = df.replace(to_replace='InvestmentBanking', value='Investing')

    df = df.rename(columns={'gender_male': 'Male'})
    df = df.rename(columns={'gender_female': 'Female'})

    df = df.melt('category', var_name='gender', value_name='count')
    sns.set()
    fig, ax = plt.subplots(1, figsize=(20, 10))
    sns.catplot(kind='bar', ax=ax, x='category', y='count', hue='gender',
                order=df['category'].value_counts().index, data=df)
    for item in ax.get_xticklabels():
        item.set_rotation(60)
    ax.set_title('Blog Category Counts by Gender, excluding Uknown/Student')
    ax.set_ylabel('Count')
    ax.set_xlabel('Category')
    fig.savefig('category_results.png')


def visualize_averages(df):
    """
    Takes a cleaned dataframe of processed blogs and visualizes average
    characters per word, average words per post, and average posts per blog.
    Saves the 3 plots as 'average_results.png'.
    """
    df['avg chars'] = df['char count'] / df['word count']
    cols = ['gender', 'avg chars', 'avg words', 'post count']
    df = df.loc[:, cols]
    df = df.groupby('gender').mean()
    df['gender'] = df.index

    sns.set(font_scale=1.3)
    fig, [ax1, ax2, ax3] = plt.subplots(3, figsize=(8, 20))
    sns.catplot(kind='bar', ax=ax1, x='gender', y='avg chars', data=df)
    sns.catplot(kind='bar', ax=ax2, x='gender', y='avg words', data=df)
    sns.catplot(kind='bar', ax=ax3, x='gender', y='post count', data=df)

    ax1.set_title('Average Characters per Word')
    ax1.set_ylabel('Characters')
    ax1.set_xlabel('Gender')
    ax1.set(ylim=(3, 7))

    ax2.set_title('Average Words per Post')
    ax2.set_ylabel('Words')
    ax2.set_xlabel('Gender')
    ax2.set(ylim=(200, 325))

    ax3.set_title('Average Posts per Blog')
    ax3.set_ylabel('Posts')
    ax3.set_xlabel('Gender')

    fig.savefig('averages_results.png')


def learn_gender(df):
    """
    Takes a pandas dataframe with precomputed 'lang scores' cand trains
    a machine learning model to predict a blog author's gender.
    Saves an image of the final decision tree, prints out training and test
    model accuracies, and returns the model.
    """
    df = df.drop(['lemma counts', 'pos counts', 'age', 'char count',
                  'word count', 'flang score', 'mlang score'], axis=1)

    features = df.loc[:, df.columns != 'gender']
    labels = df['gender']

    features = pd.get_dummies(features)

    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.3)

    model = DecisionTreeClassifier(max_depth=4)
    model.fit(features_train, labels_train)

    train_predictions = model.predict(features_train)
    print('Train Accuracy:', accuracy_score(labels_train, train_predictions))

    test_predictions = model.predict(features_test)
    print('Test  Accuracy:', accuracy_score(labels_test, test_predictions))

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(15, 5))
    tree.plot_tree(model, filled=True,
                   feature_names=features.columns,
                   class_names=labels.unique(),
                   rounded=True,
                   fontsize=7)
    fig.savefig('decision_tree')
    return model


def main():
    directory = 'cleaned_data.pickle'
    raw_df = import_pickle(directory)


    visualize_pos(raw_df)
    visualize_categories(raw_df)
    visualize_averages(raw_df)

    grouped_df = group_gender_counts(raw_df)
    m_df, f_df = gender_word_score(grouped_df, raw_df)
    visualize_word_freq(m_df, f_df)
    print(raw_df)
    learn_gender(raw_df)


if __name__ == '__main__':
    main()