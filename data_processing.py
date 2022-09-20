import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import jieba
import emoji
from gensim import corpora, models
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

file_loc = "files/"


def data_cleaning():
    global data
    data_temp = pd.read_csv(file_loc + "data.csv")
    data_temp.columns = ['Time', 'Name', 'Text']
    time_list = []
    for t in data_temp['Time']:
        t = t.replace("'", '').replace("[", '').replace("]", '')
        if t != '':
            time_list.append(t)
        else:
            time_list.append(np.NaN)  # 对于无时间信息的数据，填入NaN
    data_temp['Time'] = pd.DataFrame(time_list)

    data = data_temp.dropna(axis=0).drop_duplicates(subset='Text')  # 清洗含缺失值、重复的行
    data2 = data_temp.dropna(axis=0)

    data['Time'] = pd.to_datetime(data['Time'])
    data = data.sort_values(by='Time')
    return data2


def get_stopwords():
    stopwords = [line.strip() for line in open(file_loc + "stopwords.txt", 'r', encoding='utf-8').readlines()]
    return stopwords


def clean_list(list1):  # 清除list中的非中文字符串
    def contains_chinese(strs):  # 检测字符串是否包含中文
        for _char in strs:
            if '\u4e00' <= _char <= '\u9fa5':
                return True
        return False

    to_clean = get_stopwords()
    list2 = list1.copy()
    for i in to_clean:  # 先清除停用词
        while list2.count(i) > 0:
            list2.remove(i)

    for i in set(list2):
        while list2.count(i) > 0 and (not contains_chinese(i)):
            list2.remove(i)  # 去除非中文字符串

    return list2


def get_data_features():  # 获取特征，并写入数据集用于训练、分类
    data1 = data_cleaning()
    data1 = data1.drop(columns=['Time', 'Name'])
    data1['category'] = [1] * len(data1['Text'])  # 不实信息标记为1

    nd = [line.strip() for line in open(file_loc + "normal_weibo.txt", 'r', encoding='utf-8').readlines()]
    data2 = pd.DataFrame(nd)
    data2.columns = ['Text']
    data2['category'] = [2] * len(data2)  # 正常微博标记为2

    data3 = pd.concat([data1, data2])

    lengths = []  # 微博文本长度
    has_http = []  # 是否含有超链接
    has_emoji = []  # 是否含有表情
    has_mentioned = []  # 是否@他人
    has_question = []  # 是否含疑问句

    for text in data3['Text'].tolist():
        lengths.append(len(text))
        has_http.append(int('http' in text))
        has_emoji.append(int(emoji.emoji_count(text) > 0))
        has_mentioned.append(int('@' in text))
        has_question.append(int(('？' in text) or ('?' in text)))

    data3['length'] = lengths
    data3['contains_link'] = has_http
    data3['contains_emoji'] = has_emoji
    data3['contains_question'] = has_question
    data3['mentioned_others'] = has_mentioned

    texts = data3['Text'].tolist()
    corpus = []
    for text in texts:
        cleaned_text = clean_list(jieba.lcut(text))
        corpus.append(' '.join(cleaned_text))
    tfidf = TfidfVectorizer(token_pattern=r"(?u)\b\w\w+\b")
    weight = tfidf.fit_transform(corpus).toarray()
    pca = PCA(n_components=2)
    weight_pca = pca.fit(weight).transform(weight)

    features = data3.drop(columns=['Text', 'category'])
    features_pca = pca.fit(features).transform(features)

    data3[['pca1', 'pca2']] = features_pca
    data3[['pca3', 'pca4']] = weight_pca  # 把降维后的数据加入到数据集内
    data3.to_csv(file_loc + "training_dataset.csv")
    return


def write_bag_of_words():
    # 往csv文件写入词袋矩阵
    texts = data['Text'].tolist()
    corpus = []
    for text in texts:
        cleaned_text = clean_list(jieba.lcut(text))
        corpus.append(' '.join(cleaned_text))

    # 向量化
    vectorizer = CountVectorizer()
    feature_words = vectorizer.get_feature_names()
    bow = vectorizer.fit_transform(corpus).toarray()  # 词袋矩阵

    # TF-IDF 词频-逆文本频率
    tfidf = TfidfVectorizer(token_pattern=r"(?u)\b\w\w+\b", stop_words=get_stopwords())
    weight = tfidf.fit_transform(corpus).toarray()  # TF_IDF矩阵
    feature_words2 = tfidf.get_feature_names()

    df_bow = pd.DataFrame(columns=feature_words, data=bow)
    df_bow_alt = pd.DataFrame(columns=feature_words2, data=weight)

    to_drop_words = []
    for word in feature_words:
        if df_bow[word].sum() < 10:  # 去除出现次数较少的词
            to_drop_words.append(word)

    df_bow.drop(columns=to_drop_words, axis=1, inplace=True)
    df_bow_alt.drop(columns=to_drop_words, axis=1, inplace=True)

    df_bow.to_csv(file_loc + "bow.csv")
    df_bow_alt.to_csv(file_loc + "bow_TFIDF.csv")

    return


def sort_word_num():  # 从高到低排序词袋中的单词总次数
    stopwords = get_stopwords()
    df_bow = pd.read_csv(file_loc + "bow.csv").drop(['Unnamed: 0'], axis=1)
    # df_bow_2 = df_bow.drop(['Time'], axis=1)
    sum_list = []
    word_list = df_bow.columns.tolist()

    for word in word_list:
        sum_list.append(df_bow[word].iloc[:].sum())  # 该单词的总出现次数
    df_bow.loc['sum'] = sum_list
    df_bow.sort_values(by='sum', axis=1, ascending=False, inplace=True)  # 按照每列的和的大小排序

    to_drop_list = list(set(stopwords) & set(df_bow.columns.tolist()))
    df_bow.drop(columns=to_drop_list, axis=1, inplace=True)
    df_bow.to_csv(file_loc + "bow_sorted.csv")

    return


def PCA_result():  # PCA 主成分分析

    plt.rcParams['font.sans-serif'] = ['STSong']
    plt.rcParams['figure.figsize'] = (12.8, 7.2)
    df = pd.read_csv(file_loc + "bow_TFIDF.csv").drop(['Unnamed: 0'], axis=1)
    mat_tfidf = np.array(df).T

    pca = PCA(n_components=2)
    x_r = pca.fit(mat_tfidf).transform(mat_tfidf)

    fig, ax = plt.subplots(figsize=(18, 10))
    colors = np.arange(0, len(x_r[:, 1]))
    ax.scatter(x_r[:, 0], x_r[:, 1], c=colors, alpha=0.8, lw=2)
    ax.set_title("PCA降维结果")
    plt.show()
    return


def get_LDA_model():  # 生成LDA主题模型
    texts = data['Text'].tolist()
    corpus = []
    for text in texts:
        cleaned_text = clean_list(jieba.lcut(text))
        corpus.append(cleaned_text)

    global dictionary
    global lda
    dictionary = corpora.Dictionary(corpus)
    corpus2 = [dictionary.doc2bow(words) for words in corpus]  # 将向量放入列表

    num_topics = 5
    num_words = 10
    lda = models.ldamodel.LdaModel(corpus=corpus2, id2word=dictionary, num_topics=num_topics, passes=100)
    topics = lda.print_topics(num_topics=num_topics, num_words=num_words)  # LDA生成的所有主题
    inference = np.array(lda.inference(corpus2)[0])  # 用于主题推断的矩阵

    keywords_list = []  # 提取每个主题的关键词
    for t in range(num_topics):
        keywords = []
        for i in range(num_words):
            keywords.append(dictionary[lda.get_topic_terms(topicid=t)[i][0]])
        keywords_list.append(keywords)

    data_keywords = []  # 写入每条数据的关键词
    data_topic = []
    for i, text in enumerate(data['Text']):
        inference_max = np.where(inference[i] == np.max(inference[i]))[0]  # 求出每条数据所推测的主题号

        if inference_max.size > 1:
            data_keywords.append("无")  # 多个值相同时，无主题
            data_topic.append("None")
        else:
            cleaned_text = clean_list(jieba.lcut(text))
            inter_kw = list(set(cleaned_text) & set(keywords_list[inference_max[0]]))  # 关键词交集
            if inter_kw != []:
                data_keywords.append(' '.join(inter_kw))  # 写出文本中的关键词
                data_topic.append(inference_max[0])
            else:
                data_keywords.append("无")
                data_topic.append("None")
    data["关键词"] = data_keywords
    data["LDA主题类别"] = data_topic
    data[["Time", "关键词", "LDA主题类别"]].to_csv(file_loc + "topics.csv")

    return


def show_LDA_result(to_search):  # LDA主题模型可视化
    index_by_word = []
    for word in to_search:
        for item in lda.get_term_topics(dictionary.doc2idx([word])[0]):
            index_by_word.append(item[0])

    plt.rcParams['font.sans-serif'] = ['STSong']
    plt.rcParams['figure.figsize'] = (12.8, 7.2)

    num_words = 10  # 每个主题下显示的词数量
    flag = 0
    plot_i = 0

    for i in index_by_word:
        plot_i += 1
        ax = plt.subplot(2, int(len(index_by_word) / 2) + 1, plot_i)
        words_prob = lda.get_topic_terms(topicid=i)  # get_topic_terms 返回指定主题的重要词汇
        words_show = np.array(words_prob[:num_words])
        word_id = words_show[:, 0].astype(np.int)
        words = [dictionary.id2token[i] for i in word_id]

        x_pos = np.arange(0, len(words))
        ym = np.max(words_show[:, 1])
        y_pos = np.round(np.arange(ym * 0.3, ym * 1.3, ym * 0.3), 2)

        bar = ax.bar(range(num_words), words_show[:, 1], width=0.4, color='#20B2AA', align='center')
        ax.bar_label(bar, labels=np.round(words_show[:, 1], 3), fmt='%.2f', fontsize=12)
        ax.set_xticks(x_pos)
        ax.set_yticks(y_pos)
        ax.set_xticklabels(words, rotation=0, fontsize=14)
        ax.set_yticklabels(y_pos, fontsize=14)
        if flag == 0:
            ax.set_ylabel("主题内的词概率", fontsize=14)
            flag = 1
        ax.grid(True, alpha=0.25)

    plt.suptitle("\nLDA模型主题分类结果", fontsize=18)
    plt.show()

    return


if __name__ == '__main__':
    data_cleaning()
    write_bag_of_words()
    sort_word_num()
    PCA_result()
    get_LDA_model()
    show_LDA_result(['疫情', '武汉'])
    get_data_features()

