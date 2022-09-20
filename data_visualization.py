import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


file_loc = "files/"


def set_chinese_font():
    plt.rcParams["font.sans-serif"] = ["STSong"]  # 设置字体
    plt.rcParams["axes.unicode_minus"] = False  # 该语句解决图像中的“-”负号的乱码问题
    plt.rcParams['figure.figsize'] = (19.2, 10.8)
    return


def show_words_bar():
    # 被举报信息中的词数排行柱状图
    df_bow_sorted = pd.read_csv(file_loc + "bow_sorted.csv")

    words = df_bow_sorted.columns.tolist()[1:25]
    sums = df_bow_sorted.iloc[-1].tolist()[1:25]

    set_chinese_font()
    fig, ax = plt.subplots(figsize=(18, 10))
    ax.grid(True, axis='y', alpha=0.25)
    x_pos = np.arange(1, len(words) + 1)
    y_pos = [200, 400, 600, 800, 1000]
    bar = ax.bar(x_pos, np.array(sums), width=0.4, color='#20B2AA', align='center')
    ax.bar_label(bar, labels=sums, fontsize=18)
    ax.set_title("被举报信息中的词数排行\n", fontsize=28)
    ax.set_xticks(x_pos)
    ax.set_yticks(y_pos)
    ax.set_xticklabels(words, rotation=0, fontsize=16)
    ax.set_yticklabels(y_pos, fontsize=18)
    plt.show()
    return


def countries_piechart():
    # 被举报信息中的国家名称占比
    df_bow_sorted = pd.read_csv(file_loc + "bow_sorted.csv")
    countries = [line.strip() for line in open(file_loc + "countries.txt", 'r', encoding='utf-8').readlines()]

    # 取词袋中含有的国家名
    countries = list(set(df_bow_sorted.columns.tolist()) & set(countries))
    countries_df = df_bow_sorted[countries].iloc[-1]
    countries_df.sort_values(ascending=False, inplace=True)

    # 国家名和占比
    countries = countries_df.index.tolist()
    countries_arr = np.array(countries_df)
    
    countries_arr[8] = np.sum(countries_arr[8:])
    countries_arr = countries_arr[:9]
    countries[8] = '其他'
    countries = countries[:9]
    proportions = countries_arr / np.sum(countries_arr) * 100.
    
    #countries_temp = countries.copy()
    #for i in range(12, len(proportions)):
    #    countries_temp[i] = " "
    
    set_chinese_font()
    plt.style.use('seaborn-notebook')
    fig, ax = plt.subplots()
    patches, texts, autotexts = ax.pie(proportions[:],
                                       labels=countries[:],
                                       autopct='%1.1f%%',
                                       pctdistance=0.8,
                                       textprops=dict(color="black", fontsize=14),
                                       startangle=90)
    ax.axis('equal')
    
    # 添加图例，并排序
    labels = ['{0} - {1:1.2f} %'.format(i, j) for i, j in zip(countries[:], proportions[:])]
    sort_legend = True
    if sort_legend:
        patches, labels, dummy = zip(*sorted(zip(patches, labels, countries_arr),
                                             key=lambda x: x[2],
                                             reverse=True))
    ax.legend(patches, labels, loc='best', bbox_to_anchor=(0.5, 0.55, 0.5, 0.5), fontsize=14)
    ax.set_title("\n被举报信息中出现的国家占比", fontsize=26)
    plt.show()
    return




def provincial_piechart():
    # 被举报信息中的省份名称占比
    df_bow_sorted = pd.read_csv(file_loc + "bow_sorted.csv")
    provinces = [line.strip() for line in open(file_loc + "provinces.txt", 'r', encoding='utf-8').readlines()]

    # 取词袋中含有的省份名
    provinces = list(set(df_bow_sorted.columns.tolist()) & set(provinces))
    provinces_df = df_bow_sorted[provinces].iloc[-1]
    provinces_df.sort_values(ascending=False, inplace=True)

    # 省份名和占比
    provinces = provinces_df.index.tolist()
    provinces_arr = np.array(provinces_df)
    
    provinces_arr[8] = np.sum(provinces_arr[8:])
    provinces_arr = provinces_arr[:9]
    provinces[8] = '其他'
    provinces = provinces[:9]
    proportions = provinces_arr / np.sum(provinces_arr) * 100.
    
    provinces_temp = provinces.copy()
    for i in range(24, len(proportions)):
        provinces_temp[i] = " "

    set_chinese_font()
    plt.style.use('ggplot')
    fig, ax = plt.subplots()
    patches, texts, autotexts = ax.pie(proportions[:],
                                       labels=provinces_temp[:],
                                       autopct='%1.1f%%',
                                       pctdistance=0.8,
                                       textprops=dict(color="black", fontsize=14),
                                       startangle=90)
    ax.axis('equal')

    # 添加图例，并排序
    labels = ['{0} - {1:1.2f} %'.format(i, j) for i, j in zip(provinces[:], proportions[:])]
    sort_legend = True
    if sort_legend:
        patches, labels, dummy = zip(*sorted(zip(patches, labels, provinces_arr),
                                             key=lambda x: x[2],
                                             reverse=True))
    ax.legend(patches, labels, loc='best', bbox_to_anchor=(0.5, 0.55, 0.5, 0.5), fontsize=14)
    ax.set_title("\n被举报信息中出现的省级行政区占比", fontsize=26)
    plt.show()
    return


def keyword_linechart(word="疫情"):  # 展示关键词随时间变化的热度
    topic_data = pd.read_csv(file_loc + "topics.csv").drop(['Unnamed: 0'], axis=1)
    topic_data['Time'] = pd.to_datetime(topic_data['Time'])
    topic_data['年月'] = topic_data['Time'].dt.strftime('%Y-%m')
    topic_data.index = topic_data['年月']  # 把年月设置为索引

    year_month = topic_data['年月'].drop_duplicates().tolist()
    counts = []
    for ym in year_month:
        count = 0
        for kw in topic_data.loc[ym]['关键词']:
            if word in kw:
                count += 1
        counts.append(count)  # 每个月含有此关键词的数量

    for i in range(len(year_month)):
        if i % 4 != 0:
            year_month[i] = " "
        
    x = np.linspace(1, 50, 50)
    y = np.array(counts)
    rate = 100/np.max(y)
    y = y*rate
    y_pos = np.arange(0, 101, 25)
    plt.style.use('default')
    set_chinese_font()
    
    fig, ax = plt.subplots()
    ax.plot(x, y, color='#20B2AA', linewidth=1.5, markersize=12)

    title = '关键词“' + word + '”的微博不实信息的热度变化'
    ax.set_title(title, fontsize=28)
    ax.set_xlabel("月份", loc='right', fontsize=16)
    
    ax.set_xticks(x)
    ax.set_xticklabels(year_month, rotation=40, fontsize=14, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_pos, fontsize=14, alpha=0.7)
    
    ax.grid(True, axis='y', alpha=0.25)
    plt.show()
    return


if __name__ == '__main__':
    show_words_bar()
    countries_piechart()
    provincial_piechart()
    keyword_linechart(word="美国")
