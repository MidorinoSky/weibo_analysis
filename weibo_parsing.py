from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from time import sleep
from bs4 import BeautifulSoup
from lxml import etree
import re
import csv


file_loc = "files/"


def login_weibo():
    # 使用webdriver登录微博
    global driver
    driver = webdriver.Edge()
    driver.get("https://weibo.com/login.php/")
    sleep(1)
    driver.maximize_window()
    print("输入用户名：")
    login_name = input()
    print("输入密码：")
    password = input()
    sleep(0.5)
    driver.find_element_by_css_selector("#loginname").send_keys(login_name)
    sleep(0.5)
    driver.find_element_by_css_selector(  # 在密码框处传入密码
        "#pl_login_form > div > div:nth-child(3) > div.info_list.password > div > input").send_keys(password)
    sleep(0.5)
    driver.find_element_by_css_selector("#pl_login_form > div > div:nth-child(3) > div.info_list.login_btn > a").click()
    # 点击登录键

    enter = ''
    while enter != 'y':
        print("是否已经扫码(y/n):")
        enter = input()

    driver.get("https://service.account.weibo.com/index?type=5&status=4&page=1")  # 不实信息公示页面
    try:
        element = WebDriverWait(driver, 30).until(
            EC.presence_of_all_elements_located((By.CLASS_NAME, 'm_table_tit'))
        )  # 若class名为m_table_tit的公示表加载完毕，则成功进入
    except Exception as e:
        print(e)
        print("进入失败！")
        return 0
    print("成功进入微博社区管理系统！")
    return


def get_page_source(driver, com_url):
    # 读取一个页面的html
    try:
        driver.get(com_url)
    except Exception as e:
        print(e)
        print("该页面不能读取哦: ", com_url)
        return 0
    sleep(0.3)
    return driver.page_source


def get_detail_text_html(driver, com_url):
    # 获取详细微博文本的html
    try:
        driver.get(com_url)
    except Exception as e:
        print(e)
        print("该页面不能读取哦: ", com_url)
        return 0
    try:
        element = WebDriverWait(driver, 30).until(
            EC.presence_of_all_elements_located((By.CLASS_NAME, 'detail_wbtext_4CRf9'))
        )
    except Exception as e:
        print(e)
        print("读取失败！", com_url)
        return 0
    js = "window.scrollTo(0,20000)"
    driver.execute_script(js)
    sleep(1)
    return driver.page_source


def get_announced_links():  # 提取300页的结果公示，存储被举报信息的链接
    for page_num in range(1, 301):
        page_addr = 'https://service.account.weibo.com/index?type=5&status=4&page=' + str(page_num)
        announcement = get_page_source(driver, page_addr)
        soup_ann = BeautifulSoup(announcement, "lxml")
        soup_tbody = soup_ann.find('tbody')

        linknum = 0
        for link in soup_tbody.find_all('a'):
            linkget = link.get('href')
            linknum += 1
            if linknum % 3 == 0:
                user_link.append(linkget)
            if linkget.find('show?') != -1:
                link_head = 'https://service.account.weibo.com'
                link_addr = link_head + linkget
                link_list.append(link_addr)

    f = open(file_loc + "rumorlinks.txt", 'w', encoding='utf-8', newline='')
    f.write('\n'.join(link_list))
    f.close()
    f = open(file_loc + "userlinks.txt", 'w', encoding='utf-8', newline='')
    f.write('\n'.join(user_link))
    f.close()
    return


def get_link_lists():
    global link_list
    global user_link
    link_list = [line.strip() for line in open(file_loc + "rumorlinks.txt", 'r', encoding='utf-8').readlines()]
    user_link = [line.strip() for line in open(file_loc + "userlinks.txt", 'r', encoding='utf-8').readlines()]
    return


def write_weibo_rumor(first_i, last_i, filename):
    f = open(filename, 'a+', encoding='utf-8', newline='')
    csv_writer = csv.writer(f)
    for i in range(first_i, last_i):
        untrue_source = get_page_source(driver, link_list[i])
        if untrue_source != 0:
            soup_untrue = BeautifulSoup(untrue_source, "lxml")
            divs = soup_untrue.find_all('div', {'class': "con"})  # 截取class为con的内容
            time_source = soup_untrue.find_all('p', {'class': "publisher"})  # 截取class为publisher的内容

            time_pattern = re.compile("\d+-\d+-\d+ \d+:\d+:\d+")
            ptime = time_pattern.findall(time_source[-1].text)  # 根据正则表达式提取发布时间
            text = divs[-1].text.replace('\n', '').replace('\t', '').replace(' ', '').replace('\u200b', '')
            wbname = text.split('：')[0]

            detail_link = ''
            weibo_text = ''

            '''如果有原文则查看原文'''

            for link in soup_untrue.find_all('a', text='原文'):
                detail_link = link.get('href')
            if detail_link != '' and '查看全文' in text:
                detail_html = get_detail_text_html(driver, detail_link)
                wbtree = etree.HTML(detail_html)
                weibo_text = ''.join(wbtree.xpath('.//div[contains(@class, "detail_wbtext_4CRf9")]//text()')).replace(
                    "'", r"\'")
                # class 名为 detail_wbtext_4CRf9 内，包含微博详细文本
                csv_writer.writerow([ptime, wbname, weibo_text])
                print(weibo_text)
                print("------", i)
                continue

            text2 = text.split('：')[0], '：'
            weibo_text = text.replace(''.join(text2), '')
            csv_writer.writerow([ptime, wbname, weibo_text])
        else:
            continue
    f.close()
    return


def get_user_html(driver, com_url):
    # 获取用户的html
    try:
        driver.get(com_url)
    except Exception as e:
        print(e)
        print("该页面不能读取哦: ", com_url)
        return 0
    try:
        element = WebDriverWait(driver, 10).until(
            EC.presence_of_all_elements_located((By.CLASS_NAME, 'detail_wbtext_4CRf9'))
        )
    except Exception as e:
        print(e)
        print("读取失败！", com_url)
        return 0
    js = "window.scrollTo(0,50000)"
    driver.execute_script(js)
    return driver.page_source


def write_normal_weibo():  # 读取6000条正常微博
    normal_texts = []
    for i in range(0, 6000):
        userhtml = get_user_html(driver, user_link[i])
        if userhtml == 0:
            continue
        user_soup = BeautifulSoup(userhtml, "lxml")
        divs = user_soup.find_all('div', {'class': "detail_wbtext_4CRf9"})
        texts = [divs[j].text.replace('\n', '').replace('\t', '').replace('\u200b', '') for j in range(len(divs))]
        normal_texts.extend(texts)
        print("已获取微博条数：", len(normal_texts))
        if len(normal_texts) >= 6000:
            print("已获取6000条正常微博。")
            break
    f = open(file_loc + 'normal_weibo.txt', 'w', encoding='utf-8', newline='')
    f.write('\n'.join(normal_texts))
    f.close()
    return


if __name__ == '__main__':
    login_weibo()
    get_announced_links()
    get_link_lists()
    write_weibo_rumor(0, 6000, filename=file_loc + "data.csv")
    write_normal_weibo()

