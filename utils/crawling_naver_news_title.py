import requests
from bs4 import BeautifulSoup


def get_naver_news_title(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.81 Safari/537.36"
    }

    # 웹 페이지에 요청을 보내고 HTML 문서를 가져옵니다.
    response = requests.get(url, headers=headers)
    html = response.text

    # BeautifulSoup을 사용하여 HTML 문서를 파싱합니다.
    soup = BeautifulSoup(html, "html.parser")

    # 제목을 가져옵니다. 각 웹 페이지의 HTML 구조에 따라 적절한 CSS 선택자를 사용해야 합니다.
    title = soup.select_one("div.media_end_head_title span").text.strip()

    return title


# 네이버 뉴스 기사 URL
news_url = "https://n.news.naver.com/mnews/article/001/0013965529?rc=N&ntype=RANKING"

# 제목을 크롤링합니다.
news_title = get_naver_news_title(news_url)
print(news_title)
