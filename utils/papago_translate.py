import os
import sys
import urllib.request
import json


client_id = ""  # 개발자센터에서 발급받은 Client ID 값
client_secret = ""  # 개발자센터에서 발급받은 Client Secret 값


"""
영어로 1차 번역
"""
encText = urllib.parse.quote("아시아나여객기, 문 열린채 대구공항 착륙…승객들 공포(종합)")
data = "source=ko&target=en&text=" + encText
url = "https://openapi.naver.com/v1/papago/n2mt"
request = urllib.request.Request(url)
request.add_header("X-Naver-Client-Id", client_id)
request.add_header("X-Naver-Client-Secret", client_secret)
response = urllib.request.urlopen(request, data=data.encode("utf-8"))
rescode = response.getcode()

en_result = ""
if rescode == 200:
    response_body = response.read()
    en_result = json.loads(response_body.decode("utf-8"))["message"]["result"][
        "translatedText"
    ]
else:
    print("Error Code:" + rescode)


"""
영어로 번역된 것을 다시 한국어로 재번역
"""
data2 = "source=en&target=ko&text=" + en_result
url = "https://openapi.naver.com/v1/papago/n2mt"
request = urllib.request.Request(url)
request.add_header("X-Naver-Client-Id", client_id)
request.add_header("X-Naver-Client-Secret", client_secret)
response = urllib.request.urlopen(request, data=data2.encode("utf-8"))
rescode = response.getcode()


result = ""
if rescode == 200:
    response_body = response.read()
    result = json.loads(response_body.decode("utf-8"))["message"]["result"][
        "translatedText"
    ]
else:
    print("Error Code:" + rescode)


print("*" * 100)
print(result)
print("*" * 100)
