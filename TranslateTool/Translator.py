import time
import hashlib
import json
import urllib
import random
import urllib.request

from .HandleJs import Py4Js


class Translator:
    def __init__(self):
        self.js = Py4Js()
        self.id_baidu = "20151113000005349"
        self.key_baidu = "osubCEzlGjzvw8qdQc41"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 6.1; WOW64; rv:23.0) Gecko/20100101 Firefox/23.0"
        }
        self.url_google = (
            "http://translate.google.com/translate_a/t?client=t"
            + "&sl={src}&tl={dest}&hl={dest}&dt=at&dt=bd&dt=ex&dt=ld&dt=md&dt=qca"
            + "&dt=rw&dt=rm&dt=ss&dt=t&ie=UTF-8&oe=UTF-8&clearbtn=1&otf=1&pc=1"
            + "&srcrom=0&ssel=0&tsel=0&kc=2&tk={tk}&q={content}"
        )
        self.url_baidu = (
            "http://api.fanyi.baidu.com/api/trans/vip/translate"
            + "?appid="
            + self.id_baidu
            + "&q={content}&from={src}&to={dest}&salt={salt}&sign={sign}"
        )

    def open_url(self, url):
        req = urllib.request.Request(url=url, headers=self.headers)
        response = urllib.request.urlopen(req)
        data = response.read().decode("utf-8")
        return data

    def translate_google(self, content, src="en", dest="zh-tw"):
        content_hex = urllib.parse.quote(content)

        url = self.url_google.format(
            src=src, dest=dest, tk=self.js.getTk(content), content=content_hex
        )
        result = self.open_url(url)

        return result[1:-1]

    def translate_baidu(self, content, src="zh", dest="en"):
        salt = random.randint(32768, 65536)
        sign = hashlib.md5(
            (self.id_baidu + content + str(salt) + self.key_baidu).encode()
        ).hexdigest()
        content_hex = urllib.parse.quote(content)

        url = self.url_baidu.format(
            content=content_hex, src=src, dest=dest, salt=salt, sign=sign
        )
        response = urllib.request.urlopen(url)
        json_response = response.read().decode("utf-8")
        js_data = json.loads(json_response)
        # add by stvhuang
        if "error_msg" in js_data:
            print("[ERROR]: ", js_data["error_msg"])
        result = str(js_data["trans_result"][0]["dst"])

        return result


def main():
    tt = Translator()
    while True:
        content = input("輸入待翻譯內容：")
        start = time.time()
        print(tt.translate_google(content))
        print("Translate Cost Time: ", time.time() - start)


if __name__ == "__main__":
    main()
