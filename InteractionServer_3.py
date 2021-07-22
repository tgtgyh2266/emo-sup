from concurrent import futures
import logging

import aiml
import zhconv
from hanziconv import HanziConv
import socket
import time
import web
from InteractionManagement import InteractionManagement

class web_server_template:  ##宣告一個class,在下文的web.application實例化時，會根據定義將對應的url連接到這個class
    def __init__(self):  ##初始化類別
        print('initial in {}'.format(time.time()))
    def POST(self):  ##當server收到一個指向這個class URL的POST請求，會觸發class中命名為POST的函數，GET請求同理
        # recive = json.loads(str(web.data(),encoding='utf-8'))  ##使用json.loads將json格式讀取為字典
        # print('[Message] Post message recive:{}'.format(recive))
        # result = True
        # msg = 'Server recive'

        # return_json = {'results':result,'return_message':msg}
        # return_data = json.dumps(return_json,sort_keys=True,separators=(',',':'),ensure_ascii=False) ##打包回傳信息為json

        received_data = web.data()
        received_data = received_data.decode()

        global query_id_interaction_signal
        global first_question_signal
        # global start_interaction_signal
        global emotional_detect_signal
        global during_interaction_signal
        global query_id_chat
        global current_id
        global IM
        global start_time
        if first_question_signal:
            passing_information = "嗨！今天想和我聊些什麼呢？"
            first_question_signal = False

        elif during_interaction_signal:
            # Heartbeat live signal
            if received_data == "Sentence":
                passing_information = "empty"

                # Idle 2 minutes: Reset current_id
                print("Idle Time: ", time.time() - start_time)
                if time.time() - start_time > 120:
                    query_id_interaction_signal = True

            # Return speaking finish
            elif received_data == "Finish":
                passing_information = "empty"

            # Noise cancel
            elif "\uff36\uff2f\uff29\uff23\uff25\uff30\uff26" in received_data:
                passing_information = "抱歉，我沒有聽清楚，麻煩請再說一次!"

            # Feed into the emotional support system
            else:
                print("Received Message: ", received_data)
                start_time = time.time()
                input_sentence = zhconv.convert(received_data, "zh-cn")
                passing_information = IM.emotional_support_system(
                    input_sentence, current_id
                )

                print("System Cost Time: ", time.time() - start_time)
        else:
            passing_information = "empty"

        return passing_information  ##回傳

    def GET(self):
        return 'Hello World!'

    def toTraditional(self, term):
        # Convert traditional Chinese to simplified Chinese.
        return HanziConv.toTraditional(term)

    def get_ip(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            # doesn't even have to be reachable
            s.connect(("10.255.255.255", 1))
            IP = s.getsockname()[0]
        except:
            IP = "127.0.0.1"
        finally:
            s.close()
        return IP


if __name__ == '__main__':
    global query_id_interaction_signal
    global first_question_signal
    # global start_interaction_signal
    global emotional_detect_signal
    global during_interaction_signal
    global query_id_chat
    global current_id
    global IM
    global start_time

    # Initialize
    first_question_signal = True
    query_id_interaction_signal = True
    emotional_detect_signal = False
    video_capture_signal = False
    during_interaction_signal = True

    # Set personality trait of the user
    personality_trait = int(
        input(
            "\n\n(0) Extraversion 外向性\n(1) Openness to experience 經驗開放性\n(2) Emotional stability 情緒不穩定性\n(3) Conscientiousness 盡責性\n(4) Agreeableness 親和性\nYour personality trait: "
        )
    )

    # Initialize AIML for requesting id
    query_id_chat = aiml.Kernel()
    query_id_chat.learn("./model/startup.xml")
    query_id_chat.respond("LOAD AIML ID")
    current_id = False
    start_time = time.time()
    IM = InteractionManagement(personality_trait)
    time.sleep(5)

    logging.basicConfig()
    URL_facereg_main = ("/","web_server_template")  ##宣告URL與class的連接
    app = web.application(URL_facereg_main,locals())  ##初始化web application，默認地址為127.0.0.1:8080，locals()代表web.py會在當前文件內尋找url對應的class
    app.run()  ##運行web application

    # while not robot.is_robot_connected:
    #     time.sleep(1)
    # print("connected!")

    # while True:
    #     time.sleep(1)

