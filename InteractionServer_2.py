from concurrent import futures
import logging

import aiml
import grpc
import zhconv
import interaction_pb2
import interaction_pb2_grpc
from hanziconv import HanziConv
import socket
import time
from InteractionManagement import InteractionManagement


class Interacter(interaction_pb2_grpc.InteractServicer):

    locale = ""
    is_robot_connected = False
    robot_type = ""
    robot_command = None
    user_utterance = ""

    def __init__(self, port, locale):

        # Actions allowed for each robot
        common_actions = [
            "take_photo",
            "take_video",
            "show_photo",
            "show_video",
            "random_dance",
        ]
        self.robohon_actions = common_actions + [
            "sit",
            "stand",
            "walk",
            "turn_left",
            "turn_right",
            "move_head",
            "move_left",
            "move_right",
        ]
        self.zenbo_actions = common_actions + [
            "music_browse",
            "music_search",
            "news_browse",
            "news_search",
            "weather_now",
        ]

        # Start gRPC server
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        interaction_pb2_grpc.add_InteractServicer_to_server(self, self.server)
        self.server.add_insecure_port( "0.0.0.0:" + str(port))
        # self.server.add_insecure_port("[::]:" + str(port))
        print("gRPC server started on address " + self.get_ip() + ":" + str(port))
        if locale in ["zh", "en", "jp"]:
            self.locale = locale
            self.server.start()
        else:
            raise ValueError(
                "Interacter locale supports only Chinese (zh), English (en) or Japanese (jp)."
            )

    def _RobotConnect(self, request, context):
        self.is_robot_connected = True
        self.robot_type = request.status
        return interaction_pb2.RobotConnectReply(status=self.locale)

    # def _RobotSend(self, request, context):
    #     self.user_utterance = self.toTraditional(request.utterance)
    #     # print('GOT: ' + self.user_utterance)
    #     self.robot_command = None
    #     while self.robot_command == None:
    #         time.sleep(0.5)

    #     return interaction_pb2.RobotOutput(utterance=self.robot_command)

    def _RobotSend(selg, request, context):
        global query_id_interaction_signal
        global first_question_signal
        # global start_interaction_signal
        global emotional_detect_signal
        global during_interaction_signal
        global passing_information
        global query_id_chat
        global current_id
        global IM
        global start_time

        # Get info from client
        # If the received sentence is "Sentence", that means it has nothing to update
        # start_time = time.time()
        # print('Received Message: ', request.utterance)

        # if query_id_interaction_signal:
        #     if first_question_signal:
        #         # time.sleep(5)
        #         passing_information = "嗨！我是Zenbo Junior。請問你叫什麼名字？"
        #         first_question_signal = False
        #         start_time = time.time()

        #     elif request.utterance == "Finish":
        #         passing_information = "empty"

        #     elif "\uff36\uff2f\uff29\uff23\uff25\uff30\uff26" in request.utterance:
        #         passing_information = "抱歉，我沒有聽清楚，麻煩請再說一次!"

        #     else:
        #         print("Received Message: ", request.utterance)
        #         start_time = time.time()
        #         # Use AIML to obtain user's name
        #         input_sentence = zhconv.convert(request.utterance, "zh-tw")
        #         user_name = query_id_chat.respond(request.utterance)

        #         # # Check the speech to text
        #         # check = input("Yes/No: ")
        #         # if check == 'y':
        #         #   input_sentence = zhconv.convert(request.utterance, 'zh-tw')
        #         #   user_name = query_id_chat.respond(request.utterance)
        #         # else:
        #         #   senetence = input("Sentence: ")
        #         #   input_sentence = zhconv.convert(request.utterance, 'zh-tw')
        #         #   user_name = query_id_chat.respond(request.utterance)

        #         if user_name:
        #             current_id = user_name.replace(" ", "")
        #             # current_id = '宇謙'
        #             query_id_interaction_signal = False
        #             # if not emotional_detect_signal:
        #             #     start_interaction_signal = True
        #         else:
        #             # passing_information = "請用我是或我叫什麼名字來回答，謝謝。舉例來說：我叫Zenbo Junior。"

        # elif start_interaction_signal:
        #     passing_information = current_id + ""
        #     start_interaction_signal = False
        #     during_interaction_signal = True

        if first_question_signal:
            passing_information = "嗨！今天想和我聊些什麼呢？"
            first_question_signal = False

        elif during_interaction_signal:
            # Heartbeat live signal
            if request.utterance == "Sentence":
                passing_information = "empty"

                # Idle 2 minutes: Reset current_id
                print("Idle Time: ", time.time() - start_time)
                if time.time() - start_time > 120:
                    query_id_interaction_signal = True

            # Return speaking finish
            elif request.utterance == "Finish":
                passing_information = "empty"

            # Noise cancel
            elif "\uff36\uff2f\uff29\uff23\uff25\uff30\uff26" in request.utterance:
                passing_information = "抱歉，我沒有聽清楚，麻煩請再說一次!"

            # Feed into the emotional support system
            else:
                print("Received Message: ", request.utterance)
                start_time = time.time()
                input_sentence = zhconv.convert(request.utterance, "zh-cn")
                passing_information = IM.emotional_support_system(
                    input_sentence, current_id
                )

                # # Check the speech to text
                # check = input("Yes/No: ")
                # if check == 'y':
                #   input_sentence = zhconv.convert(request.utterance, 'zh-cn')
                #   passing_information = IM.emotional_support_system(input_sentence, current_id)
                # else:
                #   senetence = input("Sentence: ")
                #   input_sentence = zhconv.convert(senetence, 'zh-cn')
                #   passing_information = IM.emotional_support_system(input_sentence, current_id)

                print("System Cost Time: ", time.time() - start_time)
        else:
            passing_information = "empty"

        return interaction_pb2.RobotOutput(utterance=passing_information)

    def say(self, speech, listen=False):
        if speech != "":
            if listen:
                print("say and listen: " + speech)
                #self.robot_command = "mgetreply#" + speech
                self.robot_command = speech
                while self.robot_command != None:
                    time.sleep(0.1)
                return self.user_utterance
            else:
                print("say: " + speech)
                # self.robot_command = "mcont#" + speech
                self.robot_command = speech
                while self.robot_command != None:
                    time.sleep(0.1)
                return None
        else:
            raise ValueError("Speech cannot be empty!")

    def move(self, x=0, y=0, theta=0, pitch=0):
        if self.robot_type == "zenbo":
            print(
                "moving robot to ("
                + str(x)
                + ","
                + str(y)
                + ","
                + str(theta)
                + ","
                + str(pitch)
                + ")"
            )
            self.robot_command = (
                "mmove_cmd,"
                + str(x)
                + ","
                + str(y)
                + ","
                + str(theta)
                + ","
                + str(pitch)
            )
            while self.robot_command != None:
                time.sleep(0.1)
            return
        else:
            raise ValueError("move() method not available for " + self.robot_type)

    def action(self, value, args=None):
        if (
            self.robot_type == "zenbo"
            and value in self.zenbo_actions
            or self.robot_type == "robohon"
            and value in self.robohon_actions
        ):
            if args is not None:
                value += "," + str(args)
            print("Performing action(" + value + ")")
            self.robot_command = "m" + value
            while self.robot_command != None:
                time.sleep(0.1)
            return
        else:
            raise ValueError(
                "action(" + value + ") not available for " + self.robot_type
            )

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

    # Load emotion detect model
    if emotional_detect_signal:
        ED = EmotionDetect()
        ED.build_model()
        video_capture_signal = True
        # start_interaction_signal = False

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
    robot = Interacter(port=50059, locale="zh")
    while not robot.is_robot_connected:
        time.sleep(1)
    print("connected!")

    while True:
        time.sleep(1)

    # is_acad = False
    # is_work = False
    # advice_count = 0
    
    # sentence = robot.say("你好!", listen=True)
    # for turn in range(0, 100):
    #     #print("asdklfklas;jfkl;asd;l", turn)
    #     print(f"turn {turn}")
    #     #sentence = input("Input sentence: ")
    #     #sentence = robot.say(" ", listen=True)
    #     sentence = zhconv.convert(sentence, "zh-cn")
    #     print("[INFO] SENTENCE: ", sentence)
    #     start = time.time()
    #     # stressor, feeling, advice = IM.flow_interaction(sentence)
    #     #response = IM.emotional_support_system(sentence, "123")
    #     response = ""
        
    #     # if turn == 0:
    #     #     print("[INFO] TURN#0")
    #     #     if "工作" in sentence:
    #     #         is_work = True
    #     #         #print("Now working!!!")
    #     #     elif "考试" in sentence:
    #     #         is_acad = True
    #     if "工作" in sentence:
    #         is_work = True
            
    #     elif "考试" in sentence:
    #         is_acad = True

    #     print(is_work, is_acad, sentence)
    #     if is_work:
    #         # turn 1
    #         if "压力" in sentence:
    #             #is_scripted()
    #             response = "要不要跟我一起出去走走，放下心中的憂鬱"
    #         # turn 3
    #         elif "建议" in sentence:
    #             #is_scripted()
    #             response = "安排規律生活 改掉不良習慣 培養起良好的生活規律是改善失眠的關鍵"
    #         # turn 4
    #         elif "鼓励" in sentence:
    #             #is_scripted()
    #             response = "工作上的不愉快是正常的 戰時的挫折很快就會過去"
    #         # turn 2
    #         elif ("失眠" in sentence) or ("睡眠" in sentence):
    #             #is_scripted()
    #             response = "要不要睡前一小時喝點熱牛奶，它能夠有效的幫助睡眠"
    #     elif is_acad:
    #         # turn 1
    #         if "沮丧" in sentence:
    #             #is_scripted()
    #             response = "我能理解考試考不好 你一定覺得很傷心"
    #         elif "建议" in sentence:
    #             #is_scripted()
    #             if advice_count == 0:
    #                 response = "下次考試前要不要找同學來一起學習 這樣能夠更有效率"
    #                 advice_count += 1
    #             elif advice_count == 1:
    #                 response = "要不要整理一下這次考試犯錯的地方 為下次考試好好做準備"
    #         elif "鼓励" in sentence:
    #             #is_scripted()
    #             response = "雖然你這次考試考不好，但你這陣子的努力我都看在眼裡，繼續保持下去， 相信下次你一定會進步"
    #         # elif '' in sentence:
    #         #    is_scripted()
    #         #    response = "我能理解考試考不好 你一定覺得很傷心"

    #     if "谢谢" in sentence:
    #         response = "再見，希望你能順利度過困境，下次難過的時候，隨時歡迎你繼續找我聊天"

    #     if response == "":
    #         response = "可以再說一次嗎?"

    #     #robot.say(response, listen=False)

    #     if response == "再見，希望你能順利度過困境，下次難過的時候，隨時歡迎你繼續找我聊天":
    #         robot.say(response, listen=False)
    #         break

    #     sentence = robot.say(response, listen=True)

    #     print("Cost time: ", time.time() - start)
