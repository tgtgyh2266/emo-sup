import grpc
import interaction_pb2
import interaction_pb2_grpc
import time
from concurrent import futures
from hanziconv import HanziConv
import socket


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
        interaction_pb2.add_InteractServicer_to_server(self, self.server)
        self.server.add_insecure_port("[::]:" + str(port))
        print("gRPC server started on address " + self.get_ip() + ":" + str(port))
        if locale in ["zh", "en", "jp"]:
            self.locale = locale
            self.server.start()
        else:
            raise ValueError(
                "Interacter locale supports only Chinese (zh), English (en) or Japanese (jp)."
            )

    def RobotConnect(self, request, context):
        self.is_robot_connected = True
        self.robot_type = request.status
        return interaction_pb2.RobotConnectReply(status=self.locale)

    def RobotSend(self, request, context):
        self.user_utterance = self.toTraditional(request.utterance)
        # print('GOT: ' + self.user_utterance)
        self.robot_command = None
        while self.robot_command == None:
            time.sleep(0.5)

        return interaction_pb2.RobotOutput(utterance=self.robot_command)

    def say(self, speech, listen=False):
        if speech != "":
            if listen:
                print("say and listen: " + speech)
                self.robot_command = "mgetreply#" + speech
                while self.robot_command != None:
                    time.sleep(0.1)
                return self.user_utterance
            else:
                print("say: " + speech)
                self.robot_command = "mcont#" + speech
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
