import grpc
import interaction_pb2
import interaction_pb2_grpc


def main():
    # Connect localhost:50051
    # channel = grpc.insecure_channel("140.112.187.97:50051", options=(('grpc.enable_http_proxy', 0),))
    # channel = grpc.insecure_channel("localhost:50051")
    channel = grpc.insecure_channel("127.0.0.1:50059")
    # Build a stub for grpc client
    stub = interaction_pb2_grpc.InteractStub(channel)

    request = interaction_pb2.RobotInput(utterance="Sentence")
    response = stub._RobotSend(request)
    print("Robot Sentence: ", response.utterance)

    while True:
        input_sentence = input("Enter sentence: ")
        request = interaction_pb2.RobotInput(utterance=input_sentence)
        response = stub._RobotSend(request)

        print("Robot Sentence: ", response.utterance)


if __name__ == "__main__":
    main()
