import aiml
import os
import requests
import random
import zhconv
import time

from TranslateTool.Translator import Translator


# PRETRAINED_MODEL = "microsoft/DialoGPT-large"
# from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

mname = 'facebook/blenderbot-400M-distill'
model = BlenderbotForConditionalGeneration.from_pretrained(mname)
tokenizer = BlenderbotTokenizer.from_pretrained(mname)

# from googletrans import Translator

"""
# Let's chat for 5 lines
for step in range(5):
    # encode the new user input, add the eos_token and return a tensor in Pytorch
    new_user_input_ids = tokenizer.encode(input(">> User:") + tokenizer.eos_token, return_tensors='pt')

    # append the new user input tokens to the chat history
    bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids

    # generated a response while limiting the total chat history to 1000 tokens,
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    # pretty print last ouput tokens from bot
    print("DialoGPT: {}".format(tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)))
"""


class PandoraBot:
    def __init__(self, user_id="123456", dbs_path="./dbs", verbose=False):
        self.bot_id = "e397abf70e345a0e"
        self.botcust2 = self.gen_rnd(seed=int(user_id))
        self.verbose = verbose
        self.url = (
            "https://kakko.pandorabots.com/pandora/talk?botid={}&skin=mobile".format(
                self.bot_id
            )
        )

        # self.model = AutoModelForCausalLM.from_pretrained(PRETRAINED_MODEL)
        # self.tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)

    def gen_rnd(self, seed=None, bites=8):
        random.seed(seed)
        hex_str = hex(
            random.randint(1 + int((bites - 1) * "ff", 16), int(bites * "ff", 16))
        )[2:]
        return hex_str

    def parse_resp(self, text):
        chat_v = []
        for t in text.split("<B>You:</B> ")[1:]:
            a, b = t.split("<br> <B>Mitsuku:</B> ")

            a = a.strip()
            b = b.split("<br> <br>")[0]

            if "</P>" in b:
                b = b.split("</P>")[-1].strip()

            chat_v.append((a, b))

            if self.verbose:
                print("You: ", a)
                print("Mitsuku: ", b)
                print()

        return chat_v

    def query(self, q):
        # DialoGPT
        """
        user_input_ids = self.tokenizer.encode(">> User:" + q + self.tokenizer.eos_token, return_tensors='pt')
        response_ids = self.model.generate(user_input_ids, max_length=1000, pad_token_id=self.tokenizer.eos_token_id)[0][len(user_input_ids[0]):-1]
        response = self.tokenizer.decode(response_ids)
        print(response)
        #return response
        """

        # Mizuku
        text = q
        r = requests.post(self.url, data={"botcust2": self.botcust2, "message": text})
        self.chat_v = self.parse_resp(r.text)
        if len(self.chat_v) > 0:
            ret_text = self.chat_v[0][-1]
        else:
            ret_text = "sorry, I don't have an anwer"

        return ret_text

    def get_chat(self):
        if self.verbose:
            for a, b in self.chat_v:
                print("You: ", a)
                print("Mitsuku: ", b)
                print()

        return self.chat_v


class GeneralChat:
    def __init__(self):
        # self.translator = Translator(service_urls=['translate.google.com'])
        self.pb = PandoraBot(user_id=random.randint(1, 1e7))
        # self.pb = PandoraBot()
        self.translator = Translator()

    def modify_sentence(self, sentence):
        q_mark = ["嘛", "呢", "嗎"]
        e_mark = ["啊", "唷", "吧", "呀", "啦", "喔", "哇", "唉"]

        for qm in q_mark:
            if qm in sentence:
                sentence = sentence.replace(qm, qm + "？")

        for em in e_mark:
            if em in sentence:
                sentence = sentence.replace(em, em + "！")

        return sentence

    def respond(self, text, src_language="zh", dest_language="en", tool="facebook"):
        if tool == "baidu":
            time.sleep(2)
            input_zh = zhconv.convert(text, "zh-tw")
            input_en = self.translator.translate_baidu(
                input_zh, src=src_language, dest=dest_language
            )
            print("You: ", input_zh, input_en)
            respond_en = self.pb.query(input_en)
            print("Mitsuku: ", respond_en)
            respond_zh = self.translator.translate_baidu(
                respond_en, src=dest_language, dest=src_language
            )
            respond_zh = zhconv.convert(respond_zh, "zh-tw")
            respond_zh = respond_zh.replace("<br>", "")

        elif tool == "google":
            input_zh = zhconv.convert(text, "zh-tw")
            input_en = self.translator.translate_google(
                input_zh, src=src_language, dest=dest_language
            )
            print("You: ", input_zh, input_en)
            respond_en = self.pb.query(input_en)
            print("Mitsuku: ", respond_en)
            respond_zh = self.translator.translate_google(
                respond_en, src=dest_language, dest=src_language
            )
            respond_zh = respond_zh.replace("<br>", "")

        elif tool == "facebook":
            input_zh = zhconv.convert(text, "zh-tw")
            input_en = self.translator.translate_google(
                input_zh, src=src_language, dest=dest_language
            )
            print("You: ", input_zh, input_en)
            inputs = tokenizer([input_en], return_tensors='pt')
            reply_ids = model.generate(**inputs)
            respond_en = str(tokenizer.batch_decode(reply_ids))
            respond_en = respond_en.replace("[\"<s>","")
            respond_en = respond_en.replace("</s>\"]","")
            respond_en = respond_en.replace("[\'<s>","")
            respond_en = respond_en.replace("</s>\']","")
            respond_en = respond_en.replace("<s>","")

            print("BlenderBot: ", respond_en)
            respond_zh = self.translator.translate_google(
                respond_en, src=dest_language, dest=src_language
            )
            respond_zh = zhconv.convert(respond_zh, "zh-tw")
            respond_zh = respond_zh.replace("<br>", "")

        return respond_zh


def main():
    GC = GeneralChat()
    while True:
        input_zh = input("Enter your message >> ")
        print("BlenderBot: ", GC.respond(input_zh))


if __name__ == "__main__":
    main()
