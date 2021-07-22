import pickle
import zhconv
import jieba
import jieba.posseg
import thulac
import pkuseg

a = dict()
with open("phraseTypeCache.pickle", "rb") as file:
    a = pickle.load(file, encoding="bytes")

emotion_set = set()
with open("emotion_word.txt", "r") as file:
    for _ in file.readlines():
        emotion_word = _.strip("\n")
        emotion_set.add(emotion_word)

keys = [key.decode("utf-8") for key in a]
values = [value for value in a.values()]

# Tool for pkuseg, thulac
thu1 = thulac.thulac(filt=True)
seg = pkuseg.pkuseg(postag=True)

noun_set = ["n", "np", "ns", "ni", "nz"]
part_set = ["n", "np", "ns", "ni", "nz", "m", "q", "mq", "t", "f", "s", "v", "a", "d"]


def check_part(part_speech):
    if "v" in part_speech:
        return "verbPhrase"
    elif "a" in part_speech:
        index = part_speech.index("a")
        if index == len(part_speech) - 1:
            return "adjPhrase"
        else:
            if not (
                part_speech[index + 1] in noun_set or part_speech[index + 1] == "u"
            ):
                return "adjPhrase"
            else:
                return "either"
    else:
        return "either"


# verbPhrase - has v in the word
# adjPhrase - adj or adv + adj
phrase_dict = dict()
for word in keys:
    tw_word = zhconv.convert(word, "zh-tw")
    cn_word = zhconv.convert(word, "zh-cn")
    if tw_word in emotion_set:
        phrase_dict[tw_word] = "emotion"
        continue
    thu_text = thu1.cut(cn_word)
    if len(thu_text) >= 3:
        pku_text = seg.cut(cn_word)
        part_speech = [temp_part[1] for temp_part in pku_text if temp_part[1] != "u"]
        result = check_part(part_speech)
    elif len(thu_text) == 1 and len(cn_word) >= 4 and (not thu_text[0][1] in part_set):
        result = a[word.encode("utf-8")]
    else:
        part_speech = [temp_part[1] for temp_part in thu_text]
        result = check_part(part_speech)

    phrase_dict[tw_word] = result

with open("./phraseTypeCache_new.pickle", "wb") as file:
    pickle.dump(phrase_dict, file)
