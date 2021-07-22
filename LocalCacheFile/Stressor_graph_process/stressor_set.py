import pickle
import networkx as nx

stressors = dict()
DG_stressor = nx.DiGraph()
cache = dict()

# Physiology
stressors["Physiology"] = dict()
PH1 = ["疲憊", "疲乏", "疲勞", "勞累", "疲累", "疲倦", "累"]
PH2 = ["吸毒", "嗑藥", "濫用藥物", "吸菸", "抽菸", "酗酒"]
PH3 = ["失眠", "睡眠不足", "沒睡飽"]
PH4 = ["營養不良", "節食", "挑食", "暴飲暴食"]
PH5 = ["不運動", "運動不足", "運動過量"]
PH6 = ["受傷", "不舒服", "不健康", "身體不適", "身體差", "身體不好", "疾病", "生病", "高血壓"]
PH7 = ["環境差", "環境不好", "衛生差", "衛生不好", "衛生條件差", "衛生條件不好"]
PH_query = [
    "疲憊",
    "疲乏",
    "疲勞",
    "勞累",
    "疲累",
    "疲倦",
    "累",
    "失眠",
    "睡眠不足",
    "沒睡飽",
    "營養不良",
    "疾病",
    "生病",
    "身體不適",
    "身體不好",
]

# Frustration
stressors["Frustration"] = dict()
F1 = ["挫折"]
F2 = ["失落"]
F3 = ["歧視", "鄙視", "侮蔑", "輕視", "藐視", "排擠", "虐待", "忽視", "霸凌"]
F4 = ["缺錢", "缺乏", "欠缺"]
F5 = ["挫敗", "失敗"]
F6 = ["死亡", "往生", "死掉", "去逝", "喪命", "亡故", "殞命", "歸天", "失去"]
F7 = ["炒魷魚", "開除", "失業", "革職", "解雇"]
F8 = ["離婚", "分居", "分離", "分手"]
F9 = ["失望", "絕望", "灰心", "沮喪"]
F10 = ["浪費時間"]
F11 = ["無法達成目標", "未完成", "沒有完成"]
F12 = ["沒信心", "沒自信", "自卑"]
F13 = ["失聯"]
F_query = [
    "挫折",
    "失落",
    "歧視",
    "鄙視",
    "侮蔑",
    "輕視",
    "藐視",
    "排擠",
    "虐待",
    "忽視",
    "挫敗",
    "失敗",
    "炒魷魚",
    "開除",
    "失業",
    "革職",
    "解雇",
    "離婚",
    "分居",
    "分離",
    "分手",
    "失望",
    "絕望",
    "灰心",
    "沮喪",
    "沒信心",
    "沒自信",
    "自卑",
]

# Pressure
stressors["Pressure"] = dict()
PR1 = ["超負荷", "負荷不足", "負荷過重", "責任", "升職", "降職", "升遷", "大材小用", "關係不良", "溝通不良", "受限"]
PR2 = [
    "考試",
    "口試",
    "筆試",
    "報告",
    "作業",
    "論文",
    "進度太快",
    "成績",
    "學分",
    "修課",
    "重修",
    "被當",
    "被二一",
    "延畢",
]
PR3 = ["缺錢", "沒錢", "破產", "貸款", "欠款", "欠錢", "借款", "經濟困難"]
PR4 = ["競爭", "競賽", "比賽"]
PR5 = ["超時", "時間不夠", "時間不足", "時間壓力", "忙不完", "做不完"]
PR6 = ["要求完美", "完美", "要求", "要求過高"]
PR7 = ["被打擾"]
PR8 = ["婚姻問題", "情侶問題", "員工問題", "出問題"]
PR9 = ["過度期待", "期待", "期望", "期望過高"]
PR10 = ["繁文縟節", "瑣事"]
PR11 = ["壓力"]
PR_query = ["重修", "被當", "被二一", "延畢", "沒錢", "欠錢", "壓力"]

# Conflict
stressors["Conflict"] = dict()
CF1 = ["吵架", "爭執", "爭吵", "衝突", "吵"]
CF2 = ["選擇", "抉擇", "為難"]
CF3 = ["意見不合"]
CF_query = ["吵架", "爭執", "爭吵", "衝突", "吵", "選擇", "抉擇", "為難", "意見不合"]

# Change
stressors["Change"] = dict()
CH1 = ["改變", "變化", "改動", "更動", "換"]
CH2 = ["結婚", "成婚"]
CH3 = ["懷孕", "有喜", "生小孩"]
CH4 = ["搬家", "喬遷", "遷居", "新生活"]
CH5 = ["分居", "離家", "外宿"]
CH6 = ["退休", "離職"]
CH7 = ["轉學", "轉科", "轉系", "留學"]
CH8 = ["新工作", "換工作"]

# Ioslation
stressors["Ioslation"] = dict()
I1 = ["沒有朋友", "沒朋友"]
I2 = ["寂寞", "孤單"]
I3 = ["獨自一人", "一個人"]
I_query = ["寂寞", "孤單", "獨自一人", "一個人"]


def add_edge_for_graph(graph, start_node, end_node, weight):
    if graph.has_edge(start_node, end_node):
        graph[start_node][end_node]["weight"] += weight
    else:
        graph.add_edge(start_node, end_node, weight=weight)

    return graph


with open("stressorSet.pickle", "rb") as file:
    stressors = pickle.load(file)

with open("initialConceptGraph_Stressor.pickle", "rb") as file:
    DG_stressor = pickle.load(file)

# print(stressors['Pressure'])
# print(DG_stressor.nodes(data='label'))

end_concerned_relations = ["Causes", "HasSubevent", "HasFirstSubevent"]

with open("conceptNetCache.pickle", "rb") as file:
    cache = pickle.load(file)

# temp = list()

# for node in PH1:
# 	temp.append(node)

# for node in PH2:
# 	temp.append(node)

# for node in PH3:
# 	temp.append(node)

# for seed in PH_query:
# 	for concept in cache:
# 		for index, relation in enumerate(cache[concept]['rel']):
# 			if relation in end_concerned_relations and cache[concept]['end'][index] == seed and (not concept in temp):
# 				start_node = concept
# 				end_node = seed
# 				weight = cache[concept]['weight'][index]

# 				DG_stressor = add_edge_for_graph(DG_stressor, start_node, end_node, weight= weight)

# print(DG_stressor.edges(data='weight'))
print(DG_stressor.has_edge("工作", "壓力"))
# print(cache['工作'])
print(list(DG_stressor.successors("考試考不好")))
# print(DG_stressor['工作']['壓力']['weight'])

# with open('stressorSet.pickle', 'wb') as file:
# 	pickle.dump(stressors, file)

# with open('initialConceptGraph_Stressor.pickle', 'wb') as file:
# 	pickle.dump(DG_stressor, file)