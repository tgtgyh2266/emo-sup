import pickle

with open("./LocalCacheFile/emotionalSupportResponse.pickle", "rb") as file:
    a = pickle.load(file)

new_key = '發現蜘蛛'
encouragement = '蜘蛛其實會幫忙清除一些小蟲子，對人類是有幫助的呢'
empathy = '我能理解你的感受，我也很怕蜘蛛'
advice = '平時保持環境整潔，也能間接減少蜘蛛食物的數量'

if new_key in a:
    print(a[new_key])
    print('already exists!')
else :
    a[new_key] = {'encouragement':[encouragement], 'empathy':[empathy], 'advice':[advice]}
    print(a[new_key])
    print('new key added!')


with open("./LocalCacheFile/emotionalSupportResponse.pickle", "wb") as file:
    pickle.dump(a, file)

with open("./LocalCacheFile/emotionalSupportResponse.pickle", "rb") as file:
    a = pickle.load(file)
print(len(a))

