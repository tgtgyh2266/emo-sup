import pickle

with open("./LocalCacheFile/emotionalSupportResponse.pickle", "rb") as file:
    a = pickle.load(file)

key = '爽約'

print(len(a))

if key not in a:
    print('doesn\'t exist!')
else:
    a.pop(key,None)

with open("./LocalCacheFile/emotionalSupportResponse.pickle", "wb") as file:
    pickle.dump(a, file)

with open("./LocalCacheFile/emotionalSupportResponse.pickle", "rb") as file:
    a = pickle.load(file)
print(len(a))