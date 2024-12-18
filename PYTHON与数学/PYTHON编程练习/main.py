N = int(input())
cicle = {}
for i in range(N):
    line = input().split()
    k = int(line[0])
    if k == 1:
        continue
    for j in range(1, k + 1):
        cicle[line[j]] = k
chaxun = int(input())
person = input().split()

ishand = False
handsomeId = []
for i in range(chaxun):
    if (person[i] not in cicle or cicle[person[i]] == 1) and person[i] not in handsomeId:
        handsomeId.append(person[i])
        ishand = True
if not ishand:
    print('No one is handsome')
else:
    print(' '.join(handsomeId))
