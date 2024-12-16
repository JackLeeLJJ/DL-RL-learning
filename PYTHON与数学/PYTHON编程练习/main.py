import fractions

N=int(input())
nums=input().split()
fenzi=[]
fenmu=[]
for num in nums:
    temp1,temp2=num.split('/')
    fenzi.append(int(temp1))
    fenmu.append(int(temp2))
sum=fractions.Fraction(0,1)
for i in range(N):
    sum+=fractions.Fraction(fenzi[i],fenmu[i])
zhengshu=fractions.Fraction(int(sum),1)
if sum==zhengshu:
    print(zhengshu)
elif zhengshu==0:
    print(sum)
else:
    print(zhengshu,' ',sum-zhengshu)