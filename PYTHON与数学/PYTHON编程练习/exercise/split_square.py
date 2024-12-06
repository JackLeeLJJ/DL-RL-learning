m,n=map(int,input().split())
count=int(0)
while(m!=1 or n!=1):
    if m>n:
        m-=n
        count+=1
    elif m<n:
        n-=m
        count+=1
    else:
        count+=1
        break
if m==n:
    print(count)
else:
    x=max(m,n)
    count+=x
    print(count)

