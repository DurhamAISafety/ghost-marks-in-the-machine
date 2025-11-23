def solve():
    n = int(input())
    s = input()
    if s[0] != '[' or s[n - 1] != ']':
        print(-1)
        return
    flag = 0
    for i in range(n):
        if s[i] == '[':
            flag += 1
        if s[i] == ']':
            flag -= 1
        if flag == -1:
            print(-1)
            return
    if flag == 0 and n >= 4:
        print(n - 2)
    else:
        print(-1)

solve()