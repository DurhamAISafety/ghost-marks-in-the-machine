def solve():
    s = input()
    l, r = 0, len(s) - 1
    while l <= r and s[l] == '[' and s[r] == ']':
        l += 1
        r -= 1
        if l > r:
            print('-1')
        else:
            cnt = 0
            while l <= r and s[l] == ':':
                cnt += 1
                l += 1
            while l <= r and s[r] == ':':
                cnt += 1
                r -= 1
            if l > r or s[l] != ':' or s[r] != ':':
                print('-1')
            elif s[l + 1:r] == '|':
                print(l + 3 + cnt)
            else:
                print(2 + cnt)
                
solve()