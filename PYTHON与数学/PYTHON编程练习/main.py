def check_matrix(matrix):
    num = []
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            num.append(matrix[i][j])
    for i in range(1, 10):
        if i in num:
            continue
        else:
            return False
    return True


def check_row_col(num):
    for i in range(1, 10):
        if i in num:
            continue
        else:
            return False
    return True


n = int(input())
for i in range(n):
    is_ok = True
    matrix = [[0] * 9 for _ in range(9)]
    for j in range(9):
        num = list(map(int, input().split()))
        matrix[j] = num
    # 判断行
    for j in range(9):
        temp = matrix[j]
        is_ok = check_row_col(temp)
        if not is_ok:
            break
    if not is_ok:
        print('0')
        continue
    # 判断列
    for j in range(9):
        temp = []
        for k in range(9):
            temp.append(matrix[k][j])
        is_ok = check_row_col(temp)
        if not is_ok:
            break
    if not is_ok:
        print('0')
        continue
    # 判断宫格
    for j in range(9):
        c = (j % 3) * 3
        r = (j // 3) * 3
        m = [[0] * 3 for _ in range(3)]
        for k in range(3):
            for h in range(3):
                m[k][h] = matrix[k + r][c + h]
        is_ok = check_matrix(m)
        if not is_ok:
            break
    if not is_ok:
        print('0')
        continue
    if is_ok:
        print('1')
