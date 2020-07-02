def swap(data, i, j):
    tmp = data[i]
    data[i] = data[j]
    data[j] = tmp

def pushdown(data, s, t):
    i = s
    mid = int((t-1)/2)
    while i <= mid:
        j = 2*i+1 if 2*i+2 > t or data[2*i+1][1] > data[2*i+2][1] else 2*i+2
        if data[i][1] < data[j][1]:
            swap(data, i, j)
            i = j
        else:
            return

def heapsort(data, k):
    ret = []
    n = len(data)
    if n == 1:
        return data
    mid = int((n-2)/2)
    for i in range(mid, -1, -1):
        pushdown(data, i, n-1)
    for i in range(n-1, n-k-1, -1):
        ret += [data[0]]
        data[0] = data[i]
        pushdown(data, 0, i-1)
    return ret