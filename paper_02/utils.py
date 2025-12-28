from heapq import *

def dijkstra(start, n, to):
    prevs = [0 for _ in range(n*n)]
    d = [float('inf') for _ in range(n*n)]
    d[start] = 0
    q = [(0, start)]
    while q:
        (ci, i) = heappop(q)
        if d[i] < ci:
            continue
        for j in to[i]:
            if d[j] > d[i] + 1:
                d[j] = d[i] + 1
                prevs[j] = i
                heappush(q, (d[j], j))
    return d, prevs


def get_path(s, v, prevs):
    '''
    s -> vへの最短経路path
    '''
    if v == s:
        return [s]
    path = [v]
    while True:
        path.append(prevs[v])
        v = prevs[v]
        if v == s:
            return path