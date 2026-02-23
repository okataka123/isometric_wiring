from heapq import *
from typing import List

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
        
# ===========================================================================
def full_search_path(
        n, 
        to : List[List[int]], 
        start: int, 
        goal: int,
        limit: int,
        ):
    """
    頂点vからスタートし、事前に決めたゴール地点に長さLで進める経路をすべて列挙する。
    """
    if isinstance(n, (tuple, list)):
        n_x, n_y, n_z = n
    else:
        n_x = n_y = n_z = n
    N = n_x * n_y * n_z

    used = [False for _ in range(N)]
    path = []
    anss = []

    def dfs(v):
        path.append(v)
        used[v] = True    
        if len(path) >= limit:
            if path[-1] == goal:
                anss.append(path.copy())
            path.remove(v)
            used[v] = False
            return 
        for u in to[v]:
            if not used[u]:
                dfs(u)
        path.remove(v)
        used[v] = False

    dfs(start)

    return anss