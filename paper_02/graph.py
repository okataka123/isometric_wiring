import sys
import math
from heapq import *
from collections import defaultdict
import random
import networkx as nx
import matplotlib.pyplot as plt
from typing import List
sys.setrecursionlimit(10**7)


class Graph_2d:
    def __init__(
            self, 
            grid_size: int, 
            ):
        self.grid_size = grid_size
        self.to = self.make_to()

    def make_to(self) -> List[List[int]]:
        n = self.grid_size
        to = [[] for _ in range(n*n)]
        for i in range(n*n):
            if (i+1) % n != 0:
                to[i].append(i+1)
            if i % n != 0:
                to[i].append(i-1)
            if i+n < n*n:
                to[i].append(i+n)
            if i-n >= 0:
                to[i].append(i-n)
        return to



class Graph_3d:
    def __init__(
            self, 
            grid_size: int, 
            ):
        self.grid_size = grid_size
        self.obs = self.generate_obstacle_nodes_3d(obs_size=3)
        self.to = self.make_to()
        self.to_without_obs = self.make_to_without_obs()



    def make_to(self) -> List[List[int]]:
        """
        グラフ構造の定義（3次元格子グラフ）
        """
        n = self.grid_size  # 各軸方向のサイズ（n×n×n）
        to = [[] for _ in range(n * n * n)]  # 隣接リスト

        for i in range(n * n * n):
            # nodenum → (x, y, z) に変換
            z = i // (n * n)
            rem = i % (n * n)
            y = rem // n
            x = rem % n

            # +x 方向
            if x + 1 < n:
                if not i+1 in self.obs:
                    to[i].append(i + 1)
            # -x 方向
            if x - 1 >= 0:
                if not i-1 in self.obs:
                    to[i].append(i - 1)
            # +y 方向
            if y + 1 < n:
                if not i+n in self.obs:
                    to[i].append(i + n)
            # -y 方向
            if y - 1 >= 0:
                if not i-n in self.obs:
                    to[i].append(i - n)
            # +z 方向
            if z + 1 < n:
                if not i + n*n in self.obs:
                    to[i].append(i + n * n)
            # -z 方向
            if z - 1 >= 0:
                if not i - n*n in self.obs:
                    to[i].append(i - n * n)
        return to


    def make_to_without_obs(self):
        """
        グラフ構造の定義（3次元格子グラフ）
        """
        n = self.grid_size # 各軸方向のサイズ（n×n×n）
        to_without_obs = [[] for _ in range(n * n * n)]  # 隣接リスト

        for i in range(n * n * n):
            # nodenum → (x, y, z) に変換
            z = i // (n * n)
            rem = i % (n * n)
            y = rem // n
            x = rem % n

            # +x 方向
            if x + 1 < n:
                to_without_obs[i].append(i + 1)
            # -x 方向
            if x - 1 >= 0:
                to_without_obs[i].append(i - 1)
            # +y 方向
            if y + 1 < n:
                to_without_obs[i].append(i + n)
            # -y 方向
            if y - 1 >= 0:
                to_without_obs[i].append(i - n)
            # +z 方向
            if z + 1 < n:
                to_without_obs[i].append(i + n * n)
            # -z 方向
            if z - 1 >= 0:
                to_without_obs[i].append(i - n * n)
        return to_without_obs



    def generate_obstacle_nodes_3d(self, obs_size=3, is_random=False):
        """
        3次元的な障害物ノードを作る。障害物の大きさは3*3*3に固定する。

        Args:
            grid_size (int):
            is_random (bool):
        Returns:
            obs (list[int]): 障害物ノードのリスト
        """
        if self.grid_size <= 3: assert False, "グリッドサイズが小さすぎます。"
        obs = []

        if is_random:
            basepoint = [random.randint(0, self.grid_size**3-1) for _ in range(n_obs)]
        else:
            # basepoint = [445]
            # basepoint = [11]
            # basepoint = [6]
            # basepoint = [16]
            basepoint = [6, 11]

        n_obs = len(basepoint)
        for i in range(n_obs):
            bp = basepoint[i]

            # 障害物ノードが連結になるかの確認
            bp_x, bp_y, bp_z = self.nodenum_to_coord(bp)
            print(f'(bp_x, bp_y, bp_z) = ({bp_x}, {bp_y}, {bp_z})')
            cond = (self.grid_size-bp_x >= obs_size) & (self.grid_size-bp_y >= obs_size) & (self.grid_size-bp_z >= obs_size)        
            if not cond: continue
            for layer in range(obs_size):
                for row in range(obs_size):
                    for col in range(obs_size):
                        add_point = bp + layer*self.grid_size*self.grid_size + row*self.grid_size + col
                        print('add_point =', add_point)
                        obs.append(add_point)

            # 重複ノードを除外する。
            obs = list(set(obs))
        return obs


    def coord_to_nodenum(self, x, y, z):
        nodenum = self.grid_size * self.grid_size*z + self.grid_size*y + x
        return nodenum


    def nodenum_to_coord(self, nodenum):
        z = nodenum // (self.grid_size * self.grid_size)
        rem = nodenum % (self.grid_size * self.grid_size)
        y = rem // self.grid_size
        x = rem % self.grid_size
        return x, y, z