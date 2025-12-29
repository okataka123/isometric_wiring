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
    def __init__(self, grid_size):
        if isinstance(grid_size, (tuple, list)):
            n_x, n_y, n_z = grid_size
        else:
            n_x = n_y = n_z = grid_size
        self.n_x = n_x
        self.n_y = n_y
        self.n_z = n_z
        self.N = n_x * n_y * n_z
        self.grid_size = grid_size
        self.obs = self.generate_obstacle_nodes_3d(obs_size=3)
        self.to = self.make_to()
        self.to_without_obs = self.make_to_without_obs()


    def make_to(self) -> List[List[int]]:
        """
        グラフ構造の定義（3次元格子グラフ）
        """
        to = [[] for _ in range(self.N)]  # 隣接リスト

        for i in range(self.N):
            # nodenum → (x, y, z) に変換
            z = i // (self.n_x * self.n_y)
            rem = i % (self.n_x * self.n_y)
            y = rem // self.n_x
            x = rem % self.n_x

            # +x 方向
            if x + 1 < self.n_x:
                if not i+1 in self.obs:
                    to[i].append(i + 1)
            # -x 方向
            if x - 1 >= 0:
                if not i-1 in self.obs:
                    to[i].append(i - 1)
            # +y 方向
            if y + 1 < self.n_y:
                if not i+self.n_x in self.obs:
                    to[i].append(i + self.n_x)
            # -y 方向
            if y - 1 >= 0:
                if not i-self.n_x in self.obs:
                    to[i].append(i - self.n_x)
            # +z 方向
            if z + 1 < self.n_z:
                if not i+self.n_x*self.n_y in self.obs:
                    to[i].append(i + self.n_x * self.n_y)
            # -z 方向
            if z - 1 >= 0:
                if not i - self.n_x*self.n_y in self.obs:
                    to[i].append(i - self.n_x * self.n_y)
        return to


    def make_to_without_obs(self) -> List[List[int]]:
        """
        グラフ構造の定義（3次元格子グラフ）
        """
        # n = self.grid_size # 各軸方向のサイズ（n×n×n）
        to_without_obs = [[] for _ in range(self.N)]  # 隣接リスト

        for i in range(self.N):
            # nodenum → (x, y, z) に変換
            z = i // (self.n_x * self.n_y)
            rem = i % (self.n_x * self.n_y)
            y = rem // self.n_x
            x = rem % self.n_x

            # +x 方向
            if x + 1 < self.n_x:
                to_without_obs[i].append(i + 1)
            # -x 方向
            if x - 1 >= 0:
                to_without_obs[i].append(i - 1)
            # +y 方向
            if y + 1 < self.n_y:
                to_without_obs[i].append(i + self.n_x)
            # -y 方向
            if y - 1 >= 0:
                to_without_obs[i].append(i - self.n_x)
            # +z 方向
            if z + 1 < self.n_z:
                to_without_obs[i].append(i + self.n_x * self.n_y)
            # -z 方向
            if z - 1 >= 0:
                to_without_obs[i].append(i - self.n_x * self.n_y)
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
        if min(self.n_x, self.n_y, self.n_z) <= 3: assert False, "グリッドサイズが小さすぎます。"
        obs = []

        if is_random:
            basepoint = [random.randint(0, self.N-1) for _ in range(n_obs)]
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
            cond = (self.n_x-bp_x >= obs_size) & (self.n_y-bp_y >= obs_size) & (self.n_z-bp_z >= obs_size)
            if not cond: continue
            for layer in range(obs_size):
                for row in range(obs_size):
                    for col in range(obs_size):
                        add_point = bp + layer*self.n_y*self.n_x + row*self.n_x + col
                        # print('add_point =', add_point)
                        obs.append(add_point)

            # 重複ノードを除外する。
            obs = list(set(obs))
        return obs


    def coord_to_nodenum(self, x, y, z):
        nodenum = self.n_y*self.n_x*z + self.n_x*y + x
        return nodenum


    def nodenum_to_coord(self, nodenum):
        z = nodenum // (self.n_y * self.n_x)
        rem = nodenum % (self.n_y * self.n_x)
        y = rem // self.n_x
        x = rem % self.n_x
        return x, y, z
    

    def counting_corner(self, path):
        """
        pathの曲がり回数をカウントする
        """
        corner_count = 0
        path_coordinates = []
        for p in path:
            x_, y_, z_ = self.nodenum_to_coord(p)
            path_coordinates.append((x_, y_, z_))

        for i in range(len(path_coordinates)):
            if i == 0 or i == len(path_coordinates)-1:
                # nop
                pass
            else:
                vec_1_x = path_coordinates[i][0] - path_coordinates[i-1][0]
                vec_1_y = path_coordinates[i][1] - path_coordinates[i-1][1]
                vec_1_z = path_coordinates[i][2] - path_coordinates[i-1][2]
                vec_1 = (vec_1_x, vec_1_y, vec_1_z)

                vec_2_x = path_coordinates[i+1][0] - path_coordinates[i][0]
                vec_2_y = path_coordinates[i+1][1] - path_coordinates[i][1]
                vec_2_z = path_coordinates[i+1][2] - path_coordinates[i][2]
                vec_2 = (vec_2_x, vec_2_y, vec_2_z)

                if vec_1 != vec_2:
                    corner_count += 1
    
        return corner_count