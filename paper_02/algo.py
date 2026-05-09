import numpy as np
import random

class Algo:
    def __init__(self, graph, dim, algo_name):
        self.graph = graph
        self.dim = dim
        self.algo_name = algo_name
        self.w1 = None # 経路長に関する重み 
        self.w2 = None # 交差数に関する重み
        self.w3 = None # 長さ整合に関する重み
        self.w4 = None # 曲げ回数に関する重み
    

    # --- 評価関数（3目的の加重平均）---
    def _evaluate_path(self, path: list[int], best_paths: list[list[int]]):
        """
        今のアリが生成したpathに対する目的関数の値を計算する（論文実装）
        3目的（経路長・交差数・長さ整合）の線形加重和により計算。

        Args:
            path (list[int]): 今のアリのpath 
            best_paths (list[list[int]]): それぞれのコロニーにおける（現状の）最良経路たちの集合
        Returns:   
            score (float): 今のアリのpathが生成した目的関数の値
        """
        # # for debug
        # print('len(best_paths) =', len(best_paths))
        # print('[_evaluate_path] path =', path)
        # print('[_evaluate_path] best_paths[0] =', best_paths[0])
        # print('[_evaluate_path] best_paths[1] =', best_paths[1])
        # print('[_evaluate_path] best_paths[2] =', best_paths[2])
        # print()

        length = len(path)
        # 他の経路と交差したノード数
        cross_count = sum(1 for node in path for other in best_paths if other and node in other)
        # 他の経路との長さ差の2乗の合計（長さ整合）
        length_diff = sum((length - len(p)) ** 2 for p in best_paths if p)
        # 加重平均としてスコアを返す
        score = (self.w1 * length + self.w2 * cross_count + self.w3 * length_diff) / (self.w1 + self.w2 + self.w3)
        return score
    

    def _evaluate_path_cumtom_1(self, path: list[int], best_paths: list[list[int]]):
        """
        カスタム目的関数その1。
        「曲げ回数の最小化」も目的関数に含めた。
    
        Args:
            path (list[int]): 今のアリのpath 
            best_paths (list[list[int]]): それぞれのコロニーにおける（現状の）最良経路たちの集合
        Returns:   
            score (float): 今のアリのpathに対する目的関数の値
        """
        length = len(path)
        # 他の経路と交差したノード数
        cross_count = sum(1 for node in path for other in best_paths if other and node in other)
        # 他の経路との長さ差の2乗の合計（長さ整合）
        length_diff = sum((length - len(p)) ** 2 for p in best_paths if p)
        # pathの曲がり回数
        # corner_count = self.counting_corner(path)
        corner_count = self.graph.counting_corner(path)
        # 加重平均としてスコアを返す
        score = (self.w1 * length + self.w2 * cross_count + self.w3 * length_diff + self.w4 * corner_count) / (self.w1 + self.w2 + self.w3 + self.w4)
        return score


    def _evaluate_path_cumtom_2(self, path, best_paths):
        """
        カスタム目的関数その2。
        「高さ(z座標)の最大値の最小化」も目的関数に含めた。
    
        - 経路長
        - 交差数
        - 長さ整合
        - 高さ(z座標)の最大値
        """
        length = len(path)
        # 他の経路と交差したノード数
        cross_count = sum(1 for node in path for other in best_paths if other and node in other)
        # 他の経路との長さ差の2乗の合計（長さ整合）
        length_diff = sum((length - len(p)) ** 2 for p in best_paths if p)
        # 高さ(z座標)の最大値
        z_max = None

        # 加重平均としてスコアを返す
        score = (self.w1 * length + self.w2 * cross_count + self.w3 * length_diff + self.w4 * z_max) / (self.w1 + self.w2 + self.w3 + self.w4)
        return score



    # --- 経路生成関数（ACO風ランダム探索） ---
    def _generate_path(self, start, goal, pheromone, occupied, alpha=2, beta=2):
        """
        或るコロニーの或るアリの生成するpathを算出する。

        Args:
            start (int): スタート地点のnode番号
            goal (int): ゴール地点のnode番号
            pheromone (np.ndarray): 各ノードのフェロモン。pheromone[node]のようにアクセスする。
            occupied (np.ndarray): 占領済みのノードを記録しておくarray
            alpha (int): ACOのハイパラ
            beta (int): ACOのハイパラ
        Returns:
            path (list[int]): 経路
        """
        current = start
        path = [current]
        visited = set([current])

        while current != goal:
            # neighbors = [v for v in self.to[current] if v not in visited]
            neighbors = [v for v in self.graph.to[current] if v not in visited]
            if not neighbors:
                return []  # 行き止まり

            desirability = []
            for node in neighbors:
                tau = pheromone[node] ** alpha
                eta = 1 / (1 + occupied[node])  # 他配線と重なってないほど好ましい
                desirability.append(tau * (eta ** beta))

            total = sum(desirability)
            probs = [d / total for d in desirability]
            current = random.choices(neighbors, weights=probs)[0]
            path.append(current)
            visited.add(current)
        return path
    

    # --- ACOによる等長・非交差ルーティング本体 ---
    def equal_length_routing(self, pairs, max_iter=100, num_ants=30, w1=10, w2=45, w3=45, w4=None):
        """
        ACOによる等長・非交差ルーティング本体（main処理）

        Args:
            pairs (list[tuple[int, int]]): 各配線のスタート地点・ゴール地点のペアの配列。
            max_iter (int): 最大イテレーション回数
            num_ants (int): 各コロニーのアリ数
            w1 (int): 目的関数1の重み
            w2 (int): 目的関数2の重み
            w3 (int): 目的関数3の重み
            w4 (int | None): 目的関数4の重み
        Returns:
            best_paths (list[list[int]]): それぞれのコロニーにおける（現状の）最良経路たちの集合
        """
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.w4 = w4

        n = self.graph.grid_size
        if self.dim == 2:
            pheromones = [np.ones(n * n) * 0.1 for _ in pairs]  # 初期フェロモン
        elif self.dim == 3:
            pheromones = [np.ones(n * n * n) * 0.1 for _ in pairs]  # 初期フェロモン
        else:
            assert False

        best_paths = [None] * len(pairs)  # 各ペアの最良経路

        for iteration in range(max_iter):
            for i, (start, goal) in enumerate(pairs):
                paths = []
                for _ in range(num_ants):
                    # 他のトレースが使っているセルを記録（混雑度）
                    if self.dim == 2:
                        occupied = np.zeros(n * n)
                    elif self.dim == 3:
                        occupied = np.zeros(n * n * n)
                    else:
                        assert False
                    for j, other_path in enumerate(best_paths):
                        if i != j and other_path:
                            for node in other_path:
                                occupied[node] += 1

                    # 経路生成 → 評価
                    path = self._generate_path(start, goal, pheromones[i], occupied)

                    # for debug 20260217
                    print(f"start = {start}, goal = {goal}")
                    print(f'path = {path}')

                    # # for debug 20260509
                    # print(f"[debug 20260509] len(best_paths) = {len(best_paths)}")
                    # print(f"[debug 20260509] best_paths[0] = {best_paths[0]}")
                    # print(f"[debug 20260509] best_paths[1] = {best_paths[1]}")
                    # print(f"[debug 20260509] best_paths[2] = {best_paths[2]}")
                    # print(f"[debug 20260509] path = {path}")
                    # print(f"[debug 20260509] type(best_paths) = {type(best_paths)}")

                    if path:
                        match self.algo_name:
                            case "original":
                                score = self._evaluate_path(path, best_paths) # 論文実装
                            case "add_corner_constraint":
                                score = self._evaluate_path_cumtom_1(path, best_paths) # カスタム目的関数その１
                            case "XXXX":
                                score = self._evaluate_path_cumtom_2(path, best_paths) # カスタム目的関数その２
                            case _:
                                assert False
                        paths.append((score, path))

                # 最良経路を採用・フェロモン更新
                if paths:
                    best = min(paths, key=lambda x: x[0])[1]
                    best_paths[i] = best
                    for node in best:
                        pheromones[i][node] += 1.0 / len(best)
                    pheromones[i] *= 0.9  # フェロモン蒸発

            # 等長化判定 → 成功したら早期停止
            lengths = [len(p) for p in best_paths if p]
            if len(lengths) == len(pairs) and len(set(lengths)) == 1:
                return best_paths

        return best_paths