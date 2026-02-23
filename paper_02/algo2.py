import numpy as np
import random
import math

class Algo2:
    """
    満足化トレードオフ実装. 第0版
    """
    def __init__(self, graph, dim, algo_name):
        self.graph = graph
        self.dim = dim
        self.algo_name = algo_name

        # --- weighted(論文) 用 ---
        self.w1 = None  # 経路長
        self.w2 = None  # 交差数
        self.w3 = None  # 長さ整合
        self.w4 = None  # 曲げ回数など

        # --- satisficing 用（満足化トレードオフ法） ---
        # 1:交差 → 2:等長性 → 3:最短性
        self.phase = 1
        self.select_mode = "weighted"  # "weighted" or "satisficing"

        # 交差=0 / 等長性=0 を満たすかの判定に使う（厳密に0のみで良いならそのまま）
        self.cross_target = 0
        self.lengthdiff_target = 0

    # ============================================================
    # 目的値（交差Υ・等長性Γ・長さΛ）を「タプル」で計算（満足化で必須）
    # ============================================================
    def _objectives(self, path, best_paths):
        """
        f1: 交差点数（Υ）
        f2: 等長性（Γ） ＝ 他 best_paths との長さ差二乗和
        f3: 経路長（Λ）
        """
        length = len(path)

        # 交差：他の best_paths とノード共有した回数（論文実装に合わせる）
        cross_count = sum(
            1 for node in path
            for other in best_paths
            if other and node in other
        )

        # 等長性：他 best_paths との長さ差二乗和（論文実装に合わせる）
        length_diff = sum((length - len(p)) ** 2 for p in best_paths if p)

        # (f1, f2, f3)
        return cross_count, length_diff, length

    # ============================================================
    # 論文実装：重み付き単目的スコア
    # ============================================================
    def _evaluate_path(self, path, best_paths):
        """
        目的関数（論文実装）：
        - 経路長
        - 交差数
        - 長さ整合
        """
        length = len(path)
        cross_count = sum(1 for node in path for other in best_paths if other and node in other)
        length_diff = sum((length - len(p)) ** 2 for p in best_paths if p)
        score = (self.w1 * length + self.w2 * cross_count + self.w3 * length_diff) / (self.w1 + self.w2 + self.w3)
        return score

    def _evaluate_path_cumtom_1(self, path, best_paths):
        """
        カスタム目的関数その1（曲げ回数含む）
        """
        length = len(path)
        cross_count = sum(1 for node in path for other in best_paths if other and node in other)
        length_diff = sum((length - len(p)) ** 2 for p in best_paths if p)
        corner_count = self.graph.counting_corner(path)
        score = (self.w1 * length + self.w2 * cross_count + self.w3 * length_diff + self.w4 * corner_count) / (
            self.w1 + self.w2 + self.w3 + self.w4
        )
        return score

    def _evaluate_path_cumtom_2(self, path, best_paths):
        """
        カスタム目的関数その2（例：z_maxなど）
        """
        length = len(path)
        cross_count = sum(1 for node in path for other in best_paths if other and node in other)
        length_diff = sum((length - len(p)) ** 2 for p in best_paths if p)
        z_max = None
        score = (self.w1 * length + self.w2 * cross_count + self.w3 * length_diff + self.w4 * z_max) / (
            self.w1 + self.w2 + self.w3 + self.w4
        )
        return score

    # ============================================================
    # 満足化トレードオフ法： min_x max_i ((f_i - f_i^hope) / (f_i^hope - f_i^ideal))
    # ここでは f_i^ideal = 0 とするので、分母は f_i^hope
    # ============================================================
    @staticmethod
    def _satisficing_score(f, f_hope):
        """
        f, f_hope は同じ次元のタプル（(f1,f2,f3) など）
        ideal=0 前提で、max_i ((f_i - hope_i) / hope_i) を返す
        hope_i=0 のときは
          - f_i==0 なら 0（完全満足）
          - f_i>0 なら +inf（希望を満たせない）
        """
        ratios = []
        for fi, hi in zip(f, f_hope):
            if hi == 0:
                ratios.append(0.0 if fi == 0 else float("inf"))
            else:
                ratios.append((fi - hi) / hi)
        return max(ratios)

    def _pick_hope(self, candidates_f, phase):
        """
        希求解の選び方（あなたの記述に対応）：
        - phase1: f1（交差）最小の候補 i を取り、その (f1,f2,f3) を hope にする
        - phase2: まず f1==0 を満たす候補に絞り、f2（等長性）最小の候補を hope にする
        - phase3: f1==0 & f2==0 を満たす候補に絞り、f3（長さ）最小の候補を hope にする

        候補が空の場合は、段階を緩めて選ぶ（フォールバック）。
        """
        if not candidates_f:
            return None

        if phase == 1:
            # 交差最小
            return min(candidates_f, key=lambda x: x[0])

        if phase == 2:
            feas = [f for f in candidates_f if f[0] == self.cross_target]
            if not feas:
                # まだ交差0が出ないなら、交差最小の個体を hope にして継続（phase1に近い挙動）
                return min(candidates_f, key=lambda x: x[0])
            return min(feas, key=lambda x: x[1])

        if phase == 3:
            feas = [f for f in candidates_f if f[0] == self.cross_target and f[1] == self.lengthdiff_target]
            if not feas:
                # 等長性0がまだ出ないなら phase2 に近い hope を取る
                feas2 = [f for f in candidates_f if f[0] == self.cross_target]
                if feas2:
                    return min(feas2, key=lambda x: x[1])
                return min(candidates_f, key=lambda x: x[0])
            return min(feas, key=lambda x: x[2])

        raise ValueError("invalid phase")

    def _select_best_by_satisficing(self, candidates, best_paths, phase):
        """
        candidates: list[path]
        best_paths: 現時点の他コロニー best_paths
        phase: 1/2/3

        手順：
        1) 各候補の f=(f1,f2,f3) を計算
        2) hope を _pick_hope で決める
        3) min_path: min_x max_i ((f_i - hope_i)/hope_i) を最小にする候補を選ぶ
           - phase2/3 では「制約として扱う」ので、優先制約を満たす候補を強く優先
             具体的には：
               phase2: f1==0 の候補を優先集合とし、その中で満足化スコア最小
               phase3: f1==0 & f2==0 の候補を優先集合とし、その中で満足化スコア最小
        """
        # f を全候補分計算
        cand_f = [self._objectives(p, best_paths) for p in candidates]
        hope = self._pick_hope(cand_f, phase)
        if hope is None:
            return None

        # phase ごとの「制約優先」集合を作る
        if phase == 1:
            pool = list(zip(candidates, cand_f))
        elif phase == 2:
            pool0 = [(p, f) for p, f in zip(candidates, cand_f) if f[0] == self.cross_target]
            pool = pool0 if pool0 else list(zip(candidates, cand_f))
        elif phase == 3:
            pool0 = [(p, f) for p, f in zip(candidates, cand_f)
                     if f[0] == self.cross_target and f[1] == self.lengthdiff_target]
            if pool0:
                pool = pool0
            else:
                pool1 = [(p, f) for p, f in zip(candidates, cand_f) if f[0] == self.cross_target]
                pool = pool1 if pool1 else list(zip(candidates, cand_f))
        else:
            raise ValueError("invalid phase")

        # 満足化スコア最小を選ぶ（同点なら辞書式で f が小さい方）
        best_p, best_f, best_s = None, None, float("inf")
        for p, f in pool:
            s = self._satisficing_score(f, hope)
            if (s < best_s) or (s == best_s and (best_f is None or f < best_f)):
                best_p, best_f, best_s = p, f, s

        return best_p

    # ============================================================
    # 経路生成（そのまま）
    # ============================================================
    def _generate_path(self, start, goal, pheromone, occupied, alpha=2, beta=2):
        current = start
        path = [current]
        visited = set([current])

        while current != goal:
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

    # ============================================================
    # ACO本体：select_mode により「論文（weighted）」と「満足化」を切替
    # ============================================================
    def equal_length_routing(
        self,
        pairs,
        max_iter=100,
        num_ants=30,
        w1=10, w2=45, w3=45, w4=None,
        select_mode="weighted",  # "weighted" or "satisficing"
        phase_init=1,            # satisficing の開始フェーズ
        alpha=2, beta=2          # ACO遷移のハイパーパラメータ（generate_pathに渡す）
    ):
        # 重みの設定（weightedモードで使用）
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.w4 = w4

        # モード切替
        self.select_mode = select_mode
        self.phase = phase_init

        n = self.graph.grid_size
        if self.dim == 2:
            pheromones = [np.ones(n * n) * 0.1 for _ in pairs]
        elif self.dim == 3:
            pheromones = [np.ones(n * n * n) * 0.1 for _ in pairs]
        else:
            raise ValueError("dim must be 2 or 3")

        best_paths = [None] * len(pairs)

        for iteration in range(max_iter):
            # print(f"iteration = {iteration}")

            # ---------------------------
            # 各トレース（コロニー）ごとに最良経路を更新
            # ---------------------------
            for i, (start, goal) in enumerate(pairs):
                candidates = []

                for _ in range(num_ants):
                    # 混雑度（他トレースが使っているセル数）
                    if self.dim == 2:
                        occupied = np.zeros(n * n)
                    else:
                        occupied = np.zeros(n * n * n)

                    for j, other_path in enumerate(best_paths):
                        if i != j and other_path:
                            for node in other_path:
                                occupied[node] += 1

                    # 経路生成
                    path = self._generate_path(start, goal, pheromones[i], occupied, alpha=alpha, beta=beta)
                    if path:
                        candidates.append(path)

                if not candidates:
                    continue

                # ---------------------------
                # 候補から「採用する best」を選ぶ
                # ---------------------------
                if self.select_mode == "weighted":
                    scored = []
                    for path in candidates:
                        match self.algo_name:
                            case "original":
                                score = self._evaluate_path(path, best_paths)
                            case "add_corner_constraint":
                                score = self._evaluate_path_cumtom_1(path, best_paths)
                            case "XXXX":
                                score = self._evaluate_path_cumtom_2(path, best_paths)
                            case _:
                                raise ValueError(f"unknown algo_name: {self.algo_name}")
                        scored.append((score, path))
                    best = min(scored, key=lambda x: x[0])[1]

                elif self.select_mode == "satisficing":
                    # 満足化：あなたの優先順位に沿って phase を使って選択
                    best = self._select_best_by_satisficing(candidates, best_paths, self.phase)
                    if best is None:
                        continue

                else:
                    raise ValueError("select_mode must be 'weighted' or 'satisficing'")

                # ---------------------------
                # best を採用してフェロモン更新（論文同様：簡易版）
                # ※ 満足化でも pheromone 更新は同じでOK（選択規則だけ変えている）
                # ---------------------------
                best_paths[i] = best
                for node in best:
                    pheromones[i][node] += 1.0 / len(best)
                pheromones[i] *= 0.9  # 蒸発（保持率0.9）

            # ---------------------------
            # 満足化のフェーズ遷移（グローバルに満たせたら次へ）
            # ---------------------------
            if self.select_mode == "satisficing":
                # 全トレースが揃っている前提で判定（Noneがあるなら判定を緩める）
                ready = all(p is not None for p in best_paths)
                if ready:
                    # グローバル交差（簡易：全パス間の共有ノード延べ数）
                    # ※ 論文定義は「他コロニーbestとの共有を数える」なので、ここもそれに近い形
                    global_cross = 0
                    for a in range(len(best_paths)):
                        for b in range(len(best_paths)):
                            if a == b:
                                continue
                            pa = best_paths[a]
                            pb = set(best_paths[b])
                            global_cross += sum(1 for node in pa if node in pb)

                    # グローバル等長性：各パス長のばらつき（論文のΓそのものではないが遷移判定用）
                    lens = [len(p) for p in best_paths]
                    global_equal = (len(set(lens)) == 1)

                    # phase1 -> phase2 : 交差0が達成できたら
                    if self.phase == 1 and global_cross == self.cross_target:
                        self.phase = 2

                    # phase2 -> phase3 : 交差0 かつ 等長が達成できたら
                    if self.phase == 2 and global_cross == self.cross_target and global_equal:
                        self.phase = 3

            # ---------------------------
            # 早期停止：等長（論文実装）※必要なら交差0も追加推奨
            # ---------------------------
            lengths = [len(p) for p in best_paths if p]
            if len(lengths) == len(pairs) and len(set(lengths)) == 1:
                return best_paths

        return best_paths