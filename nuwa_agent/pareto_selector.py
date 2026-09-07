"""Pareto-Optimal Selector: NSGA-II 非支配排序 + 拥挤度 + Hypervolume"""

import numpy as np
from dataclasses import dataclass
from evaluator import EvaluationResult


@dataclass
class ParetoSolution:
    result: EvaluationResult
    rank: int
    crowding_distance: float


class ParetoSelector:
    """Pareto 最优选择器 + Hypervolume 计算"""

    def select(self, results: list, top_k: int = 10) -> list:
        """Pareto 选择: 可行解优先 → 非支配排序 → 拥挤度 → Top-K"""

        feasible = [r for r in results if r.is_feasible]
        infeasible = [r for r in results if not r.is_feasible]

        if not feasible:
            infeasible.sort(key=lambda r: len(r.constraint_violations))
            return [ParetoSolution(r, rank=999, crowding_distance=0.0)
                    for r in infeasible[:top_k]]

        obj_matrix = np.array([
            [r.te_score, r.stability_score, r.expression_score]
            for r in feasible
        ])

        fronts = self._non_dominated_sort(obj_matrix)

        solutions = []
        for rank, front_indices in enumerate(fronts):
            front_obj = obj_matrix[front_indices]
            crowding = self._crowding_distance(front_obj)
            for i, idx in enumerate(front_indices):
                solutions.append(ParetoSolution(
                    result=feasible[idx],
                    rank=rank,
                    crowding_distance=crowding[i],
                ))

        solutions.sort(key=lambda s: (s.rank, -s.crowding_distance))
        return solutions[:top_k]

    def compute_hypervolume(self, solutions: list,
                            reference_point: np.ndarray = None) -> float:
        """计算 Hypervolume (最大化问题, 值越大 HV 越大)

        目标: TE / Stability / Expression, 均为 [0,1] 越大越好。
        转换为最小化问题后调用 pymoo HV:
          loss = ref_ideal - obj, ref_point_for_minimization = [0,0,0]
        """
        front = [s for s in solutions if s.rank == 0]
        if not front:
            return 0.0

        obj_matrix = np.array([
            [s.result.te_score, s.result.stability_score, s.result.expression_score]
            for s in front
        ])

        # 理想点: 各目标理论上界 (均为 [0,1] 归一化)
        # 转换为最小化问题: loss = 1 - obj (loss ∈ [0,1], 越小越好)
        loss_matrix = 1.0 - obj_matrix

        # pymoo HV: 最小化问题, 参考点设为各维上界 (比所有 loss 值都大)
        try:
            from pymoo.indicators.hv import HV
            hv_obj = HV(ref_point=np.array([1.0, 1.0, 1.0]))
            return float(hv_obj(loss_matrix))
        except ImportError:
            pass

        # Fallback: 近似 HV — 对 Pareto 前沿每点计算目标值乘积后求和
        # 直观含义: 目标值越大 (越接近 [1,1,1]), 近似 HV 越大
        try:
            # 对每行 (每个 Pareto 解) 计算 TE * Stability * Expression
            per_solution = np.prod(obj_matrix, axis=1)   # shape: (n_front,)
            return float(per_solution.sum())
        except Exception:
            return 0.0

    def _non_dominated_sort(self, obj_matrix: np.ndarray) -> list:
        """NSGA-II 快速非支配排序"""
        n = len(obj_matrix)
        domination_count = np.zeros(n, dtype=int)
        dominated_set = [[] for _ in range(n)]

        for i in range(n):
            for j in range(i + 1, n):
                if self._dominates(obj_matrix[i], obj_matrix[j]):
                    dominated_set[i].append(j)
                    domination_count[j] += 1
                elif self._dominates(obj_matrix[j], obj_matrix[i]):
                    dominated_set[j].append(i)
                    domination_count[i] += 1

        fronts = []
        current_front = [i for i in range(n) if domination_count[i] == 0]
        fronts.append(current_front)

        while True:
            next_front = []
            for i in current_front:
                for j in dominated_set[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            if not next_front:
                break
            fronts.append(next_front)
            current_front = next_front

        return fronts

    @staticmethod
    def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
        return bool(np.all(a >= b) and np.any(a > b))

    def _crowding_distance(self, front_obj: np.ndarray) -> np.ndarray:
        """拥挤度距离"""
        n = len(front_obj)
        if n <= 2:
            return np.array([np.inf, np.inf] + [0.0] * max(0, n - 2))[:n]

        distances = np.zeros(n)
        for m in range(front_obj.shape[1]):
            sorted_idx = np.argsort(front_obj[:, m])
            distances[sorted_idx[0]] = np.inf
            distances[sorted_idx[-1]] = np.inf
            obj_range = front_obj[:, m].max() - front_obj[:, m].min()
            if obj_range < 1e-10:
                continue
            for i in range(1, n - 1):
                distances[sorted_idx[i]] += (
                    front_obj[sorted_idx[i + 1], m] - front_obj[sorted_idx[i - 1], m]
                ) / obj_range
        return distances
