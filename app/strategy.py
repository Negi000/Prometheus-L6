"""
戦略・ポートフォリオ生成モジュール (strategy.py)
AIの予測結果から購入戦略を生成し、高度な意思決定支援機能を提供
"""

import pandas as pd
import numpy as np
import logging
import configparser
from typing import Dict, List, Tuple, Any
from itertools import combinations

logger = logging.getLogger(__name__)

class Strategy:
    def __init__(self, config_path='config.ini'):
        """
        戦略クラスの初期化
        
        Args:
            config_path (str): 設定ファイルのパス
        """
        self.config = configparser.ConfigParser()
        self.config.read(config_path, encoding='utf-8')
        
        # 設定値の読み込み
        self.ticket_price = int(self.config['Parameters']['ticket_price'])
        self.weight_ai = float(self.config['Contrarian']['weight_ai_score'])
        self.weight_unpop = float(self.config['Contrarian']['weight_unpopularity'])
        
        logger.info("戦略モジュールを初期化しました")
    
    def generate_from_probabilities(self, probabilities: np.ndarray, 
                                   purchase_count: int,
                                   axis_numbers: List[int] = None,
                                   exclude_numbers: List[int] = None,
                                   enable_contrarian: bool = False) -> Dict[str, List[Tuple]]:
        """
        AIの予測確率からポートフォリオを生成
        
        Args:
            probabilities (np.ndarray): 各数字の出現確率（43要素）
            purchase_count (int): 購入口数
            axis_numbers (List[int]): 軸数字（必須含有）
            exclude_numbers (List[int]): 除外数字
            enable_contrarian (bool): 逆張り戦略の有効化
            
        Returns:
            Dict[str, List[Tuple]]: コア戦略とサテライト戦略のポートフォリオ
        """
        logger.info(f"ポートフォリオ生成開始: 購入口数={purchase_count}, "
                   f"軸数字={axis_numbers}, 除外={exclude_numbers}, 逆張り={enable_contrarian}")
        
        # 制約の適用
        available_numbers = self._apply_constraints(probabilities, axis_numbers, exclude_numbers)
        
        # 候補組み合わせの生成
        candidate_combinations = self._generate_candidate_combinations(
            available_numbers, probabilities, axis_numbers
        )
        
        # スコアリング
        scored_combinations = self._score_combinations(
            candidate_combinations, probabilities, enable_contrarian
        )
        
        # ポートフォリオの分割（コア70%、サテライト30%）
        core_count = int(purchase_count * 0.7)
        satellite_count = purchase_count - core_count
        
        portfolio = {
            'core': scored_combinations[:core_count],
            'satellite': self._generate_satellite_strategies(
                scored_combinations, available_numbers, satellite_count
            )
        }
        
        logger.info(f"ポートフォリオ生成完了: コア={len(portfolio['core'])}口, "
                   f"サテライト={len(portfolio['satellite'])}口")
        
        return portfolio
    
    def _apply_constraints(self, probabilities: np.ndarray, 
                          axis_numbers: List[int] = None,
                          exclude_numbers: List[int] = None) -> List[int]:
        """
        制約条件を適用して利用可能な数字を決定
        """
        # 基本的な候補数字（確率上位25個程度）
        top_n = min(25, len(probabilities))
        top_indices = np.argsort(probabilities)[-top_n:]
        available_numbers = [idx + 1 for idx in top_indices]
        
        # 軸数字の追加（必須）
        if axis_numbers:
            for num in axis_numbers:
                if 1 <= num <= 43 and num not in available_numbers:
                    available_numbers.append(num)
        
        # 除外数字の削除
        if exclude_numbers:
            available_numbers = [num for num in available_numbers 
                               if num not in exclude_numbers]
        
        return sorted(available_numbers)
    
    def _generate_candidate_combinations(self, available_numbers: List[int],
                                       probabilities: np.ndarray,
                                       axis_numbers: List[int] = None) -> List[Tuple[int, ...]]:
        """
        候補組み合わせを生成
        """
        if len(available_numbers) < 6:
            logger.warning(f"利用可能な数字が不足しています: {len(available_numbers)}個")
            # 不足分を確率の高い順で補完
            all_numbers = list(range(1, 44))
            sorted_by_prob = sorted(all_numbers, key=lambda x: probabilities[x-1], reverse=True)
            
            for num in sorted_by_prob:
                if num not in available_numbers:
                    available_numbers.append(num)
                if len(available_numbers) >= 20:
                    break
        
        candidates = []
        
        if axis_numbers:
            # 軸数字を含む組み合わせのみ生成
            remaining_slots = 6 - len(axis_numbers)
            remaining_numbers = [num for num in available_numbers if num not in axis_numbers]
            
            if remaining_slots > 0 and len(remaining_numbers) >= remaining_slots:
                for combo in combinations(remaining_numbers, remaining_slots):
                    full_combo = tuple(sorted(list(axis_numbers) + list(combo)))
                    candidates.append(full_combo)
        else:
            # 通常の6個組み合わせ生成
            candidates = list(combinations(available_numbers, 6))
        
        # 候補数が多すぎる場合は上位1000個に制限
        if len(candidates) > 1000:
            # 確率の合計でソートして上位を選択
            candidates_with_score = [
                (combo, sum(probabilities[num-1] for num in combo))
                for combo in candidates
            ]
            candidates_with_score.sort(key=lambda x: x[1], reverse=True)
            candidates = [combo for combo, _ in candidates_with_score[:1000]]
        
        return candidates
    
    def _score_combinations(self, combinations: List[Tuple[int, ...]],
                           probabilities: np.ndarray,
                           enable_contrarian: bool) -> List[Tuple[int, ...]]:
        """
        組み合わせをスコアリングして順序付け
        """
        scored_combinations = []
        
        for combo in combinations:
            # AI予測スコア（確率の合計）
            ai_score = sum(probabilities[num - 1] for num in combo)
            
            # 逆張りスコア
            if enable_contrarian:
                unpopularity_score = self._calculate_unpopularity_score(combo)
                final_score = (self.weight_ai * ai_score) - (self.weight_unpop * unpopularity_score)
            else:
                final_score = ai_score
            
            scored_combinations.append((combo, final_score))
        
        # スコアで降順ソート
        scored_combinations.sort(key=lambda x: x[1], reverse=True)
        
        return [combo for combo, _ in scored_combinations]
    
    def _calculate_unpopularity_score(self, combination: Tuple[int, ...]) -> float:
        """
        不人気スコアを計算（高いほど人気＝避けるべき）
        """
        score = 0.0
        combo_list = list(combination)
        
        # 誕生日バイアス: 1-31の数字が4つ以上
        birthday_count = sum(1 for num in combo_list if 1 <= num <= 31)
        if birthday_count >= 4:
            score += 3.0
        
        # 連続数字バイアス: 3つ以上の連続
        sorted_combo = sorted(combo_list)
        consecutive_count = 0
        for i in range(len(sorted_combo) - 2):
            if (sorted_combo[i+1] == sorted_combo[i] + 1 and 
                sorted_combo[i+2] == sorted_combo[i+1] + 1):
                consecutive_count += 1
        score += consecutive_count * 5.0
        
        # キリ番バイアス: 10の倍数が2つ以上
        round_numbers = sum(1 for num in combo_list if num % 10 == 0)
        if round_numbers >= 2:
            score += 2.0
        
        # 両端バイアス: 1と43の両方
        if 1 in combo_list and 43 in combo_list:
            score += 1.5
        
        # 一の位の重複
        endings = [num % 10 for num in combo_list]
        unique_endings = len(set(endings))
        if unique_endings <= 3:
            score += 2.0
        
        # 等差数列パターン
        if self._is_arithmetic_sequence(sorted_combo):
            score += 4.0
        
        return score
    
    def _is_arithmetic_sequence(self, numbers: List[int]) -> bool:
        """
        等差数列かどうかを判定
        """
        if len(numbers) < 3:
            return False
        
        diffs = [numbers[i+1] - numbers[i] for i in range(len(numbers)-1)]
        return len(set(diffs)) == 1  # 全ての差が同じ
    
    def _generate_satellite_strategies(self, core_combinations: List[Tuple[int, ...]],
                                     available_numbers: List[int],
                                     satellite_count: int) -> List[Tuple[int, ...]]:
        """
        サテライト戦略を生成（リスク分散・特殊戦略）
        """
        satellite_combos = []
        
        # 戦略1: 逆張り強化（最も不人気な組み合わせ）
        unpopular_combos = []
        for combo in combinations(available_numbers, 6):
            if combo not in core_combinations[:len(core_combinations)//2]:  # コア上位半分を除外
                unpop_score = self._calculate_unpopularity_score(combo)
                unpopular_combos.append((combo, unpop_score))
        
        # 不人気スコアが低い（＝人気がない）順にソート
        unpopular_combos.sort(key=lambda x: x[1])
        satellite_combos.extend([combo for combo, _ in unpopular_combos[:satellite_count//3]])
        
        # 戦略2: バランス重視（奇偶・高低のバランス）
        balanced_combos = []
        for combo in combinations(available_numbers, 6):
            if combo not in [c for c, _ in unpopular_combos[:satellite_count//3]]:
                balance_score = self._calculate_balance_score(combo)
                balanced_combos.append((combo, balance_score))
        
        balanced_combos.sort(key=lambda x: x[1], reverse=True)
        satellite_combos.extend([combo for combo, _ in balanced_combos[:satellite_count//3]])
        
        # 戦略3: ランダム要素（確率的多様性）
        remaining_combos = list(combinations(available_numbers, 6))
        used_combos = set(satellite_combos + core_combinations[:10])
        remaining_combos = [combo for combo in remaining_combos if combo not in used_combos]
        
        # ランダムに選択（ただし最低限の確率スコアを満たすもの）
        np.random.shuffle(remaining_combos)
        remaining_count = satellite_count - len(satellite_combos)
        satellite_combos.extend(remaining_combos[:remaining_count])
        
        return satellite_combos
    
    def _calculate_balance_score(self, combination: Tuple[int, ...]) -> float:
        """
        バランススコアを計算（奇偶、高低、分散など）
        """
        combo_list = list(combination)
        
        # 奇偶バランス（3:3が理想）
        odd_count = sum(1 for num in combo_list if num % 2 == 1)
        odd_balance = 1.0 - abs(odd_count - 3) / 3.0
        
        # 高低バランス（1-22 vs 23-43）
        low_count = sum(1 for num in combo_list if num <= 22)
        high_low_balance = 1.0 - abs(low_count - 3) / 3.0
        
        # 数字の分散（適度なばらつき）
        variance = np.var(combo_list)
        normalized_variance = min(variance / 200.0, 1.0)  # 正規化
        
        # 合計バランススコア
        balance_score = (odd_balance + high_low_balance + normalized_variance) / 3.0
        
        return balance_score
    
    def calculate_kelly_criterion(self, strategy_performance: Dict[str, float], 
                                 total_bankroll: float) -> Dict[str, Any]:
        """
        ケリー基準による最適投資額を計算
        
        Args:
            strategy_performance (Dict): 戦略のパフォーマンス統計
            total_bankroll (float): 総資金
            
        Returns:
            Dict: ケリー基準による推奨投資情報
        """
        # パフォーマンス指標の型安全な取得
        if not isinstance(strategy_performance, dict):
            logger.warning(f"strategy_performance が辞書型ではありません: {type(strategy_performance)}")
            win_rate_p = 0.0
        else:
            win_rate_p = strategy_performance.get('hit_rate_4', 0.0)  # 4等以上の勝率
            # 値の型チェック
            if not isinstance(win_rate_p, (int, float)):
                logger.warning(f"hit_rate_4 が数値型ではありません: {type(win_rate_p)}, デフォルト値使用")
                win_rate_p = 0.0
        
        # 設定からオッズを取得
        average_prize = float(self.config['KellyCriterion']['average_win_prize'])
        odds_b = average_prize / self.ticket_price
        
        # ケリーの公式: f* = (bp - q) / b
        lose_rate_q = 1 - win_rate_p
        kelly_fraction = (odds_b * win_rate_p - lose_rate_q) / odds_b
        
        # 安全策としてハーフケリーを採用
        safe_kelly_fraction = max(0, kelly_fraction / 2)
        
        # 推奨投資額と口数
        recommended_investment = total_bankroll * safe_kelly_fraction
        recommended_tickets = int(recommended_investment / self.ticket_price)
        
        kelly_info = {
            'win_rate': win_rate_p,
            'odds': odds_b,
            'kelly_fraction': kelly_fraction,
            'safe_kelly_fraction': safe_kelly_fraction,
            'recommended_investment': recommended_investment,
            'recommended_tickets': recommended_tickets,
            'risk_level': 'Low' if safe_kelly_fraction < 0.05 else 'Medium' if safe_kelly_fraction < 0.15 else 'High'
        }
        
        logger.info(f"ケリー基準計算完了: 勝率={win_rate_p:.1%}, 推奨口数={recommended_tickets}口")
        
        return kelly_info
    
    def format_portfolio_output(self, portfolio: Dict[str, List[Tuple]], 
                               probabilities: np.ndarray = None) -> str:
        """
        ポートフォリオを読みやすい形式でフォーマット
        
        Args:
            portfolio (Dict): ポートフォリオ情報
            probabilities (np.ndarray): 各数字の確率（オプション）
            
        Returns:
            str: フォーマットされたポートフォリオ文字列
        """
        output = []
        
        for strategy_type, combinations in portfolio.items():
            output.append(f"\n=== {strategy_type.upper()}戦略 ({len(combinations)}口) ===")
            
            for i, combo in enumerate(combinations, 1):
                numbers_str = ', '.join(f"{num:02d}" for num in sorted(combo))
                
                if probabilities is not None:
                    ai_score = sum(probabilities[num - 1] for num in combo)
                    total_sum = sum(combo)
                    odd_count = sum(1 for num in combo if num % 2 == 1)
                    
                    output.append(f"{i:2d}. [{numbers_str}] | スコア:{ai_score:.3f} | "
                                f"合計:{total_sum:3d} | 奇偶:{odd_count}:{6-odd_count}")
                else:
                    output.append(f"{i:2d}. [{numbers_str}]")
        
        return '\n'.join(output)
