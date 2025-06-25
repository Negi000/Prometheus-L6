"""
シミュレーションエンジン (simulator.py)
モンテカルロ法による戦略パフォーマンスの将来シミュレーション
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Any
import random

logger = logging.getLogger(__name__)

class Simulator:
    def __init__(self):
        """
        シミュレーターの初期化
        """
        # LOTO6の基本設定
        self.total_combinations = 6096454  # 43C6
        self.ticket_price = 200
        
        # 各等級の理論確率
        self.theoretical_probabilities = {
            1: 1 / 6096454,                    # 1等: 6個すべて一致
            2: 6 / 6096454,                    # 2等: 5個一致+ボーナス
            3: 216 / 6096454,                  # 3等: 5個一致
            4: 9990 / 6096454,                 # 4等: 4個一致
            5: 155400 / 6096454                # 5等: 3個一致
        }
        
        # 各等級の理論賞金額
        self.theoretical_prizes = {
            1: 200000000,  # 2億円
            2: 10000000,   # 1000万円
            3: 300000,     # 30万円
            4: 6800,       # 6800円
            5: 1000        # 1000円
        }
        
        logger.info("シミュレーターを初期化しました")
    
    def run_monte_carlo_simulation(self, strategy_performance: Dict[str, float],
                                  purchase_count: int,
                                  simulation_rounds: int = 10000,
                                  draws_per_session: int = 8) -> Dict[str, Any]:
        """
        モンテカルロシミュレーションを実行
        
        Args:
            strategy_performance (Dict): 戦略の過去パフォーマンス統計
            purchase_count (int): 1回あたりの購入口数
            simulation_rounds (int): シミュレーション回数
            draws_per_session (int): 1セッションあたりの抽選回数（月間想定）
            
        Returns:
            Dict: シミュレーション結果
        """
        logger.info(f"モンテカルロシミュレーション開始: {simulation_rounds}回, "
                   f"{draws_per_session}回/セッション, {purchase_count}口/回")
        
        session_results = []
        
        for round_num in range(simulation_rounds):
            session_profit = 0
            session_hits = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
            
            # 1セッション（通常1ヶ月＝8回抽選）をシミュレート
            for draw in range(draws_per_session):
                draw_cost = purchase_count * self.ticket_price
                draw_winnings, draw_hits = self._simulate_single_draw(
                    strategy_performance, purchase_count
                )
                
                session_profit += draw_winnings - draw_cost
                
                # 当選回数を累積
                for level in session_hits:
                    session_hits[level] += draw_hits[level]
            
            session_results.append({
                'profit': session_profit,
                'hits': session_hits.copy(),
                'any_hit_3_or_above': session_hits[1] + session_hits[2] + session_hits[3] > 0,
                'any_hit_4_or_above': session_hits[1] + session_hits[2] + session_hits[3] + session_hits[4] > 0
            })
            
            # 進捗ログ
            if (round_num + 1) % 1000 == 0:
                logger.info(f"シミュレーション進捗: {round_num + 1}/{simulation_rounds}")
        
        # 結果の分析
        analysis = self._analyze_simulation_results(session_results, purchase_count, draws_per_session)
        
        logger.info("モンテカルロシミュレーション完了")
        return analysis
    
    def _simulate_single_draw(self, strategy_performance: Dict[str, float], 
                             purchase_count: int) -> Tuple[int, Dict[int, int]]:
        """
        1回の抽選をシミュレート
        
        Args:
            strategy_performance (Dict): 戦略のパフォーマンス統計
            purchase_count (int): 購入口数
            
        Returns:
            Tuple[int, Dict[int, int]]: (当選金額, 等級別当選回数)
        """
        total_winnings = 0
        hits_by_level = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        
        # 戦略の調整確率を取得（バックテスト結果から）
        adjusted_probabilities = self._get_adjusted_probabilities(strategy_performance)
        
        for ticket in range(purchase_count):
            # 各等級の当選判定
            for level in [1, 2, 3, 4, 5]:
                if random.random() < adjusted_probabilities[level]:
                    hits_by_level[level] += 1
                    total_winnings += self._get_prize_amount(level)
                    break  # 重複当選はなし（上位等級優先）
        
        return total_winnings, hits_by_level
    
    def _get_adjusted_probabilities(self, strategy_performance: Dict[str, float]) -> Dict[int, float]:
        """
        戦略のパフォーマンスに基づいて調整された当選確率を計算
        """
        # バックテストの的中率を基準とした調整
        base_hit_rate_4 = strategy_performance.get('hit_rate_4', 0.0)
        base_hit_rate_3 = strategy_performance.get('hit_rate_3', 0.0)
        
        # 理論確率に対する戦略の改善率
        theoretical_hit_rate_4 = (self.theoretical_probabilities[4] + 
                                 self.theoretical_probabilities[3] + 
                                 self.theoretical_probabilities[2] + 
                                 self.theoretical_probabilities[1])
        
        if theoretical_hit_rate_4 > 0:
            improvement_factor_4 = base_hit_rate_4 / theoretical_hit_rate_4
        else:
            improvement_factor_4 = 1.0
        
        # 調整された確率
        adjusted_probs = {}
        for level in [1, 2, 3, 4, 5]:
            base_prob = self.theoretical_probabilities[level]
            
            if level >= 4:  # 4等以上は戦略効果を反映
                adjusted_probs[level] = min(base_prob * improvement_factor_4, 0.1)  # 上限設定
            else:  # 3等以上は更に保守的に
                improvement_3 = base_hit_rate_3 / (theoretical_hit_rate_4 / 10) if theoretical_hit_rate_4 > 0 else 1.0
                adjusted_probs[level] = min(base_prob * improvement_3, 0.01)
        
        return adjusted_probs
    
    def _get_prize_amount(self, level: int) -> int:
        """
        等級に応じた賞金額を取得（変動要素を考慮）
        """
        base_prize = self.theoretical_prizes[level]
        
        # 賞金額に現実的な変動を追加
        if level == 1:  # 1等は大きく変動する可能性
            variation = random.uniform(0.5, 3.0)  # キャリーオーバー等
        elif level in [2, 3]:  # 2等、3等は中程度の変動
            variation = random.uniform(0.7, 1.5)
        else:  # 4等、5等は固定的
            variation = random.uniform(0.9, 1.1)
        
        return int(base_prize * variation)
    
    def _analyze_simulation_results(self, session_results: List[Dict], 
                                   purchase_count: int, 
                                   draws_per_session: int) -> Dict[str, Any]:
        """
        シミュレーション結果を分析
        """
        profits = [result['profit'] for result in session_results]
        
        # 基本統計
        analysis = {
            'basic_stats': {
                'mean_profit': np.mean(profits),
                'median_profit': np.median(profits),
                'std_profit': np.std(profits),
                'min_profit': np.min(profits),
                'max_profit': np.max(profits)
            },
            
            # リスク指標
            'risk_metrics': {
                'win_probability': sum(1 for p in profits if p > 0) / len(profits),
                'loss_probability': sum(1 for p in profits if p < 0) / len(profits),
                'break_even_probability': sum(1 for p in profits if p == 0) / len(profits),
                'var_95': np.percentile(profits, 5),  # 5%ile（Value at Risk）
                'var_99': np.percentile(profits, 1)   # 1%ile
            },
            
            # 当選確率
            'hit_probabilities': {
                'prob_3_or_above': sum(1 for result in session_results if result['any_hit_3_or_above']) / len(session_results),
                'prob_4_or_above': sum(1 for result in session_results if result['any_hit_4_or_above']) / len(session_results)
            },
            
            # 投資効率
            'investment_efficiency': {
                'total_investment_per_session': purchase_count * draws_per_session * self.ticket_price,
                'roi_mean': (np.mean(profits) / (purchase_count * draws_per_session * self.ticket_price)) * 100,
                'roi_median': (np.median(profits) / (purchase_count * draws_per_session * self.ticket_price)) * 100
            },
            
            # 分布情報
            'distribution': {
                'profit_distribution': profits,
                'percentiles': {
                    '10th': np.percentile(profits, 10),
                    '25th': np.percentile(profits, 25),
                    '75th': np.percentile(profits, 75),
                    '90th': np.percentile(profits, 90)
                }
            }
        }
        
        return analysis
    
    def compare_strategies(self, strategy1_performance: Dict[str, float],
                          strategy2_performance: Dict[str, float],
                          purchase_count: int,
                          simulation_rounds: int = 10000) -> Dict[str, Any]:
        """
        2つの戦略を比較
        
        Args:
            strategy1_performance (Dict): 戦略1のパフォーマンス
            strategy2_performance (Dict): 戦略2のパフォーマンス
            purchase_count (int): 購入口数
            simulation_rounds (int): シミュレーション回数
            
        Returns:
            Dict: 比較結果
        """
        logger.info("戦略比較シミュレーション開始")
        
        # それぞれの戦略でシミュレーション実行
        result1 = self.run_monte_carlo_simulation(
            strategy1_performance, purchase_count, simulation_rounds
        )
        result2 = self.run_monte_carlo_simulation(
            strategy2_performance, purchase_count, simulation_rounds
        )
        
        # 比較分析
        comparison = {
            'strategy1': result1,
            'strategy2': result2,
            'comparison': {
                'mean_profit_diff': result1['basic_stats']['mean_profit'] - result2['basic_stats']['mean_profit'],
                'risk_diff': result1['basic_stats']['std_profit'] - result2['basic_stats']['std_profit'],
                'win_prob_diff': result1['risk_metrics']['win_probability'] - result2['risk_metrics']['win_probability'],
                'sharpe_ratio_1': self._calculate_sharpe_ratio(result1),
                'sharpe_ratio_2': self._calculate_sharpe_ratio(result2)
            }
        }
        
        # 優劣判定
        comparison['recommendation'] = self._make_strategy_recommendation(comparison)
        
        logger.info("戦略比較シミュレーション完了")
        return comparison
    
    def _calculate_sharpe_ratio(self, simulation_result: Dict[str, Any]) -> float:
        """
        シャープレシオを計算（リスク調整後リターン）
        """
        mean_return = simulation_result['basic_stats']['mean_profit']
        std_return = simulation_result['basic_stats']['std_profit']
        
        if std_return == 0:
            return 0.0
        
        # リスクフリーレート（0%と仮定）
        return mean_return / std_return
    
    def _make_strategy_recommendation(self, comparison: Dict[str, Any]) -> Dict[str, str]:
        """
        比較結果に基づく推奨戦略を決定
        """
        comp = comparison['comparison']
        
        # 総合判定のスコアリング
        score1 = 0
        score2 = 0
        
        # 期待リターン
        if comp['mean_profit_diff'] > 0:
            score1 += 2
        else:
            score2 += 2
        
        # リスク（低い方が良い）
        if comp['risk_diff'] < 0:
            score1 += 1
        else:
            score2 += 1
        
        # 勝率
        if comp['win_prob_diff'] > 0:
            score1 += 1
        else:
            score2 += 1
        
        # シャープレシオ
        if comp['sharpe_ratio_1'] > comp['sharpe_ratio_2']:
            score1 += 2
        else:
            score2 += 2
        
        if score1 > score2:
            recommended = 'strategy1'
            reason = f"総合スコア {score1}:{score2} で戦略1が優勢"
        elif score2 > score1:
            recommended = 'strategy2'
            reason = f"総合スコア {score1}:{score2} で戦略2が優勢"
        else:
            recommended = 'equivalent'
            reason = f"総合スコア {score1}:{score2} で両戦略は同等"
        
        return {
            'recommended_strategy': recommended,
            'reason': reason,
            'confidence': abs(score1 - score2) / 6.0  # 最大差6点での信頼度
        }
