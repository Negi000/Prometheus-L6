"""
特徴量エンジニアリングモジュール (feature_engine.py)
LOTO6データから機械学習用の特徴量を生成
"""

import pandas as pd
import numpy as np
import logging
import warnings
from typing import Dict, List, Tuple

# パフォーマンス警告を抑制
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

logger = logging.getLogger(__name__)

class FeatureEngine:
    def __init__(self, history_df: pd.DataFrame):
        """
        特徴量エンジンの初期化
        
        Args:
            history_df (pd.DataFrame): LOTO6履歴データ
        """
        self.df = history_df.copy()

        # 関連列を数値型に変換（エラーはNaNにする）
        num_cols = [col for col in self.df.columns if '本数字' in col or 'ボーナス数字' in col]
        # '第何回'も数値型に
        if '第何回' in self.df.columns:
            num_cols.append('第何回')
        elif '回' in self.df.columns:
            num_cols.append('回')

        for col in num_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

        # 「第何回」列の存在確認と統一
        if '第何回' not in self.df.columns:
            # 他の可能な列名から「第何回」に統一
            round_col_names = ['回', '回数', '回号']
            round_col = None
            for col_name in round_col_names:
                if col_name in self.df.columns:
                    round_col = col_name
                    break
            
            if round_col:
                self.df.rename(columns={round_col: '第何回'}, inplace=True)
                logger.info(f"特徴量エンジンで列名を '{round_col}' から '第何回' に変更しました")
            else:
                raise ValueError(f"回号列が見つかりません。利用可能な列: {list(self.df.columns)}")

        # データフレームを最適化
        self.df = self.df.copy()  # 断片化解除

        logger.info(f"特徴量エンジンを初期化しました。データ件数: {len(self.df)}")
    
    def run_all(self):
        """
        全ての特徴量生成を実行
        
        Returns:
            pd.DataFrame: 特徴量が追加されたデータフレーム
        """
        logger.info("特徴量生成を開始します...")
        
        try:
            # 基本特徴量
            self._generate_basic_flags()
            
            # DataFrameを最適化（断片化解除）
            self.df = self.df.copy()
            
            # 移動平均特徴量を効率的に生成
            self._generate_moving_averages_optimized()
            
            # DataFrameを最適化
            self.df = self.df.copy()
            
            # ギャップ特徴量を効率的に生成
            self._generate_gap_features_optimized()
            
            # DataFrameを最適化
            self.df = self.df.copy()
            
            # 組み合わせ特徴量
            self._generate_combination_features()
            
            # DataFrameを最適化
            self.df = self.df.copy()
            
            # セット球プロファイル特徴量
            self._generate_set_ball_profiles()
            
            # 最終的な最適化
            self.df = self.df.copy()
            
            logger.info(f"特徴量生成が完了しました。最終的な列数: {len(self.df.columns)}")
            return self.df
            
        except Exception as e:
            logger.error(f"特徴量生成中にエラーが発生しました: {e}")
            return None

    def _generate_basic_flags(self):
        """
        基本的な出現フラグを生成
        """
        logger.info("基本出現フラグを生成中...")
        
        # 本数字の出現フラグを一括生成
        main_flags_data = {}
        for n in range(1, 44):
            flags = np.zeros(len(self.df), dtype=int)
            for i in range(1, 7):  # 本数字1〜6
                col_name = f'本数字{i}'
                if col_name in self.df.columns:
                    flags |= (self.df[col_name] == n).astype(int)
            main_flags_data[f'is_appear_{n}'] = flags
        
        # ボーナス数字の出現フラグを一括生成
        bonus_flags_data = {}
        if 'ボーナス数字' in self.df.columns:
            for n in range(1, 44):
                bonus_flags_data[f'is_bonus_appear_{n}'] = (self.df['ボーナス数字'] == n).astype(int)
        
        # 一括でDataFrameに追加
        main_flags_df = pd.DataFrame(main_flags_data, index=self.df.index)
        bonus_flags_df = pd.DataFrame(bonus_flags_data, index=self.df.index)
        
        self.df = pd.concat([self.df, main_flags_df, bonus_flags_df], axis=1)
    
    def _generate_moving_averages_optimized(self):
        """
        移動平均特徴量を効率的に生成（データリーク防止のため1行シフト）
        """
        logger.info("移動平均特徴量を生成中...")
        
        ma_data = {}
        
        for n in range(1, 44):
            # 本数字
            col = f'is_appear_{n}'
            if col in self.df.columns:
                series = self.df[col]
                ma_data[f'ma_5_{n}'] = series.rolling(window=5, min_periods=1).mean().shift(1)
                ma_data[f'ma_25_{n}'] = series.rolling(window=25, min_periods=1).mean().shift(1)
                ma_data[f'ma_75_{n}'] = series.rolling(window=75, min_periods=1).mean().shift(1)

            # ボーナス数字
            bonus_col = f'is_bonus_appear_{n}'
            if bonus_col in self.df.columns:
                bonus_series = self.df[bonus_col]
                ma_data[f'ma_bonus_5_{n}'] = bonus_series.rolling(window=5, min_periods=1).mean().shift(1)
                ma_data[f'ma_bonus_25_{n}'] = bonus_series.rolling(window=25, min_periods=1).mean().shift(1)
                ma_data[f'ma_bonus_75_{n}'] = bonus_series.rolling(window=75, min_periods=1).mean().shift(1)
        
        # 一括でDataFrameに追加
        ma_df = pd.DataFrame(ma_data, index=self.df.index)
        self.df = pd.concat([self.df, ma_df], axis=1)

    def _generate_gap_features_optimized(self):
        """
        ギャップ特徴量を効率的に生成（データリーク防止）
        """
        logger.info("ギャップ特徴量を生成中...")
        
        gap_data = {}
        
        for n in range(1, 44):
            # 本数字のギャップ
            col = f'is_appear_{n}'
            if col in self.df.columns:
                gap_data[f'gap_{n}'] = self._calculate_gap_series(self.df[col])

            # ボーナス数字のギャップ
            bonus_col = f'is_bonus_appear_{n}'
            if bonus_col in self.df.columns:
                gap_data[f'gap_bonus_{n}'] = self._calculate_gap_series(self.df[bonus_col])
        
        # 一括でDataFrameに追加
        gap_df = pd.DataFrame(gap_data, index=self.df.index)
        self.df = pd.concat([self.df, gap_df], axis=1)
    
    def _calculate_gap_series(self, series: pd.Series) -> pd.Series:
        """
        ギャップ系列を計算
        
        Args:
            series (pd.Series): 入力系列
            
        Returns:
            pd.Series: ギャップ系列
        """
        gaps = np.zeros(len(series), dtype=int)
        last_appear_index = -1
        for i, appeared in enumerate(series):
            if appeared == 1:
                last_appear_index = i
                gaps[i] = 0
            else:
                gaps[i] = i - last_appear_index if last_appear_index != -1 else i + 1
        
        return pd.Series(gaps).shift(1).fillna(0)
    
    def _generate_combination_features(self):
        """
        組み合わせの構造的特徴量を生成
        """
        logger.info("組み合わせ特徴量を生成中...")
        
        main_cols = [col for col in self.df.columns if '本数字' in col and col != '本数字合計']
        main_numbers_df = self.df[main_cols].copy()

        combo_features = {}

        # ベクトル化による高速化
        # 奇数と偶数の比率
        odd_counts = main_numbers_df.apply(lambda row: sum(x % 2 == 1 for x in row if pd.notna(x)), axis=1)
        combo_features['odd_count'] = odd_counts
        combo_features['even_count'] = 6 - odd_counts
        combo_features['odd_even_ratio'] = odd_counts / 6

        # 低い数字(1-22)と高い数字(23-43)の分布
        low_counts = main_numbers_df.apply(lambda row: sum(x <= 22 for x in row if pd.notna(x)), axis=1)
        combo_features['low_count'] = low_counts
        combo_features['high_count'] = 6 - low_counts
        combo_features['low_high_ratio'] = low_counts / 6

        # 連続数字のペア数
        def count_consecutive_pairs(row):
            nums = sorted([x for x in row if pd.notna(x)])
            if len(nums) < 2:
                return 0
            return sum(1 for i in range(len(nums) - 1) if nums[i+1] - nums[i] == 1)
        combo_features['consecutive_pairs'] = main_numbers_df.apply(count_consecutive_pairs, axis=1)

        # 連続する3つの数字（トリプレット）
        def count_consecutive_triplets(row):
            nums = sorted([x for x in row if pd.notna(x)])
            if len(nums) < 3:
                return 0
            return sum(1 for i in range(len(nums) - 2) if nums[i+1] - nums[i] == 1 and nums[i+2] - nums[i+1] == 1)
        combo_features['consecutive_triplets'] = main_numbers_df.apply(count_consecutive_triplets, axis=1)

        # 一の位の分布
        combo_features['unique_ending_digits'] = main_numbers_df.apply(lambda row: len(set(int(x) % 10 for x in row if pd.notna(x))), axis=1)

        # 数字の分散（ばらつき）
        combo_features['numbers_std'] = main_numbers_df.apply(lambda row: np.std([x for x in row if pd.notna(x)]), axis=1)

        # 最大値と最小値の差
        def calculate_range(row):
            nums = [x for x in row if pd.notna(x)]
            return max(nums) - min(nums) if nums else 0
        combo_features['number_range'] = main_numbers_df.apply(calculate_range, axis=1)

        # 一括でDataFrameに追加
        combo_df = pd.DataFrame(combo_features, index=self.df.index)
        self.df = pd.concat([self.df, combo_df], axis=1)
    
    def _generate_set_ball_profiles(self):
        """
        セット球プロファイル特徴量を生成
        """
        logger.info("セット球プロファイル特徴量を生成中...")
        
        if 'セット球' not in self.df.columns:
            logger.warning("セット球の列が見つかりません。スキップします。")
            return
        
        # セット球ごとの統計量を計算
        set_ball_stats = {}
        
        for set_ball in self.df['セット球'].unique():
            if pd.isna(set_ball):
                continue
                
            subset = self.df[self.df['セット球'] == set_ball]
            
            stats = {
                'mean_sum': subset['本数字合計'].mean() if '本数字合計' in subset.columns else 0,
                'std_sum': subset['本数字合計'].std() if '本数字合計' in subset.columns else 0,
                'mean_odd_ratio': subset['odd_even_ratio'].mean() if 'odd_even_ratio' in subset.columns else 0,
                'mean_consecutive_pairs': subset['consecutive_pairs'].mean() if 'consecutive_pairs' in subset.columns else 0
            }
            set_ball_stats[set_ball] = stats
        
        # セット球プロファイル特徴量を生成
        profile_features = {}
        
        for set_ball, stats in set_ball_stats.items():
            profile_features[f'set_{set_ball}_mean_sum'] = self.df['セット球'].apply(lambda x: stats['mean_sum'] if x == set_ball else 0)
            profile_features[f'set_{set_ball}_std_sum'] = self.df['セット球'].apply(lambda x: stats['std_sum'] if x == set_ball else 0)
            profile_features[f'set_{set_ball}_mean_odd_ratio'] = self.df['セット球'].apply(lambda x: stats['mean_odd_ratio'] if x == set_ball else 0)

        # 一括でDataFrameに追加
        if profile_features:
            profile_df = pd.DataFrame(profile_features, index=self.df.index)
            self.df = pd.concat([self.df, profile_df], axis=1)
    
    def _handle_missing_values(self):
        """
        欠損値の処理
        """
        logger.info("欠損値を処理中...")
        
        # 数値列のNaN値を0で埋める
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        self.df[numeric_cols] = self.df[numeric_cols].fillna(0)
        
        # 無限大値を0で置換
        self.df.replace([np.inf, -np.inf], 0, inplace=True)
    
    def get_feature_columns(self) -> List[str]:
        """
        機械学習用の特徴量列名を取得
        
        Returns:
            List[str]: 特徴量列名のリスト
        """
        feature_prefixes = ['ma_', 'last_gap_', 'avg_gap_', 'std_gap_', 's_',
                          'odd_', 'even_', 'low_', 'high_', 'consecutive_',
                          'unique_', 'numbers_', 'number_', 'bonus_']
        
        feature_cols = []
        for col in self.df.columns:
            if any(col.startswith(prefix) for prefix in feature_prefixes):
                feature_cols.append(col)
        
        return feature_cols
    
    def get_target_columns(self) -> List[str]:
        """
        予測対象（ターゲット）列名を取得
        
        Returns:
            List[str]: ターゲット列名のリスト
        """
        return [f'is_appear_{n}' for n in range(1, 44)] + [f'is_bonus_appear_{n}' for n in range(1, 44)]
    
    def save_features(self, filepath: str):
        """
        特徴量データをファイルに保存
        
        Args:
            filepath (str): 保存先ファイルパス
        """
        self.df.to_csv(filepath, index=False, encoding='utf-8')
        logger.info(f"特徴量データを保存しました: {filepath}")
    
    def calculate_popularity_score(self, combination: Tuple[int, ...]) -> float:
        """
        組み合わせの人気スコアを計算（逆張り戦略用）
        
        Args:
            combination (Tuple[int, ...]): 数字の組み合わせ
            
        Returns:
            float: 人気スコア（高いほど人気）
        """
        score = 0.0
        combo_list = list(combination)
        
        # 誕生日バイアス: 1-31の数字が多い
        birthday_numbers = sum(1 for num in combo_list if 1 <= num <= 31)
        if birthday_numbers >= 4:
            score += 3.0
        
        # 連続数字バイアス
        sorted_combo = sorted(combo_list)
        for i in range(len(sorted_combo) - 2):
            if (sorted_combo[i+1] == sorted_combo[i] + 1 and 
                sorted_combo[i+2] == sorted_combo[i+1] + 1):
                score += 5.0
                break
        
        # キリ番バイアス: 10の倍数
        round_numbers = sum(1 for num in combo_list if num % 10 == 0)
        if round_numbers >= 2:
            score += 2.0
        
        # 両端バイアス: 1と43
        if 1 in combo_list and 43 in combo_list:
            score += 1.5
        
        # 一の位が同じ数字が多い
        endings = [num % 10 for num in combo_list]
        if len(set(endings)) <= 3:  # 6個の数字で一の位が3種類以下
            score += 2.0
        
        return score
