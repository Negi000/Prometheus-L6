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
        全ての特徴量生成を実行（高速化版）
        
        Returns:
            pd.DataFrame: 特徴量が追加されたデータフレーム
        """
        logger.info("特徴量生成を開始します...")
        
        try:
            # 基本特徴量
            self._generate_basic_flags()
            
            # 移動平均特徴量（必要最小限のウィンドウサイズのみ）
            self._generate_moving_averages_optimized()
            
            # ギャップ特徴量（高速版）
            self._generate_gap_features_optimized()
            
            # 組み合わせ特徴量（高速版）
            self._generate_combination_features()
            
            # セット球プロファイル特徴量（軽量版）
            if 'セット球' in self.df.columns:
                self._generate_set_ball_profiles()
            
            # 欠損値処理
            self._handle_missing_values()
            
            # メモリ最適化
            self._optimize_memory_usage()
            
            logger.info(f"特徴量生成が完了しました。最終的な列数: {len(self.df.columns)}")
            return self.df
            
        except Exception as e:
            logger.error(f"特徴量生成中にエラーが発生しました: {e}")
            return None
    
    def _optimize_memory_usage(self):
        """
        メモリ使用量を最適化
        """
        # データ型の最適化
        for col in self.df.columns:
            if self.df[col].dtype == 'int64':
                if self.df[col].min() >= 0 and self.df[col].max() <= 255:
                    self.df[col] = self.df[col].astype('uint8')
                elif self.df[col].min() >= -128 and self.df[col].max() <= 127:
                    self.df[col] = self.df[col].astype('int8')
                elif self.df[col].min() >= -32768 and self.df[col].max() <= 32767:
                    self.df[col] = self.df[col].astype('int16')
            elif self.df[col].dtype == 'float64':
                self.df[col] = self.df[col].astype('float32')
        
        # DataFrame断片化の解除
        self.df = self.df.copy()

    def _generate_basic_flags(self):
        """
        基本的な出現フラグを生成（高速化版）
        """
        logger.info("基本出現フラグを生成中...")
        
        # 本数字列の取得
        main_cols = [f'本数字{i}' for i in range(1, 7) if f'本数字{i}' in self.df.columns]
        
        if main_cols:
            # NumPy配列で高速処理
            main_numbers = self.df[main_cols].values
            
            # 全ての数字の出現フラグを一度に計算
            main_flags_data = {}
            for n in range(1, 44):
                # ベクトル化演算で高速化
                main_flags_data[f'is_appear_{n}'] = (main_numbers == n).any(axis=1).astype(np.int8)
        
        # ボーナス数字の出現フラグを生成
        bonus_flags_data = {}
        if 'ボーナス数字' in self.df.columns:
            bonus_numbers = self.df['ボーナス数字'].values
            for n in range(1, 44):
                bonus_flags_data[f'is_bonus_appear_{n}'] = (bonus_numbers == n).astype(np.int8)
        
        # 一括でDataFrameに追加（メモリ効率向上）
        if main_flags_data:
            main_flags_df = pd.DataFrame(main_flags_data, index=self.df.index)
            self.df = pd.concat([self.df, main_flags_df], axis=1)
        
        if bonus_flags_data:
            bonus_flags_df = pd.DataFrame(bonus_flags_data, index=self.df.index)
            self.df = pd.concat([self.df, bonus_flags_df], axis=1)
    
    def _generate_moving_averages_optimized(self):
        """
        移動平均特徴量を効率的に生成（データリーク防止のため1行シフト）
        """
        logger.info("移動平均特徴量を生成中...")
        
        # 全ての出現フラグ列を一括処理
        appear_cols = [col for col in self.df.columns if col.startswith('is_appear_') or col.startswith('is_bonus_appear_')]
        
        if not appear_cols:
            logger.warning("出現フラグ列が見つかりません。移動平均をスキップします。")
            return
        
        # パフォーマンス向上のため、必要な移動平均のみ生成
        windows = [5, 25, 75]
        ma_data = {}
        
        for col in appear_cols:
            series = self.df[col]
            # 移動平均を一括計算
            for window in windows:
                if 'bonus' in col:
                    ma_data[f'ma_bonus_{window}_{col.split("_")[-1]}'] = series.rolling(window=window, min_periods=1).mean().shift(1)
                else:
                    ma_data[f'ma_{window}_{col.split("_")[-1]}'] = series.rolling(window=window, min_periods=1).mean().shift(1)
        
        # 一括でDataFrameに追加
        if ma_data:
            ma_df = pd.DataFrame(ma_data, index=self.df.index)
            self.df = pd.concat([self.df, ma_df], axis=1)

    def _generate_gap_features_optimized(self):
        """
        ギャップ特徴量を効率的に生成（データリーク防止）
        """
        logger.info("ギャップ特徴量を生成中...")
        
        # 出現フラグ列を一括取得
        appear_cols = [col for col in self.df.columns if col.startswith('is_appear_') or col.startswith('is_bonus_appear_')]
        
        if not appear_cols:
            logger.warning("出現フラグ列が見つかりません。ギャップ特徴量をスキップします。")
            return
        
        gap_data = {}
        
        # NumPy配列で高速処理
        appear_data = self.df[appear_cols].values
        
        for idx, col in enumerate(appear_cols):
            series_data = appear_data[:, idx]
            gaps = self._calculate_gap_series_fast(series_data)
            
            if 'bonus' in col:
                gap_data[f'gap_bonus_{col.split("_")[-1]}'] = gaps
            else:
                gap_data[f'gap_{col.split("_")[-1]}'] = gaps
        
        # 一括でDataFrameに追加
        if gap_data:
            gap_df = pd.DataFrame(gap_data, index=self.df.index)
            self.df = pd.concat([self.df, gap_df], axis=1)
    
    def _calculate_gap_series_fast(self, series_data: np.ndarray) -> pd.Series:
        """
        ギャップ系列を高速計算（NumPy版）
        
        Args:
            series_data (np.ndarray): 入力系列
            
        Returns:
            pd.Series: ギャップ系列
        """
        gaps = np.zeros(len(series_data), dtype=np.int16)
        last_appear_index = -1
        
        for i in range(len(series_data)):
            if series_data[i] == 1:
                last_appear_index = i
                gaps[i] = 0
            else:
                gaps[i] = i - last_appear_index if last_appear_index != -1 else i + 1
        
        # 1行シフトしてデータリークを防止
        gaps_shifted = np.roll(gaps, 1)
        gaps_shifted[0] = 0
        
        return pd.Series(gaps_shifted, dtype=np.int16)
    
    def _generate_combination_features(self):
        """
        組み合わせの構造的特徴量を生成（高速化版）
        """
        logger.info("組み合わせ特徴量を生成中...")
        
        main_cols = [col for col in self.df.columns if '本数字' in col and col != '本数字合計']
        main_numbers_df = self.df[main_cols].copy()

        # NumPy配列で高速処理
        main_numbers = main_numbers_df.values
        
        combo_features = {}

        # ベクトル化による高速計算
        # 奇数と偶数の比率
        odd_counts = np.sum(main_numbers % 2 == 1, axis=1)
        combo_features['odd_count'] = odd_counts.astype(np.int8)
        combo_features['even_count'] = (6 - odd_counts).astype(np.int8)
        combo_features['odd_even_ratio'] = (odd_counts / 6).astype(np.float32)

        # 低い数字(1-22)と高い数字(23-43)の分布
        low_counts = np.sum(main_numbers <= 22, axis=1)
        combo_features['low_count'] = low_counts.astype(np.int8)
        combo_features['high_count'] = (6 - low_counts).astype(np.int8)
        combo_features['low_high_ratio'] = (low_counts / 6).astype(np.float32)

        # 連続数字のペア数（ベクトル化版）
        consecutive_pairs = np.zeros(len(main_numbers), dtype=np.int8)
        for i in range(len(main_numbers)):
            nums = np.sort(main_numbers[i][~np.isnan(main_numbers[i])])
            if len(nums) >= 2:
                consecutive_pairs[i] = np.sum(np.diff(nums) == 1)
        combo_features['consecutive_pairs'] = consecutive_pairs

        # 連続する3つの数字（トリプレット）
        consecutive_triplets = np.zeros(len(main_numbers), dtype=np.int8)
        for i in range(len(main_numbers)):
            nums = np.sort(main_numbers[i][~np.isnan(main_numbers[i])])
            if len(nums) >= 3:
                diffs = np.diff(nums)
                consecutive_triplets[i] = np.sum((diffs[:-1] == 1) & (diffs[1:] == 1))
        combo_features['consecutive_triplets'] = consecutive_triplets

        # 一の位の分布
        unique_endings = np.zeros(len(main_numbers), dtype=np.int8)
        for i in range(len(main_numbers)):
            nums = main_numbers[i][~np.isnan(main_numbers[i])]
            if len(nums) > 0:
                endings = nums.astype(int) % 10
                unique_endings[i] = len(np.unique(endings))
        combo_features['unique_ending_digits'] = unique_endings

        # 数字の分散（ばらつき）
        combo_features['numbers_std'] = np.nanstd(main_numbers, axis=1).astype(np.float32)

        # 最大値と最小値の差
        number_ranges = np.zeros(len(main_numbers), dtype=np.int8)
        for i in range(len(main_numbers)):
            nums = main_numbers[i][~np.isnan(main_numbers[i])]
            if len(nums) > 0:
                number_ranges[i] = np.max(nums) - np.min(nums)
        combo_features['number_range'] = number_ranges

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
