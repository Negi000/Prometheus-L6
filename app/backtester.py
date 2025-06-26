"""
バックテストエンジン (backtester.py)
適応学習・フィードバック機能を持つバックテストシステム
"""

import pandas as pd
import numpy as np
import logging
import pickle
import os
from datetime import datetime
from typing import Dict, List, Tuple, Any
from itertools import combinations

# モデルのインポート
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.linear_model import LogisticRegression 
import xgboost as xgb
import lightgbm as lgb


logger = logging.getLogger(__name__)

class Backtester:
    def __init__(self, model_type='xgboost', model_params=None):
        """
        バックテスターの初期化
        
        Args:
            model_type (str): 使用するモデルタイプ
            model_params (dict): モデルパラメータ
        """
        self.model_type = model_type
        self.model_params = model_params if model_params else {}
        
        # パターン補正システムの初期化
        self.pattern_correction = PatternCorrection()
        
        # デフォルトパラメータの設定（CPU最適化）
        self.default_params = {
            'xgboost': {
                'n_estimators': 50,  # 精度をあまり落とさずに高速化
                'max_depth': 4,      # 計算量削減
                'learning_rate': 0.2,  # 学習率上げて収束を早める
                'random_state': 42, 
                'verbosity': 0,
                'n_jobs': -1,        # 並列処理
                'tree_method': 'hist', # 高速な学習手法
                'subsample': 0.8,    # サブサンプリングで高速化
                'colsample_bytree': 0.8
            },
            'lightgbm': {
                'n_estimators': 50,
                'max_depth': 4,
                'learning_rate': 0.2,
                'random_state': 42, 
                'verbosity': -1,
                'min_child_samples': 10,  # 軽量化
                'n_jobs': -1,
                'boosting_type': 'gbdt',
                'objective': 'regression',
                'subsample': 0.8,
                'colsample_bytree': 0.8
            },
            'random_forest': {
                'n_estimators': 50,  # 木の数を削減
                'max_depth': 8,      # 深さ制限
                'random_state': 42, 
                'n_jobs': -1,
                'min_samples_split': 10,  # 分割条件を緩くして高速化
                'min_samples_leaf': 5
            },
            'logistic_regression': {
                'solver': 'liblinear', 
                'random_state': 42,
                'max_iter': 100  # 反復回数制限
            }
        }
        
        # LOTO6の賞金設定
        self.prizes = {
            1: 200000000,  # 1等: 2億円
            2: 10000000,   # 2等: 1000万円
            3: 300000,     # 3等: 30万円
            4: 6800,       # 4等: 6800円
            5: 1000        # 5等: 1000円
        }
        
        logger.info(f"バックテスターを初期化しました。モデル: {model_type}")
    
    def run(self, df_features: pd.DataFrame, start_draw: int, end_draw: int, 
            window_size: int = 100, purchase_count: int = 20, 
            detailed_log: bool = False, enable_feedback: bool = False, 
            progress_callback=None) -> Tuple[Any, List[Dict]]:
        """
        バックテストの実行（自己補正フィードバックループ機能付き）
        
        Args:
            df_features (pd.DataFrame): 特徴量データ
            start_draw (int): バックテスト開始回号
            end_draw (int): バックテスト終了回号
            window_size (int): 学習ウィンドウサイズ
            purchase_count (int): 購入口数
            detailed_log (bool): 詳細ログを記録するかどうか
            enable_feedback (bool): 予測誤差を学習する自己補正機能を有効にするか
            progress_callback (callable): 進行状況を報告するコールバック関数
            
        Returns:
            Tuple[Any, List[Dict]]: (最終学習済みモデル, パフォーマンスログ)
        """
        logger.info(f"バックテスト開始: 第{start_draw}回〜第{end_draw}回, ウィンドウ={window_size}, モデル={self.model_type}, 自己補正フィードバック: {'有効' if enable_feedback else '無効'}")
        
        df_run = df_features.copy() # バックテスト実行中のデータフレーム

        # 必要な列の存在確認
        if '第何回' not in df_run.columns:
            raise ValueError("特徴量データに「第何回」列が存在しません。データの生成方法を確認してください。")
        
        # データの確認
        available_draws = sorted(df_run['第何回'].unique())
        if not any(available_draws):
            logger.error("特徴量データに有効な「第何回」データがありません。バックテストを中止します。")
            return None, []
        min_available_draw = min(available_draws)
        max_available_draw = max(available_draws)
        logger.info(f"利用可能な回数範囲: 第{min_available_draw}回〜第{max_available_draw}回")
        
        performance_log = []
        model = None # 最終モデルを格納
        
        # 特徴量とターゲット列の定義（1-43の43個の数字）
        target_cols = [f'is_appear_{n}' for n in range(1, 44)] + [f'is_bonus_appear_{n}' for n in range(1, 44) if f'is_bonus_appear_{n}' in df_run.columns]
        
        # 自己補正フィードバック用の列を初期化
        if enable_feedback:
            logger.info("自己補正フィードバック機能を有効化。誤差特徴量列を初期化します。")
            self.error_cols = [f'error_{t}' for t in target_cols]
            self.error_ma_cols = [f'error_ma5_{t}' for t in target_cols]
            for col in self.error_cols + self.error_ma_cols:
                df_run[col] = 0.0

        feature_cols = self._get_feature_columns(df_run, enable_feedback)
        
        logger.info(f"使用特徴量数: {len(feature_cols)}")
        logger.info(f"ターゲット列数: {len(target_cols)}")
        
        # ターゲット列の存在確認
        missing_target_cols = [col for col in target_cols if col not in df_run.columns]
        if missing_target_cols:
            logger.warning(f"不足しているターゲット列: {len(missing_target_cols)}個 - {missing_target_cols[:5]}")
        
        # 実際の当選番号列の確認
        main_number_cols = [col for col in df_run.columns if '本数字' in col and col != '本数字合計']
        bonus_col = 'ボーナス数字' if 'ボーナス数字' in df_run.columns else None
        
        # バックテストのメインループ
        total_draws = end_draw - start_draw + 1
        current_draw_index = 0
        
        for draw_id in range(start_draw, end_draw + 1):
            current_draw_index += 1
            
            # 進行状況の報告
            if progress_callback:
                progress = current_draw_index / total_draws
                progress_callback(progress, draw_id, total_draws)
            
            try:
                if draw_id - window_size < min_available_draw:
                    logger.warning(f"第{draw_id}回のテストをスキップ: 学習に必要なデータが不足しています。")
                    continue
                
                if draw_id > max_available_draw:
                    logger.warning(f"第{draw_id}回のテストをスキップ: 実際の結果データが利用できません。")
                    break

                # --- 学習データ準備 ---
                train_start_draw = draw_id - window_size
                train_end_draw = draw_id - 1
                
                # --- 自己補正特徴量の計算 ---
                if enable_feedback:
                    # 学習データ期間内の誤差の移動平均を計算
                    # shift(1)を使い、当日の誤差を含めずに計算することで未来データ参照を防ぐ
                    for err_col, ma_col in zip(self.error_cols, self.error_ma_cols):
                        df_run[ma_col] = df_run[err_col].shift(1).rolling(window=5, min_periods=1).mean()

                train_mask = (df_run['第何回'] >= train_start_draw) & (df_run['第何回'] <= train_end_draw)
                train_df = df_run[train_mask].copy().fillna(0) # fillna for MA at the beginning
                
                if len(train_df) < 10:
                    logger.warning(f"第{draw_id}回: 学習データが不足しています ({len(train_df)}件)")
                    continue
                
                X_train = train_df[feature_cols]
                Y_train = train_df[target_cols]
                
                model = self._train_model(X_train, Y_train, enable_feature_selection=True)
                
                # --- 予測 ---
                # 特徴量選択が行われている場合は同じ特徴量を使用
                if hasattr(model, 'selected_features_'):
                    X_pred = X_train[model.selected_features_]
                else:
                    X_pred = X_train
                
                # is_appear_* と is_bonus_appear_* の確率を予測
                predictions = model.predict(X_pred)
                predicted_probabilities = predictions[0] # feedback loopで利用するため定義

                # 予測結果をカラム名と対応付ける
                pred_df = pd.DataFrame(predictions, columns=Y_train.columns)

                # mainとbonusのターゲットカラムを特定
                main_target_cols = [c for c in Y_train.columns if 'is_appear' in c and 'bonus' not in c]
                bonus_target_cols = [c for c in Y_train.columns if 'is_bonus_appear' in c]

                # 確率を辞書として抽出
                main_probs_map = pred_df[main_target_cols].iloc[0].to_dict()
                bonus_probs_map = pred_df[bonus_target_cols].iloc[0].to_dict()

                # --- ポートフォリオ生成 ---
                logger.info(f"[{draw_id}] Generating portfolio...")
                predicted_portfolio = self._generate_portfolio_from_probabilities(
                    main_probs_map,
                    bonus_probs_map,
                    purchase_count=purchase_count
                )

                # --- 結果の評価 ---
                actual_mask = df_run['第何回'] == draw_id
                actual_data = df_run[actual_mask]
                
                if len(actual_data) == 0:
                    logger.warning(f"第{draw_id}回: 実際の結果データが見つかりません")
                    continue

                if enable_feedback and not actual_data.empty:
                    actual_results = actual_data[target_cols].values[0]
                    error = actual_results - predicted_probabilities
                    # 計算した誤差を現在の回の行に保存 -> 次の回の学習で使われる
                    df_run.loc[actual_mask, self.error_cols] = error
                
                actual_numbers = self._get_actual_numbers(actual_data.iloc[0])
                
                winnings, hits_detail = self._calculate_winnings(
                    predicted_portfolio, actual_numbers
                )
                
                cost = purchase_count * 200
                profit = winnings - cost
                
                # 詳細分析：予想vs実際の比較（軽量化）
                prediction_analysis = self._analyze_predictions_fast(predicted_portfolio, actual_numbers) if detailed_log else None
                
                log_entry = {
                    'draw_id': draw_id,
                    'profit': profit,
                    'winnings': winnings,
                    'cost': cost,
                    'hits_detail': hits_detail,
                    'portfolio_size': len(predicted_portfolio),
                    'actual_numbers': actual_numbers,
                    'predicted_portfolio': predicted_portfolio,  # 全ての予想チケットを保存
                    'prediction_analysis': prediction_analysis,
                    'model_accuracy': self._calculate_prediction_accuracy(
                        predicted_probabilities, actual_numbers, len(target_cols)
                    )
                }
                performance_log.append(log_entry)
                
                if draw_id % 50 == 0:
                    avg_profit = np.mean([log['profit'] for log in performance_log[-50:]])
                    logger.info(f"第{draw_id}回完了 - 直近50回平均損益: {avg_profit:.0f}円")
                
            except Exception as e:
                logger.error(f"第{draw_id}回のバックテスト中にエラーが発生: {e}", exc_info=True)
                continue
        
        final_model = model # ループの最後のモデルを最終モデルとする
        
        if performance_log:
            total_profit = sum(log['profit'] for log in performance_log)
            hit_rates = self._calculate_hit_rates(performance_log)
            logger.info(f"バックテスト完了: 総損益={total_profit:.0f}円, 3等以上的中率={hit_rates['hit_rate_3']:.1%}")
        else:
            logger.warning("バックテストが実行されませんでした。ログは空です。")

        return final_model, performance_log

    def run_continuous_learning(self, df_features: pd.DataFrame, base_model: Any,
                               start_draw: int, end_draw: int, window_size: int = 100, 
                               purchase_count: int = 20, detailed_log: bool = False, 
                               enable_feedback: bool = False, learning_rate_factor: float = 0.8,
                               progress_callback=None) -> Tuple[Any, List[Dict]]:
        """
        継続学習の実行（データリーケージを防ぐ正しいパターン学習）
        
        Args:
            df_features (pd.DataFrame): 特徴量データ
            base_model (Any): 継続学習の基となる既存モデル
            start_draw (int): 継続学習開始回号
            end_draw (int): 継続学習終了回号
            window_size (int): 学習ウィンドウサイズ
            purchase_count (int): 購入口数
            detailed_log (bool): 詳細ログを記録するかどうか
            enable_feedback (bool): 予測誤差を学習する自己補正機能を有効にするか
            learning_rate_factor (float): 学習率調整係数（元の学習率に対する倍率）
            
        Returns:
            Tuple[Any, List[Dict]]: (更新された学習済みモデル, パフォーマンスログ)
        """
        logger.info(f"継続学習開始: 第{start_draw}回〜第{end_draw}回 (パターン学習モード)")
        logger.info(f"パラメータ: ウィンドウ={window_size}, モデル={self.model_type}, 学習率係数={learning_rate_factor}")
        
        df_run = df_features.copy()
        
        # 必要な列の存在確認
        if '第何回' not in df_run.columns:
            raise ValueError("特徴量データに「第何回」列が存在しません。")
        
        # データの確認
        available_draws = sorted(df_run['第何回'].unique())
        min_available_draw = min(available_draws)
        max_available_draw = max(available_draws)
        logger.info(f"利用可能な回数範囲: 第{min_available_draw}回〜第{max_available_draw}回")
        
        performance_log = []
        
        # 既存モデルをコピー
        import copy
        model = copy.deepcopy(base_model)
        
        # 既存のパターン補正値を継承（もし存在する場合）
        if hasattr(base_model, 'pattern_correction'):
            self.pattern_correction = copy.deepcopy(base_model.pattern_correction)
        
        # 特徴量とターゲット列の定義（1-43の43個の数字）
        target_cols = [f'is_appear_{n}' for n in range(1, 44)] + [f'is_bonus_appear_{n}' for n in range(1, 44) if f'is_bonus_appear_{n}' in df_run.columns]
        feature_cols = self._get_feature_columns(df_run, enable_feedback)
        
        # 時系列順に学習を実行
        total_draws = end_draw - start_draw + 1
        for i, current_draw in enumerate(range(start_draw, end_draw + 1)):
            if progress_callback:
                progress = (i + 1) / total_draws
                progress_callback(progress, current_draw, total_draws)
            
            # ⚠️ 重要: 学習データは必ず予測対象より前のデータのみ使用
            # これによりデータリーケージを完全に防ぐ
            train_start = max(current_draw - window_size, min_available_draw)
            train_end = current_draw - 1  # 予測対象回の直前まで
            
            # 学習データの準備（データリーケージなし）
            train_data = df_run[
                (df_run['第何回'] >= train_start) & 
                (df_run['第何回'] <= train_end)
            ].copy()
            
            if len(train_data) < 10:  # 最低限のデータが必要
                logger.warning(f"第{current_draw}回: 学習データが不足 ({len(train_data)}件)")
                continue
            
            # 特徴量とターゲットの準備
            X_train = train_data[feature_cols].fillna(0)
            y_train = train_data[target_cols].fillna(0)
            
            # モデルの増分学習実行
            try:
                if hasattr(model, 'fit'):
                    model.fit(X_train, y_train)
                logger.debug(f"第{current_draw}回の学習完了")
            except Exception as e:
                logger.warning(f"第{current_draw}回の学習でエラー: {e}")
                continue
            
            # 予測実行（補正値適用）
            try:
                # 現在回のデータを取得
                current_data = df_run[df_run['第何回'] == current_draw]
                if len(current_data) == 0:
                    logger.warning(f"第{current_draw}回のデータが見つかりません")
                    continue
                
                X_current = current_data[feature_cols].fillna(0)
                
                # 基本予測
                if hasattr(model, 'predict_proba'):
                    raw_prediction = model.predict_proba(X_current)
                else:
                    raw_prediction = model.predict(X_current)
                
                # 予測結果の形状をデバッグ
                logger.debug(f"第{current_draw}回: 生予測結果の形状={raw_prediction.shape if hasattr(raw_prediction, 'shape') else type(raw_prediction)}")
                
                # 予測結果の形状を安全に処理
                if isinstance(raw_prediction, list):
                    raw_prediction = np.array(raw_prediction)
                
                if len(raw_prediction.shape) > 1:
                    if raw_prediction.shape[0] > 0:
                        raw_probabilities = raw_prediction[0]  # 1回分のみ
                        logger.debug(f"第{current_draw}回: 多次元予測から1次元抽出、サイズ={len(raw_probabilities)}")
                    else:
                        logger.warning(f"第{current_draw}回: 予測結果が空です")
                        continue
                else:
                    raw_probabilities = raw_prediction
                    logger.debug(f"第{current_draw}回: 1次元予測結果、サイズ={len(raw_probabilities)}")
                
                # 配列サイズの安全チェック
                if len(raw_probabilities) != 43:
                    logger.info(f"第{current_draw}回: 予測結果のサイズ調整 ({len(raw_probabilities)}次元 → 43次元)")
                    # サイズを43に統一
                    if len(raw_probabilities) < 43:
                        # 不足分を均等確率で埋める
                        padded_probs = np.ones(43) / 43
                        padded_probs[:len(raw_probabilities)] = raw_probabilities
                        raw_probabilities = padded_probs
                    else:
                        # 43次元を超える場合は切り捨て
                        raw_probabilities = raw_probabilities[:43]
                
                # セット球情報を取得
                set_ball = current_data.iloc[0].get('セット球', 'A')
                
                # 補正適用（本数字のみ、43次元）
                corrected_probabilities = self.pattern_correction.apply_corrections(
                    raw_probabilities, set_ball
                )
                
                # 確率の正規化
                corrected_probabilities = np.maximum(corrected_probabilities, 0.001)
                corrected_probabilities = corrected_probabilities / np.sum(corrected_probabilities)
                
                # 予測数字を生成
                predicted_combinations = self._generate_numbers_from_probabilities(
                    corrected_probabilities, min(purchase_count, 5)  # 最大5組合せに制限
                )
                
                if len(predicted_combinations) == 0:
                    logger.warning(f"第{current_draw}回: 数字組み合わせの生成に失敗")
                    continue
                
                predicted_numbers = predicted_combinations[0]  # 最初の組み合わせ
                
                # 実際の当選番号を取得
                actual_numbers = []
                for j in range(1, 7):  # 本数字6個
                    col_name = f'本数字{j}'
                    if col_name in current_data.columns:
                        actual_numbers.append(int(current_data.iloc[0][col_name]))
                
                if len(actual_numbers) == 6:
                    # パターン分析実行（データリーケージなし）
                    pattern_analysis = self.pattern_correction.analyze_prediction_pattern(
                        predicted_numbers, actual_numbers, set_ball
                    )
                    
                    # パターン補正値を更新（結果の暗記ではなく傾向の学習）
                    self.pattern_correction.update_corrections(pattern_analysis)
                    
                    # 的中数計算
                    hit_count = len(set(predicted_numbers) & set(actual_numbers))
                    
                    # パフォーマンスログに記録
                    result_log = {
                        'draw': current_draw,
                        'predicted_numbers': predicted_numbers,
                        'actual_numbers': actual_numbers,
                        'hit_count': hit_count,
                        'profit': self._calculate_profit(hit_count, purchase_count),
                        'pattern_corrections': self.pattern_correction.get_correction_summary()
                    }
                    
                    if detailed_log:
                        result_log.update({
                            'raw_probabilities': raw_probabilities.tolist()[:43],
                            'corrected_probabilities': corrected_probabilities.tolist(),
                            'pattern_analysis': pattern_analysis,
                            'set_ball': set_ball
                        })
                    
                    performance_log.append(result_log)
                    
                    logger.debug(f"第{current_draw}回: 予測={predicted_numbers}, 実際={actual_numbers}, 的中={hit_count}")
                
            except Exception as e:
                logger.error(f"第{current_draw}回の予測・分析でエラー: {e}")
                continue
        
        # 学習済みモデルにパターン補正を保存
        model.pattern_correction = self.pattern_correction
        
        logger.info(f"継続学習完了: {len(performance_log)}回分の学習・パターン分析を実行")
        return model, performance_log
        
        logger.info(f"継続学習: 使用特徴量数={len(feature_cols)}, ターゲット列数={len(target_cols)}")
        
        # 継続学習のメインループ
        total_draws = end_draw - start_draw + 1
        current_draw_index = 0
        
        for draw_id in range(start_draw, end_draw + 1):
            current_draw_index += 1
            
            # 進行状況の報告
            if progress_callback:
                progress = current_draw_index / total_draws
                progress_callback(progress, draw_id, total_draws)
            
            try:
                if draw_id - window_size < min_available_draw:
                    logger.warning(f"第{draw_id}回のテストをスキップ: 学習に必要なデータが不足しています。")
                    continue
                
                if draw_id > max_available_draw:
                    logger.warning(f"第{draw_id}回のテストをスキップ: 実際の結果データが利用できません。")
                    break

                # --- 継続学習データ準備 ---
                train_start_draw = draw_id - window_size
                train_end_draw = draw_id - 1
                
                # 自己補正特徴量の計算
                if enable_feedback:
                    for err_col, ma_col in zip(self.error_cols, self.error_ma_cols):
                        df_run[ma_col] = df_run[err_col].shift(1).rolling(window=5, min_periods=1).mean()

                train_mask = (df_run['第何回'] >= train_start_draw) & (df_run['第何回'] <= train_end_draw)
                train_df = df_run[train_mask].copy().fillna(0)
                
                if len(train_df) < 10:
                    logger.warning(f"第{draw_id}回: 継続学習データが不足しています ({len(train_df)}件)")
                    continue
                
                X_train = train_df[feature_cols]
                Y_train = train_df[target_cols]
                
                # 継続学習（増分学習）
                model = self._continue_training(model, X_train, Y_train)
                
                # --- 予測と評価（通常のバックテストと同様） ---
                test_mask = df_run['第何回'] == draw_id
                test_df = df_run[test_mask].copy().fillna(0)
                
                if len(test_df) == 0:
                    continue
                
                X_test = test_df[feature_cols]
                Y_test = test_df[target_cols]
                
                # 特徴量選択が行われている場合は同じ特徴量を使用
                if hasattr(model, 'selected_features_'):
                    X_test_pred = X_test[model.selected_features_]
                else:
                    X_test_pred = X_test
                
                predictions = model.predict(X_test_pred)
                predicted_probabilities = predictions[0]

                # 予測結果をカラム名と対応付ける
                pred_df = pd.DataFrame(predictions, columns=Y_test.columns)

                # mainとbonusのターゲットカラムを特定
                main_target_cols = [c for c in Y_test.columns if 'is_appear' in c and 'bonus' not in c]
                bonus_target_cols = [c for c in Y_test.columns if 'is_bonus_appear' in c]

                # 確率を辞書として抽出
                main_probs_map = pred_df[main_target_cols].iloc[0].to_dict()
                bonus_probs_map = pred_df[bonus_target_cols].iloc[0].to_dict()

                # ポートフォリオ生成
                predicted_portfolio = self._generate_portfolio_from_probabilities(
                    main_probs_map,
                    bonus_probs_map,
                    purchase_count=purchase_count
                )

                # 結果の評価
                actual_data = df_run[test_mask]
                
                if enable_feedback and not actual_data.empty:
                    actual_results = actual_data[target_cols].values[0]
                    error = actual_results - predicted_probabilities
                    df_run.loc[test_mask, self.error_cols] = error
                
                actual_numbers = self._get_actual_numbers(actual_data.iloc[0])
                
                try:
                    winnings, hits_detail = self._calculate_winnings(
                        predicted_portfolio, actual_numbers
                    )
                except Exception as e:
                    logger.error(f"第{draw_id}回: 当選金額計算エラー: {e}")
                    winnings, hits_detail = 0, {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
                
                cost = purchase_count * 200
                profit = winnings - cost
                
                # hits_detailの安全性確認
                if not isinstance(hits_detail, dict):
                    hits_detail = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
                
                log_entry = {
                    'draw_id': draw_id,
                    'profit': profit,
                    'winnings': winnings,
                    'cost': cost,
                    'hits_detail': hits_detail,
                    'portfolio_size': len(predicted_portfolio),
                    'actual_numbers': actual_numbers,
                    'predicted_portfolio': predicted_portfolio,  # 全ての予想チケットを保存
                    'model_accuracy': self._calculate_prediction_accuracy(
                        predicted_probabilities, actual_numbers, len(target_cols)
                    ),
                    'is_continuous_learning': True
                }
                performance_log.append(log_entry)
                
                if draw_id % 50 == 0:
                    avg_profit = np.mean([log['profit'] for log in performance_log[-50:]])
                    logger.info(f"継続学習 第{draw_id}回完了 - 直近50回平均損益: {avg_profit:.0f}円")
                
            except Exception as e:
                logger.error(f"継続学習 第{draw_id}回でエラーが発生: {e}", exc_info=True)
                continue
        
        if performance_log:
            total_profit = sum(log['profit'] for log in performance_log)
            hit_rates = self._calculate_hit_rates(performance_log)
            logger.info(f"継続学習完了: 総損益={total_profit:.0f}円, 3等以上的中率={hit_rates['hit_rate_3']:.1%}")
        else:
            logger.warning("継続学習が実行されませんでした。ログは空です。")

        return model, performance_log

    def _get_feature_columns(self, df: pd.DataFrame, enable_feedback: bool = False) -> List[str]:
        """
        機械学習用の特徴量列を取得（自己補正特徴量も考慮）
        """
        feature_prefixes = ['ma_', 'last_gap_', 'avg_gap_', 'std_gap_', 's_',
                           'odd_', 'even_', 'low_', 'high_', 'consecutive_',
                           'unique_', 'numbers_', 'number_', 'bonus_']
        
        feature_cols = [col for col in df.columns if any(col.startswith(prefix) for prefix in feature_prefixes)]
        
        if enable_feedback and hasattr(self, 'error_ma_cols'):
            feature_cols.extend(self.error_ma_cols)
            logger.info(f"自己補正フィードバック有効: {len(self.error_ma_cols)}個の誤差特徴量を追加。")

        # 重複を除外して返す
        return list(dict.fromkeys(feature_cols))
    
    def _select_features_fast(self, X_train: pd.DataFrame, Y_train: pd.DataFrame, max_features: int = 50) -> List[str]:
        """
        高速な特徴量選択（分散ベース + 相関ベース）
        
        Args:
            X_train: 訓練データの特徴量
            Y_train: 訓練データのターゲット
            max_features: 選択する最大特徴量数
            
        Returns:
            選択された特徴量の列名リスト
        """
        # 分散が極めて小さい特徴量を除外
        variance_threshold = 0.01
        feature_variances = X_train.var()
        # NaN値を0で置換
        feature_variances = feature_variances.fillna(0)
        high_variance_features = feature_variances[feature_variances > variance_threshold].index.tolist()
        
        if len(high_variance_features) <= max_features:
            return high_variance_features
        
        # 相関による特徴量選択（高速版）
        X_subset = X_train[high_variance_features]
        
        # ターゲットとの相関を計算（各ターゲットの平均）
        correlations = []
        for col in X_subset.columns:
            corr_sum = 0
            valid_corr_count = 0
            for target_col in Y_train.columns:
                # 無限大値とNaN値を除外
                X_clean = X_subset[col].replace([np.inf, -np.inf], np.nan).dropna()
                Y_clean = Y_train[target_col].replace([np.inf, -np.inf], np.nan).dropna()
                
                # 共通のインデックスでデータを揃える
                common_index = X_clean.index.intersection(Y_clean.index)
                if len(common_index) > 1:
                    X_common = X_clean.loc[common_index]
                    Y_common = Y_clean.loc[common_index]
                    
                    # 標準偏差が0でない場合のみ相関を計算
                    if X_common.std() > 0 and Y_common.std() > 0:
                        corr = abs(X_common.corr(Y_common))
                        if not np.isnan(corr):
                            corr_sum += corr
                            valid_corr_count += 1
            
            # 有効な相関の平均を計算
            avg_corr = corr_sum / valid_corr_count if valid_corr_count > 0 else 0
            correlations.append((col, avg_corr))
        
        # 相関の高い順にソート
        correlations.sort(key=lambda x: x[1], reverse=True)
        
        # 上位の特徴量を選択
        selected_features = [col for col, _ in correlations[:max_features]]
        
        logger.info(f"特徴量選択: {len(X_train.columns)} -> {len(selected_features)} 個")
        return selected_features

    def _train_model(self, X_train: pd.DataFrame, Y_train: pd.DataFrame, enable_feature_selection: bool = True):
        """
        選択されたモデルタイプに基づいてモデルを学習（最適化版）
        """
        # 特徴量選択による高速化
        if enable_feature_selection and len(X_train.columns) > 50:
            selected_features = self._select_features_fast(X_train, Y_train, max_features=50)
            X_train = X_train[selected_features]
            logger.info(f"特徴量選択により {len(selected_features)} 個の特徴量を使用")
        
        model_type = self.model_type.lower()

        # ベースモデルの定義
        rf_params = self.default_params.get('random_forest', {})
        lgbm_params = self.default_params.get('lightgbm', {})
        xgb_params = self.default_params.get('xgboost', {})
        lr_params = self.default_params.get('logistic_regression', {})

        # モデルの作成と学習
        if model_type == 'xgboost':
            model = MultiOutputRegressor(xgb.XGBRegressor(**xgb_params))
        elif model_type == 'lightgbm':
            model = MultiOutputRegressor(lgb.LGBMRegressor(**lgbm_params))
        elif model_type == 'random_forest':
            model = RandomForestRegressor(**rf_params)  # ネイティブでマルチアウトプット対応
        elif model_type == 'logistic_regression':
            model = MultiOutputRegressor(LogisticRegression(**lr_params))
        elif model_type == 'ensemble_balanced':
            # バランス型: RandomForest + LightGBM
            rf_model = RandomForestRegressor(**rf_params)
            lgbm_model = lgb.LGBMRegressor(**lgbm_params)
            estimators = [('rf', rf_model), ('lgbm', lgbm_model)]
            model = MultiOutputRegressor(VotingRegressor(estimators=estimators))
        elif model_type == 'ensemble_aggressive':
            # アグレッシブ型: LightGBM + XGBoost
            lgbm_model = lgb.LGBMRegressor(**lgbm_params)
            xgb_model = xgb.XGBRegressor(**xgb_params)
            estimators = [('lgbm', lgbm_model), ('xgb', xgb_model)]
            model = MultiOutputRegressor(VotingRegressor(estimators=estimators))
        elif model_type in ['ensemble_full', 'ensemble']:
            # フルアンサンブル: RandomForest + LightGBM + XGBoost
            rf_model = RandomForestRegressor(**rf_params)
            lgbm_model = lgb.LGBMRegressor(**lgbm_params)
            xgb_model = xgb.XGBRegressor(**xgb_params)
            estimators = [('rf', rf_model), ('lgbm', lgbm_model), ('xgb', xgb_model)]
            model = MultiOutputRegressor(VotingRegressor(estimators=estimators))
        else:
            raise ValueError(f"サポートされていないモデルタイプ: {self.model_type}")
        
        # メモリ効率的な学習
        try:
            model.fit(X_train, Y_train)
            # 選択された特徴量をモデルに記録
            if enable_feature_selection:
                model.selected_features_ = X_train.columns.tolist()
            return model
        except Exception as e:
            logger.error(f"モデル学習中にエラー: {e}")
            # フォールバック: より軽量なモデルで再試行
            logger.info("軽量モデルで再試行します...")
            fallback_model = MultiOutputRegressor(lgb.LGBMRegressor(n_estimators=20, max_depth=3, learning_rate=0.3))
            fallback_model.fit(X_train, Y_train)
            if enable_feature_selection:
                fallback_model.selected_features_ = X_train.columns.tolist()
            return fallback_model
    
    def _generate_portfolio_from_probabilities(self, main_probs_map, bonus_probs_map, purchase_count=5):
        """
        予測された本数字とボーナス数字の確率から、購入するポートフォリオを生成する（高速化版）
        入力はカラム名をキー、確率を値とする辞書。
        """
        # NumPy配列で高速処理
        main_probs = np.zeros(43, dtype=np.float32)
        bonus_probs = np.zeros(43, dtype=np.float32)

        # 辞書から値を取得して配列を埋める（ベクトル化）
        for i in range(1, 44):
            main_probs[i-1] = main_probs_map.get(f'is_appear_{i}', 0)
            bonus_probs[i-1] = bonus_probs_map.get(f'is_bonus_appear_{i}', 0)

        # スコアを計算（重み付き合成）
        combined_scores = main_probs * 0.8 + bonus_probs * 0.2  # 本数字により重きを置く

        # 確率上位の数字のインデックスを取得
        top_indices = np.argsort(combined_scores)[::-1]
        
        # ポートフォリオを格納するリスト
        portfolio = []
        
        # 効率的な組み合わせ生成
        candidate_numbers = top_indices[:min(len(top_indices), 20)] + 1  # 上位20個の数字（1-based）
        
        # 確定的な組み合わせ生成（より高速）
        from itertools import combinations
        
        # まず上位6個での組み合わせを生成
        if len(candidate_numbers) >= 6:
            portfolio.append(candidate_numbers[:6].tolist())
        
        # 残りは上位12個から組み合わせを生成
        if len(candidate_numbers) >= 12:
            pool_size = min(12, len(candidate_numbers))
            pool_numbers = candidate_numbers[:pool_size]
            
            # 効率的な組み合わせ生成
            combinations_iter = combinations(pool_numbers, 6)
            for combo in combinations_iter:
                if len(portfolio) >= purchase_count:
                    break
                portfolio.append(list(combo))
        
        # 足りない場合は残りをランダムで補完
        while len(portfolio) < purchase_count:
            try:
                selected = np.random.choice(candidate_numbers, 6, replace=False)
                portfolio.append(sorted(selected.tolist()))
            except ValueError:
                # candidate_numbersが6個未満の場合
                selected = np.random.choice(range(1, 44), 6, replace=False)
                portfolio.append(sorted(selected.tolist()))

        return portfolio[:purchase_count]  # 指定数だけ返す

    def _generate_numbers_from_probabilities(self, probabilities: np.ndarray, 
                                           purchase_count: int) -> List[List[int]]:
        """
        確率分布から数字の組み合わせを生成
        
        Args:
            probabilities: 各数字の出現確率 (43次元)
            purchase_count: 生成する組み合わせ数
            
        Returns:
            数字の組み合わせリスト
        """
        combinations = []
        
        # 入力の安全性チェック
        if len(probabilities) < 43:
            logger.warning(f"確率配列のサイズが不正: {len(probabilities)} (期待値: 43)")
            # サイズを43に調整
            padded_probs = np.ones(43) / 43  # 均等分布で初期化
            padded_probs[:len(probabilities)] = probabilities
            probabilities = padded_probs
        elif len(probabilities) > 43:
            probabilities = probabilities[:43]
        
        # 確率の正規化
        probabilities = np.maximum(probabilities, 0.001)  # 最小値保証
        probabilities = probabilities / np.sum(probabilities)
        
        try:
            for _ in range(min(purchase_count, 10)):  # 最大10組合せに制限
                # 確率に基づいて6個の数字を選択
                selected_numbers = np.random.choice(
                    range(1, 44), 
                    size=6, 
                    replace=False, 
                    p=probabilities
                )
                combinations.append(sorted(selected_numbers.tolist()))
        except Exception as e:
            logger.error(f"数字組み合わせ生成エラー: {e}")
            # フォールバック: ランダム選択
            for _ in range(min(purchase_count, 5)):
                selected_numbers = np.random.choice(range(1, 44), size=6, replace=False)
                combinations.append(sorted(selected_numbers.tolist()))
        
        return combinations

    def _calculate_profit(self, hit_count: int, purchase_count: int) -> int:
        """
        的中数に基づく損益計算
        
        Args:
            hit_count: 的中数
            purchase_count: 購入口数
            
        Returns:
            損益（円）
        """
        cost = purchase_count * 200  # 1口200円
        
        if hit_count == 6:
            return self.prizes[1] - cost  # 1等
        elif hit_count == 5:
            return self.prizes[2] - cost  # 2等 （ボーナス数字一致の場合も考慮）
        elif hit_count == 4:
            return self.prizes[3] - cost  # 3等
        elif hit_count == 3:
            return self.prizes[4] - cost  # 4等
        elif hit_count == 2:
            return self.prizes[5] - cost  # 5等
        else:
            return -cost  # はずれ

    def _get_actual_numbers(self, row: pd.Series) -> Dict[str, List[int]]:
        """
        実際の当選番号を取得
        """
        try:
            main_numbers = []
            main_cols = [f'本数字{i}' for i in range(1, 7)]
            for col in main_cols:
                if col in row.index and pd.notna(row[col]):
                    try:
                        main_numbers.append(int(row[col]))
                    except (ValueError, TypeError):
                        continue
            
            bonus_number = None
            if 'ボーナス数字' in row.index and pd.notna(row['ボーナス数字']):
                try:
                    bonus_number = int(row['ボーナス数字'])
                except (ValueError, TypeError):
                    bonus_number = None # 変換できない場合はNoneのまま
            
            return {'main': sorted(main_numbers), 'bonus': bonus_number}
        
        except Exception as e:
            logger.error(f"当選番号の取得中にエラー: {e}")
            return {'main': [], 'bonus': None}

    def _calculate_winnings(self, portfolio: List[Tuple[int, ...]], 
                            actual: Dict[str, any]) -> Tuple[int, Dict[int, int]]:
        """
        当選金額と当選等級ごとのヒット数を計算
        """
        total_winnings = 0
        hits_detail = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        
        actual_main = set(actual.get('main', []))
        actual_bonus = actual.get('bonus')
        
        if not actual_main:
            return 0, hits_detail

        for ticket in portfolio:
            match_main = len(set(ticket).intersection(actual_main))
            match_bonus = 1 if actual_bonus is not None and actual_bonus in ticket else 0
            
            if match_main == 6:
                hits_detail[1] += 1
                total_winnings += self.prizes[1]
            elif match_main == 5 and match_bonus == 1:
                hits_detail[2] += 1
                total_winnings += self.prizes[2]
            elif match_main == 5:
                hits_detail[3] += 1
                total_winnings += self.prizes[3]
            elif match_main == 4:
                hits_detail[4] += 1
                total_winnings += self.prizes[4]
            elif match_main == 3:
                hits_detail[5] += 1
                total_winnings += self.prizes[5]
        
        return total_winnings, hits_detail

    def _calculate_hit_rates(self, performance_log: List[Dict]) -> Dict[str, float]:
        """
        バックテスト結果から的中率を計算（安全なキーアクセス）
        """
        if not performance_log:
            return {'hit_rate_3': 0, 'hit_rate_4': 0, 'hit_rate_5': 0}
            
        total_draws = len(performance_log)
        
        # hits_detailキーが存在することを確認してからアクセス
        hits_3_or_better = 0
        hits_4_or_better = 0 
        hits_5_or_better = 0
        
        for log in performance_log:
            hits_detail = log.get('hits_detail', {1: 0, 2: 0, 3: 0, 4: 0, 5: 0})
            
            # 3等以上（1等、2等、3等）
            if hits_detail.get(1, 0) > 0 or hits_detail.get(2, 0) > 0 or hits_detail.get(3, 0) > 0:
                hits_3_or_better += 1
            
            # 4等以上（1等、2等、3等、4等）  
            if (hits_detail.get(1, 0) > 0 or hits_detail.get(2, 0) > 0 or 
                hits_detail.get(3, 0) > 0 or hits_detail.get(4, 0) > 0):
                hits_4_or_better += 1
            
            # 5等以上（全等級）
            if (hits_detail.get(1, 0) > 0 or hits_detail.get(2, 0) > 0 or 
                hits_detail.get(3, 0) > 0 or hits_detail.get(4, 0) > 0 or 
                hits_detail.get(5, 0) > 0):
                hits_5_or_better += 1
        
        return {
            'hit_rate_3': hits_3_or_better / total_draws,
            'hit_rate_4': hits_4_or_better / total_draws,
            'hit_rate_5': hits_5_or_better / total_draws
        }

    def _analyze_predictions(self, predicted_portfolio: List[List[int]], 
                           actual_numbers: Dict[str, any]) -> Dict:
        """
        予想と実際の結果を詳細分析
        """
        analysis = {
            'total_tickets': len(predicted_portfolio),
            'actual_main': actual_numbers.get('main', []),
            'actual_bonus': actual_numbers.get('bonus'),
            'ticket_analysis': [],
            'number_hit_analysis': {},
            'summary': {}
        }
        
        actual_main_set = set(actual_numbers.get('main', []))
        actual_bonus = actual_numbers.get('bonus')
        
        # 各予想番号の出現回数と的中回数を追跡
        predicted_numbers = {}
        for ticket in predicted_portfolio:
            for num in ticket:
                predicted_numbers[num] = predicted_numbers.get(num, 0) + 1
        
        # 数字別の的中分析
        for num in range(1, 44):
            if num in predicted_numbers:
                is_main_hit = num in actual_main_set
                is_bonus_hit = num == actual_bonus
                analysis['number_hit_analysis'][num] = {
                    'predicted_count': predicted_numbers[num],
                    'is_main_hit': is_main_hit,
                    'is_bonus_hit': is_bonus_hit,
                    'hit_type': 'main' if is_main_hit else ('bonus' if is_bonus_hit else 'miss')
                }
        
        # 各チケットの分析
        for i, ticket in enumerate(predicted_portfolio):
            ticket_set = set(ticket)
            main_matches = len(ticket_set.intersection(actual_main_set))
            bonus_match = 1 if actual_bonus and actual_bonus in ticket_set else 0
            
            # 等級判定
            prize_rank = None
            if main_matches == 6:
                prize_rank = 1
            elif main_matches == 5 and bonus_match == 1:
                prize_rank = 2
            elif main_matches == 5:
                prize_rank = 3
            elif main_matches == 4:
                prize_rank = 4
            elif main_matches == 3:
                prize_rank = 5
            
            analysis['ticket_analysis'].append({
                'ticket_id': i + 1,
                'numbers': ticket,
                'main_matches': main_matches,
                'bonus_match': bonus_match,
                'prize_rank': prize_rank,
                'matched_main_numbers': list(ticket_set.intersection(actual_main_set)),
                'matched_bonus': actual_bonus if bonus_match else None
            })
        
        # サマリー統計
        hit_counts = {'main': {}, 'bonus': 0}
        for num in actual_main_set:
            if num in predicted_numbers:
                hit_counts['main'][num] = predicted_numbers[num]
        
        if actual_bonus and actual_bonus in predicted_numbers:
            hit_counts['bonus'] = predicted_numbers[actual_bonus]
        
        analysis['summary'] = {
            'main_numbers_predicted': len([n for n in actual_main_set if n in predicted_numbers]),
            'bonus_predicted': actual_bonus in predicted_numbers if actual_bonus else False,
            'total_main_predictions': sum(hit_counts['main'].values()),
            'total_bonus_predictions': hit_counts['bonus'],
            'hit_counts': hit_counts
        }
        
        return analysis

    def _analyze_predictions_fast(self, predicted_portfolio: List[List[int]], 
                                  actual_numbers: Dict[str, any]) -> Dict:
        """
        予想と実際の結果を高速分析（軽量版）
        """
        actual_main_set = set(actual_numbers.get('main', []))
        actual_bonus = actual_numbers.get('bonus')
        
        # 基本的な統計のみ
        analysis = {
            'total_tickets': len(predicted_portfolio),
            'actual_main': list(actual_main_set),
            'actual_bonus': actual_bonus,
            'summary': {}
        }
        
        # 高速な当選分析
        hit_counts = {'main': 0, 'bonus': 0}
        total_main_predictions = 0
        
        for ticket in predicted_portfolio:
            ticket_set = set(ticket)
            main_hits = len(ticket_set.intersection(actual_main_set))
            if main_hits > 0:
                hit_counts['main'] += main_hits
                total_main_predictions += len(ticket)
            
            if actual_bonus and actual_bonus in ticket_set:
                hit_counts['bonus'] += 1
        
        analysis['summary'] = {
            'main_numbers_hit': len([n for n in actual_main_set if any(n in ticket for ticket in predicted_portfolio)]),
            'bonus_predicted': actual_bonus in {n for ticket in predicted_portfolio for n in ticket} if actual_bonus else False,
            'total_main_predictions': total_main_predictions,
            'hit_counts': hit_counts
        }
        
        return analysis

    def _calculate_prediction_accuracy(self, predicted_probs: np.ndarray, 
                                       actual_numbers: Dict, num_targets: int) -> float:
        """
        予測の正解率（簡易的な評価指標）
        """
        try:
            actual_vector = np.zeros(num_targets)
            
            # 本数字
            main_indices = [n - 1 for n in actual_numbers['main']]
            actual_vector[main_indices] = 1
            
            # ボーナス数字
            if actual_numbers['bonus'] is not None:
                bonus_idx = actual_numbers['bonus'] - 1
                # is_bonus_appear_* は 43から始まる
                if (43 + bonus_idx) < len(actual_vector):
                    actual_vector[43 + bonus_idx] = 1

            # 簡単な比較（例：トップN個の予測がどれだけ当たったか）
            top_k = 6
            predicted_top_indices = np.argsort(predicted_probs)[-top_k:]
            actual_top_indices = np.where(actual_vector == 1)[0]
            
            accuracy = len(set(predicted_top_indices).intersection(set(actual_top_indices))) / top_k
            return accuracy
        except Exception:
            return 0.0
    
    def save_model(self, model: Any, filepath: str):
        """
        学習済みモデルを保存
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"モデルを保存しました: {filepath}")
    
    def load_model(self, filepath: str) -> Any:
        """
        学習済みモデルを読み込み
        """
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"モデルを読み込みました: {filepath}")
        return model

    def _get_model(self, model_name: str):
        """
        モデル名に基づいてモデルを取得
        """
        model_name = model_name.lower()
        
        rf_params = self.default_params.get('random_forest', {})
        lgbm_params = self.default_params.get('lightgbm', {})
        xgb_params = self.default_params.get('xgboost', {})
        lr_params = self.default_params.get('logistic_regression', {})

        rf_model = RandomForestRegressor(**rf_params)
        lgbm_model = lgb.LGBMRegressor(**lgbm_params)
        xgb_model = xgb.XGBRegressor(**xgb_params)

        if model_name == 'xgboost':
            return MultiOutputRegressor(xgb_model)
        
        elif model_name == 'lightgbm':
            return MultiOutputRegressor(lgbm_model)
            
        elif model_name == 'random_forest':
            return rf_model # RandomForestRegressorはネイティブでマルチアウトプットをサポート

        elif model_name == 'logistic_regression':
            return MultiOutputRegressor(LogisticRegression(**lr_params))

        elif model_name == 'ensemble_balanced':
            # バランス型: RandomForest + LightGBM
            estimators = [('rf', rf_model), ('lgbm', lgbm_model)]
            return MultiOutputRegressor(VotingRegressor(estimators=estimators))

        elif model_name == 'ensemble_aggressive':
            # アグレッシブ型: LightGBM + XGBoost
            estimators = [('lgbm', lgbm_model), ('xgb', xgb_model)]
            return MultiOutputRegressor(VotingRegressor(estimators=estimators))

        elif model_name in ['ensemble_full', 'ensemble']: # 以前の'ensemble'との互換性維持
            # フルアンサンブル: RandomForest + LightGBM + XGBoost
            estimators = [('rf', rf_model), ('lgbm', lgbm_model), ('xgb', xgb_model)]
            return MultiOutputRegressor(VotingRegressor(estimators=estimators))

        else:
            raise ValueError(f"サポートされていないモデルタイプ: {model_name}")

    def _adjust_learning_rate(self, model: Any, learning_rate_factor: float):
        """
        継続学習用にモデルの学習率を調整
        """
        try:
            if hasattr(model, 'estimators_'):
                # MultiOutputRegressorの場合
                for estimator in model.estimators_:
                    if hasattr(estimator, 'learning_rate'):
                        estimator.learning_rate *= learning_rate_factor
                    elif hasattr(estimator, 'estimators_'):
                        # VotingRegressorの場合 - named_estimators_を使用
                        if hasattr(estimator, 'named_estimators_'):
                            for name, sub_estimator in estimator.named_estimators_.items():
                                if hasattr(sub_estimator, 'learning_rate'):
                                    sub_estimator.learning_rate *= learning_rate_factor
                        else:
                            # estimators_を直接使用する場合
                            for sub_estimator in estimator.estimators_:
                                if hasattr(sub_estimator, 'learning_rate'):
                                    sub_estimator.learning_rate *= learning_rate_factor
            elif hasattr(model, 'learning_rate'):
                # 直接learning_rateを持つモデル
                model.learning_rate *= learning_rate_factor
            
            logger.info(f"学習率を {learning_rate_factor} 倍に調整しました。")
        except Exception as e:
            logger.warning(f"学習率の調整に失敗しました: {e}")
            # エラーが発生した場合でも継続できるようにデフォルト値を設定
            logger.info("学習率調整をスキップして標準設定で継続します。")

    def _continue_training(self, model: Any, X_train: pd.DataFrame, Y_train: pd.DataFrame):
        """
        継続学習を実行（増分学習）
        """
        try:
            # 一部のモデルでは増分学習をサポート
            if hasattr(model, 'partial_fit'):
                model.partial_fit(X_train, Y_train)
            else:
                # 増分学習をサポートしていない場合は再学習
                model.fit(X_train, Y_train)
            
            return model
        except Exception as e:
            logger.error(f"継続学習中にエラーが発生: {e}")
            # エラーが発生した場合は新しいモデルで学習
            return self._train_model(X_train, Y_train)

class PatternCorrection:
    """
    データリーケージを防ぐ正しいパターン学習クラス
    """
    def __init__(self):
        """
        パターン補正システムの初期化
        """
        # 数字範囲別の補正値
        self.range_corrections = {
            'small_numbers': 0.0,    # 1-10の補正
            'mid_low': 0.0,          # 11-20の補正
            'mid_high': 0.0,         # 21-30の補正
            'large_numbers': 0.0     # 31-43の補正
        }
        
        # 構造パターンの補正値
        self.pattern_corrections = {
            'odd_bias': 0.0,         # 奇数偏重の補正
            'consecutive_bias': 0.0,  # 連続数字の補正
            'sum_range_bias': 0.0,   # 合計値範囲の補正
            'gap_pattern_bias': 0.0  # 数字間隔パターンの補正
        }
        
        # セット球依存の補正値
        self.set_corrections = {
            'A': {'tendency': 0.0},  # Aセット球の傾向補正
            'B': {'tendency': 0.0}   # Bセット球の傾向補正
        }
        
        # 学習重み（新しい学習結果の重要度）
        self.learning_weight = 0.1
        
        logger.info("パターン補正システムを初期化しました")
    
    def analyze_prediction_pattern(self, predicted_numbers: List[int], 
                                 actual_numbers: List[int], 
                                 set_ball: str = None) -> Dict:
        """
        予測と実際の結果のパターン差分を分析（データリーケージなし）
        
        Args:
            predicted_numbers: 予測された数字リスト
            actual_numbers: 実際の当選数字リスト
            set_ball: セット球（A or B）
            
        Returns:
            分析結果辞書
        """
        analysis = {}
        
        # 1. 数字範囲の傾向分析
        predicted_ranges = self._categorize_by_range(predicted_numbers)
        actual_ranges = self._categorize_by_range(actual_numbers)
        
        for range_name in self.range_corrections.keys():
            predicted_count = predicted_ranges.get(range_name, 0)
            actual_count = actual_ranges.get(range_name, 0)
            bias = (actual_count - predicted_count) / 6.0  # 正規化
            analysis[f'range_{range_name}_bias'] = bias
        
        # 2. 構造パターンの分析
        analysis['odd_bias'] = self._analyze_odd_even_bias(predicted_numbers, actual_numbers)
        analysis['consecutive_bias'] = self._analyze_consecutive_bias(predicted_numbers, actual_numbers)
        analysis['sum_range_bias'] = self._analyze_sum_range_bias(predicted_numbers, actual_numbers)
        analysis['gap_pattern_bias'] = self._analyze_gap_pattern_bias(predicted_numbers, actual_numbers)
        
        # 3. セット球依存分析
        if set_ball:
            analysis['set_dependency'] = self._analyze_set_dependency(
                predicted_numbers, actual_numbers, set_ball
            )
        
        return analysis
    
    def update_corrections(self, pattern_analysis: Dict):
        """
        パターン分析結果に基づいて補正値を更新
        
        Args:
            pattern_analysis: analyze_prediction_pattern()の結果
        """
        # 数字範囲補正の更新
        for range_name in self.range_corrections.keys():
            bias_key = f'range_{range_name}_bias'
            if bias_key in pattern_analysis:
                bias = pattern_analysis[bias_key]
                # 指数移動平均で補正値を更新
                self.range_corrections[range_name] = (
                    (1 - self.learning_weight) * self.range_corrections[range_name] +
                    self.learning_weight * bias
                )
        
        # 構造パターン補正の更新
        pattern_keys = ['odd_bias', 'consecutive_bias', 'sum_range_bias', 'gap_pattern_bias']
        for key in pattern_keys:
            if key in pattern_analysis:
                self.pattern_corrections[key] = (
                    (1 - self.learning_weight) * self.pattern_corrections[key] +
                    self.learning_weight * pattern_analysis[key]
                )
        
        # セット球補正の更新
        if 'set_dependency' in pattern_analysis:
            set_data = pattern_analysis['set_dependency']
            for set_ball in ['A', 'B']:
                if set_ball in set_data:
                    self.set_corrections[set_ball]['tendency'] = (
                        (1 - self.learning_weight) * self.set_corrections[set_ball]['tendency'] +
                        self.learning_weight * set_data[set_ball]
                    )
        
        logger.debug(f"パターン補正値を更新しました: 範囲補正={self.range_corrections}")
    
    def apply_corrections(self, raw_probabilities: np.ndarray, 
                         set_ball: str = None) -> np.ndarray:
        """
        学習済み補正値を確率分布に適用
        
        Args:
            raw_probabilities: 生の確率分布 (43次元)
            set_ball: セット球情報
            
        Returns:
            補正済み確率分布
        """
        # 入力の安全性チェック
        if len(raw_probabilities) != 43:
            logger.warning(f"確率配列のサイズが不正: {len(raw_probabilities)} (期待値: 43)")
            # サイズを調整
            if len(raw_probabilities) < 43:
                padded_probs = np.ones(43) / 43  # 均等分布で初期化
                padded_probs[:len(raw_probabilities)] = raw_probabilities
                raw_probabilities = padded_probs
            else:
                raw_probabilities = raw_probabilities[:43]
        
        corrected_probs = raw_probabilities.copy()
        
        # 数字範囲別補正の適用（安全なインデックスアクセス）
        max_index = len(corrected_probs)
        for i in range(1, min(44, max_index + 1)):
            if i - 1 < max_index:  # インデックス範囲チェック
                range_name = self._get_number_range(i)
                correction = self.range_corrections.get(range_name, 0.0)
                corrected_probs[i-1] *= (1.0 + correction)
        
        # パターン補正の適用（簡略化）
        # 実際の実装では、より複雑なパターン補正ロジックを適用
        
        # 正規化
        corrected_probs = np.maximum(corrected_probs, 0.001)  # 最小値保証
        corrected_probs = corrected_probs / np.sum(corrected_probs)
        
        return corrected_probs
    
    def _categorize_by_range(self, numbers: List[int]) -> Dict[str, int]:
        """数字を範囲別に分類"""
        categorized = {'small_numbers': 0, 'mid_low': 0, 'mid_high': 0, 'large_numbers': 0}
        for num in numbers:
            if 1 <= num <= 10:
                categorized['small_numbers'] += 1
            elif 11 <= num <= 20:
                categorized['mid_low'] += 1
            elif 21 <= num <= 30:
                categorized['mid_high'] += 1
            elif 31 <= num <= 43:
                categorized['large_numbers'] += 1
        return categorized
    
    def _get_number_range(self, number: int) -> str:
        """数字の範囲を取得"""
        if 1 <= number <= 10:
            return 'small_numbers'
        elif 11 <= number <= 20:
            return 'mid_low'
        elif 21 <= number <= 30:
            return 'mid_high'
        else:
            return 'large_numbers'
    
    def _analyze_odd_even_bias(self, predicted: List[int], actual: List[int]) -> float:
        """奇偶バランスの偏り分析"""
        pred_odd_count = sum(1 for x in predicted if x % 2 == 1)
        actual_odd_count = sum(1 for x in actual if x % 2 == 1)
        return (actual_odd_count - pred_odd_count) / 6.0
    
    def _analyze_consecutive_bias(self, predicted: List[int], actual: List[int]) -> float:
        """連続数字の偏り分析"""
        pred_consecutive = self._count_consecutive(predicted)
        actual_consecutive = self._count_consecutive(actual)
        return (actual_consecutive - pred_consecutive) / 5.0  # 最大5ペア
    
    def _analyze_sum_range_bias(self, predicted: List[int], actual: List[int]) -> float:
        """合計値範囲の偏り分析"""
        pred_sum = sum(predicted)
        actual_sum = sum(actual)
        # 標準的な合計値範囲（130-150）からの偏差を分析
        standard_sum = 140
        pred_deviation = abs(pred_sum - standard_sum)
        actual_deviation = abs(actual_sum - standard_sum)
        return (actual_deviation - pred_deviation) / 50.0  # 正規化
    
    def _analyze_gap_pattern_bias(self, predicted: List[int], actual: List[int]) -> float:
        """数字間隔パターンの偏り分析"""
        pred_gaps = self._calculate_gaps(predicted)
        actual_gaps = self._calculate_gaps(actual)
        # 平均間隔の差
        pred_avg_gap = np.mean(pred_gaps)
        actual_avg_gap = np.mean(actual_gaps)
        return (actual_avg_gap - pred_avg_gap) / 10.0  # 正規化
    
    def _analyze_set_dependency(self, predicted: List[int], actual: List[int], set_ball: str) -> Dict:
        """セット球依存性の分析"""
        # 簡略化された分析
        return {set_ball: 0.0}  # 実装では、より詳細な分析を行う
    
    def _count_consecutive(self, numbers: List[int]) -> int:
        """連続数字のペア数をカウント"""
        sorted_nums = sorted(numbers)
        consecutive_count = 0
        for i in range(len(sorted_nums) - 1):
            if sorted_nums[i+1] - sorted_nums[i] == 1:
                consecutive_count += 1
        return consecutive_count
    
    def _calculate_gaps(self, numbers: List[int]) -> List[int]:
        """数字間のギャップを計算"""
        sorted_nums = sorted(numbers)
        gaps = []
        for i in range(len(sorted_nums) - 1):
            gaps.append(sorted_nums[i+1] - sorted_nums[i])
        return gaps
    
    def get_correction_summary(self) -> Dict:
        """現在の補正状況のサマリーを取得"""
        return {
            'range_corrections': self.range_corrections.copy(),
            'pattern_corrections': self.pattern_corrections.copy(),
            'set_corrections': self.set_corrections.copy()
        }
