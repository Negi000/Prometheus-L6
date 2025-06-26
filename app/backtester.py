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
        
        # デフォルトパラメータの設定
        self.default_params = {
            'xgboost': {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1, 'random_state': 42, 'verbosity': 0},
            'lightgbm': {'n_estimators': 100, 'max_depth': -1, 'learning_rate': 0.1, 'random_state': 42, 'verbosity': -1, 'min_child_samples': 20},
            'random_forest': {'n_estimators': 100, 'max_depth': 10, 'random_state': 42, 'n_jobs': -1},
            'logistic_regression': {'solver': 'liblinear', 'random_state': 42}
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
        
        # 特徴量とターゲット列の定義
        target_cols = [f'is_appear_{n}' for n in range(1, 43)] + [f'is_bonus_appear_{n}' for n in range(1, 43) if f'is_bonus_appear_{n}' in df_run.columns]
        
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
                
                model = self._train_model(X_train, Y_train)
                
                # --- 予測 ---
                logger.info(f"[{draw_id}] Predicting for draw {draw_id - 1}...")
                # is_appear_* と is_bonus_appear_* の確率を予測
                predictions = model.predict(X_train)
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
                
                # 詳細分析：予想vs実際の比較
                prediction_analysis = self._analyze_predictions(predicted_portfolio, actual_numbers) if detailed_log else None
                
                log_entry = {
                    'draw_id': draw_id,
                    'profit': profit,
                    'winnings': winnings,
                    'cost': cost,
                    'hits_detail': hits_detail,
                    'portfolio_size': len(predicted_portfolio),
                    'actual_numbers': actual_numbers,
                    'predicted_portfolio': predicted_portfolio if detailed_log else predicted_portfolio[:5],
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
        継続学習の実行（既存モデルを基に追加学習）
        
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
        logger.info(f"継続学習開始: 第{start_draw}回〜第{end_draw}回, ウィンドウ={window_size}, モデル={self.model_type}, 学習率係数={learning_rate_factor}")
        
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
        
        # 既存モデルをコピーして継続学習用モデルを作成
        import copy
        model = copy.deepcopy(base_model)
        
        # 学習率を調整（XGBoostやLightGBMの場合）
        self._adjust_learning_rate(model, learning_rate_factor)
        
        # 特徴量とターゲット列の定義
        target_cols = [f'is_appear_{n}' for n in range(1, 43)] + [f'is_bonus_appear_{n}' for n in range(1, 43) if f'is_bonus_appear_{n}' in df_run.columns]
        
        # 自己補正フィードバック用の列を初期化
        if enable_feedback:
            logger.info("継続学習で自己補正フィードバック機能を有効化。")
            self.error_cols = [f'error_{t}' for t in target_cols]
            self.error_ma_cols = [f'error_ma5_{t}' for t in target_cols]
            for col in self.error_cols + self.error_ma_cols:
                if col not in df_run.columns:
                    df_run[col] = 0.0

        feature_cols = self._get_feature_columns(df_run, enable_feedback)
        
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
                
                predictions = model.predict(X_test)
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
                
                winnings, hits_detail = self._calculate_winnings(
                    predicted_portfolio, actual_numbers
                )
                
                cost = purchase_count * 200
                profit = winnings - cost
                
                log_entry = {
                    'draw_id': draw_id,
                    'profit': profit,
                    'winnings': winnings,
                    'cost': cost,
                    'hits_detail': hits_detail,
                    'portfolio_size': len(predicted_portfolio),
                    'actual_numbers': actual_numbers,
                    'predicted_portfolio': predicted_portfolio[:5] if detailed_log else None,
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
    
    def _train_model(self, X_train: pd.DataFrame, Y_train: pd.DataFrame):
        """
        選択されたモデルタイプに基づいてモデルを学習
        """
        model_type = self.model_type.lower()

        # ベースモデルの定義
        rf_params = self.default_params.get('random_forest', {})
        lgbm_params = self.default_params.get('lightgbm', {})
        xgb_params = self.default_params.get('xgboost', {})
        lr_params = self.default_params.get('logistic_regression', {})

        rf_model = RandomForestRegressor(**rf_params)
        lgbm_model = lgb.LGBMRegressor(**lgbm_params)
        xgb_model = xgb.XGBRegressor(**xgb_params)

        if model_type == 'xgboost':
            model = MultiOutputRegressor(xgb_model)
        
        elif model_type == 'lightgbm':
            model = MultiOutputRegressor(lgbm_model)
            
        elif model_type == 'random_forest':
            model = rf_model # RandomForestRegressorはネイティブでマルチアウトプットをサポート

        elif model_type == 'logistic_regression':
            model = MultiOutputRegressor(LogisticRegression(**lr_params))

        elif model_type == 'ensemble_balanced':
            # バランス型: RandomForest + LightGBM
            estimators = [('rf', rf_model), ('lgbm', lgbm_model)]
            model = MultiOutputRegressor(VotingRegressor(estimators=estimators))

        elif model_type == 'ensemble_aggressive':
            # アグレッシブ型: LightGBM + XGBoost
            estimators = [('lgbm', lgbm_model), ('xgb', xgb_model)]
            model = MultiOutputRegressor(VotingRegressor(estimators=estimators))

        elif model_type in ['ensemble_full', 'ensemble']: # 以前の'ensemble'との互換性維持
            # フルアンサンブル: RandomForest + LightGBM + XGBoost
            estimators = [('rf', rf_model), ('lgbm', lgbm_model), ('xgb', xgb_model)]
            model = MultiOutputRegressor(VotingRegressor(estimators=estimators))

        else:
            raise ValueError(f"サポートされていないモデルタイプ: {self.model_type}")
        
        model.fit(X_train, Y_train)
        return model
    
    def _generate_portfolio_from_probabilities(self, main_probs_map, bonus_probs_map, purchase_count=5):
        """
        予測された本数字とボーナス数字の確率から、購入するポートフォリオを生成する。
        入力はカラム名をキー、確率を値とする辞書。
        """
        main_probs = np.zeros(43)
        bonus_probs = np.zeros(43)

        # 辞書から値を取得して配列を埋める
        for i in range(1, 44):
            main_probs[i-1] = main_probs_map.get(f'is_appear_{i}', 0)
            bonus_probs[i-1] = bonus_probs_map.get(f'is_bonus_appear_{i}', 0)

        # スコアを計算 (ここで形状が揃う)
        combined_scores = main_probs + bonus_probs

        # スコアに基づいてポートフォリオを複数生成
        # 確率の高い数字のインデックスを取得（+1して実際の数字に）
        top_indices = np.argsort(combined_scores)[::-1]
        
        # ポートフォリオを格納するリスト
        portfolio = []
        
        # 生成済み組み合わせを記録するセット
        generated_combinations = set()

        # 組み合わせを生成
        # 確率上位の数字から優先的に組み合わせる
        candidate_numbers = [idx + 1 for idx in top_indices]
        
        # 組み合わせが見つからない場合に備え、試行回数に上限を設定
        max_attempts = purchase_count * 20 
        attempts = 0

        while len(portfolio) < purchase_count and attempts < max_attempts:
            # 確率上位から6つを選ぶだけだと多様性がなくなるため、上位N個からランダムに6つ選ぶ
            # 上位15個程度から選ぶことで、ある程度のランダム性を確保
            pool_size = min(len(candidate_numbers), 15)
            selected_indices = np.random.choice(range(pool_size), 6, replace=False)
            
            combination = tuple(sorted([candidate_numbers[i] for i in selected_indices]))

            if combination not in generated_combinations:
                portfolio.append(list(combination))
                generated_combinations.add(combination)
            
            attempts += 1
    
        # もし上限に達しても足りない場合は、残りをランダムな組み合わせで埋める
        while len(portfolio) < purchase_count:
            combination = tuple(sorted(np.random.choice(range(1, 44), 6, replace=False)))
            if combination not in generated_combinations:
                portfolio.append(list(combination))
                generated_combinations.add(combination)

        return portfolio


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
        バックテスト結果から的中率を計算
        """
        if not performance_log:
            return {'hit_rate_3': 0, 'hit_rate_4': 0, 'hit_rate_5': 0}
            
        total_draws = len(performance_log)
        
        hits_3_or_better = sum(1 for log in performance_log if log['hits_detail'][1] > 0 or log['hits_detail'][2] > 0 or log['hits_detail'][3] > 0)
        hits_4_or_better = hits_3_or_better + sum(1 for log in performance_log if log['hits_detail'][4] > 0)
        hits_5_or_better = hits_4_or_better + sum(1 for log in performance_log if log['hits_detail'][5] > 0)
        
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
