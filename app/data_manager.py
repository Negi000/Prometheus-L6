"""
データ管理モジュール (data_manager.py)
LOTO6データの読み込み、検証、更新機能を提供
"""

import pandas as pd
import numpy as np
import os
import sqlite3
import configparser
import logging
from datetime import datetime
import json
import pickle

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataManager:
    def __init__(self, config_path='config.ini'):
        """
        データ管理クラスの初期化
        
        Args:
            config_path (str): 設定ファイルのパス
        """
        self.config = configparser.ConfigParser()
        self.config.read(config_path, encoding='utf-8')
        self.setup_database()
    
    def setup_database(self):
        """
        戦略管理データベースの初期化
        テーブルが存在しない場合は作成する
        """
        db_path = self.config['Paths']['database_path']
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # テーブル存在確認
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='strategies'")
        table_exists = cursor.fetchone() is not None
        
        if not table_exists:
            # 新規作成
            cursor.execute('''
            CREATE TABLE strategies (
                strategy_id TEXT PRIMARY KEY,
                strategy_name TEXT NOT NULL,
                model_type TEXT NOT NULL,
                model_path TEXT NOT NULL,
                created_at TEXT NOT NULL,
                backtest_profit REAL,
                backtest_hit_rate_3 REAL,
                backtest_hit_rate_4 REAL,
                backtest_hit_rate_5 REAL,
                description TEXT,
                backtest_log BLOB,
                parameters TEXT
            )
            ''')
        else:
            # 既存テーブルのカラム確認と追加
            cursor.execute("PRAGMA table_info(strategies)")
            columns = [info[1] for info in cursor.fetchall()]
            if 'backtest_log' not in columns:
                cursor.execute("ALTER TABLE strategies ADD COLUMN backtest_log BLOB")
            if 'parameters' not in columns:
                cursor.execute("ALTER TABLE strategies ADD COLUMN parameters TEXT")

        conn.commit()
        conn.close()
        logger.info("データベースが正常に初期化されました")
    
    def load_loto6_history(self):
        """
        LOTO6履歴データを読み込み
        
        Returns:
            pd.DataFrame: LOTO6履歴データ
        """
        csv_path = self.config['Paths']['loto6_history_csv']
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"LOTO6履歴ファイルが見つかりません: {csv_path}")
        
        try:
            # 複数のエンコーディングを試行
            encodings = ['shift-jis', 'utf-8', 'cp932', 'utf-8-sig', 'iso-2022-jp']
            df = None
            
            for encoding in encodings:
                try:
                    logger.info(f"エンコーディング '{encoding}' で読み込み試行中...")
                    df = pd.read_csv(csv_path, encoding=encoding)
                    logger.info(f"'{encoding}' での読み込み成功")
                    break
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    logger.warning(f"'{encoding}' でのエラー: {e}")
                    continue
            
            if df is None:
                # 最後の手段としてエラーを無視して読み込み
                try:
                    df = pd.read_csv(csv_path, encoding='shift-jis', errors='ignore')
                    logger.info("エラーを無視してshift-jisで読み込み成功")
                except:
                    try:
                        df = pd.read_csv(csv_path, encoding='utf-8', errors='ignore')
                        logger.info("エラーを無視してutf-8で読み込み成功")
                    except Exception as e:
                        raise Exception(f"すべてのエンコーディングで読み込み失敗: {e}")
            
            if df is None or df.empty:
                raise ValueError("データファイルが空か、読み込めませんでした")
            
            logger.info(f"LOTO6履歴データを読み込みました: {len(df)}件")
            
            # データの基本検証
            validated_df = self._validate_loto6_data(df)
            
            # 本数字合計の計算
            main_cols = [col for col in validated_df.columns if '本数字' in col and col != '本数字合計']
            if len(main_cols) == 6:
                validated_df['本数字合計'] = validated_df[main_cols].sum(axis=1)
            
            return validated_df
            
        except Exception as e:
            logger.error(f"LOTO6履歴データの読み込みに失敗しました: {e}")
            raise
    
    def _validate_loto6_data(self, df):
        """
        LOTO6データの基本検証
        
        Args:
            df (pd.DataFrame): 検証対象のデータフレーム
            
        Returns:
            pd.DataFrame: 検証・正規化されたデータフレーム
        """
        # DataFrameのコピーを作成（元のデータフレームを変更しない）
        validated_df = df.copy()
        
        # 列名の正規化
        validated_df.columns = validated_df.columns.str.strip()
        logger.info(f"検出された列名: {list(validated_df.columns)}")
        
        # 回号列の特定と統一（柔軟対応）
        round_col_names = ['第何回', '回', '回数', '回号']
        round_col = None
        for col_name in round_col_names:
            if col_name in validated_df.columns:
                round_col = col_name
                break
        
        if round_col is None:
            raise ValueError(f"回号列が見つかりません。利用可能な列: {list(validated_df.columns)}")
        
        # 列名を「第何回」に統一
        if round_col != '第何回':
            validated_df.rename(columns={round_col: '第何回'}, inplace=True)
            logger.info(f"列名を '{round_col}' から '第何回' に変更しました")
        
        # 本数字列の特定
        main_number_cols = [col for col in validated_df.columns if '本数字' in col and col != '本数字合計']
        
        if len(main_number_cols) < 6:
            raise ValueError(f"本数字の列が6つ見つかりません。見つかった列: {main_number_cols}")
        
        # データ型の変換と検証
        for col in main_number_cols:
            validated_df[col] = pd.to_numeric(validated_df[col], errors='coerce')
            # 範囲チェック（1-43）
            invalid_count = len(validated_df[(validated_df[col] < 1) | (validated_df[col] > 43) | validated_df[col].isna()])
            if invalid_count > 0:
                logger.warning(f"{col} に無効な値が {invalid_count} 件見つかりました")
        
        # 重複回号のチェック（統一後の列名「第何回」を使用）
        if validated_df['第何回'].duplicated().any():
            duplicates = validated_df[validated_df['第何回'].duplicated()]['第何回'].values
            logger.warning(f"重複する回号が見つかりました: {duplicates}")
        
        logger.info("データ検証が完了しました")
        return validated_df
    
    def load_all_combinations(self):
        """
        全組み合わせデータを読み込み
        
        Returns:
            list: 全組み合わせのリスト
        """
        txt_path = self.config['Paths']['all_combinations_txt']
        
        if not os.path.exists(txt_path):
            logger.warning(f"全組み合わせファイルが見つかりません: {txt_path}")
            return []
        
        try:
            combinations = []
            with open(txt_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        # カンマ区切りで分割し、数字に変換
                        combo = [int(x.strip()) for x in line.split(',')]
                        if len(combo) == 6:
                            combinations.append(tuple(sorted(combo)))
            
            logger.info(f"全組み合わせデータを読み込みました: {len(combinations)}件")
            return combinations
            
        except Exception as e:
            logger.error(f"全組み合わせデータの読み込みに失敗しました: {e}")
            return []
    
    def save_strategy(self, strategy_data: dict):
        """
        戦略をデータベースに保存
        
        Args:
            strategy_data (dict): 保存する戦略データ
        """
        db_path = self.config['Paths']['database_path']
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        strategy_name = strategy_data['strategy_name']
        model_type = strategy_data['model_type']
        model = strategy_data['model']
        description = strategy_data.get('description', "")
        
        # IDとモデルパスの生成
        strategy_id = f"{strategy_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        model_dir = os.path.join('output', 'models', strategy_id)
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "model.pkl")

        # モデルの保存
        try:
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            logger.info(f"モデルを保存しました: {model_path}")
        except Exception as e:
            logger.error(f"モデルの保存に失敗しました: {e}")
            conn.close()
            raise

        # ログとパラメータのシリアライズ
        backtest_log_blob = pickle.dumps(strategy_data.get('backtest_log', []))
        parameters_json = json.dumps(strategy_data.get('parameters', {}))

        cursor.execute('''
        INSERT OR REPLACE INTO strategies 
        (strategy_id, strategy_name, model_type, model_path, created_at,
         backtest_profit, backtest_hit_rate_3, backtest_hit_rate_4, 
         backtest_hit_rate_5, description, backtest_log, parameters)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            strategy_id, 
            strategy_name, 
            model_type, 
            model_path, 
            strategy_data['created_at'].isoformat(),
            strategy_data.get('backtest_profit'),
            strategy_data.get('backtest_hit_rate_3'),
            strategy_data.get('backtest_hit_rate_4'),
            strategy_data.get('backtest_hit_rate_5'),
            description,
            backtest_log_blob,
            parameters_json
        ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"戦略を保存しました: {strategy_name}")
    
    def update_strategy(self, strategy_name: str, strategy_data: dict):
        """
        既存戦略を上書き更新（真の継続学習用）
        
        Args:
            strategy_name (str): 更新する戦略名
            strategy_data (dict): 更新する戦略データ
        """
        db_path = self.config['Paths']['database_path']
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        try:
            # 既存の戦略IDとモデルパスを取得
            cursor.execute('SELECT strategy_id, model_path FROM strategies WHERE strategy_name = ?', (strategy_name,))
            existing = cursor.fetchone()
            
            if not existing:
                logger.error(f"更新対象の戦略が見つかりません: {strategy_name}")
                raise ValueError(f"戦略「{strategy_name}」が存在しません")
            
            strategy_id, existing_model_path = existing
            
            # 新しいモデルを既存のパスに保存（上書き）
            model = strategy_data['model']
            
            try:
                with open(existing_model_path, 'wb') as f:
                    pickle.dump(model, f)
                logger.info(f"モデルを上書き保存しました: {existing_model_path}")
            except Exception as e:
                logger.error(f"モデルの上書き保存に失敗しました: {e}")
                raise
            
            # データベースの戦略情報を更新
            backtest_log_blob = pickle.dumps(strategy_data.get('backtest_log', []))
            parameters_json = json.dumps(strategy_data.get('parameters', {}))
            
            cursor.execute('''
            UPDATE strategies SET
                created_at = ?,
                backtest_profit = ?,
                backtest_hit_rate_3 = ?,
                backtest_hit_rate_4 = ?,
                backtest_hit_rate_5 = ?,
                description = ?,
                backtest_log = ?,
                parameters = ?
            WHERE strategy_name = ?
            ''', (
                strategy_data['created_at'].isoformat(),
                strategy_data.get('backtest_profit'),
                strategy_data.get('backtest_hit_rate_3'),
                strategy_data.get('backtest_hit_rate_4'),
                strategy_data.get('backtest_hit_rate_5'),
                strategy_data.get('description', ""),
                backtest_log_blob,
                parameters_json,
                strategy_name
            ))
            
            conn.commit()
            logger.info(f"戦略「{strategy_name}」を上書き更新しました（継続学習）")
            
        except Exception as e:
            conn.rollback()
            logger.error(f"戦略更新に失敗しました: {e}")
            raise
        finally:
            conn.close()
    
    def get_all_strategies(self):
        """
        全ての戦略を取得
        
        Returns:
            pd.DataFrame: 戦略一覧
        """
        db_path = self.config['Paths']['database_path']
        if not os.path.exists(db_path):
            return pd.DataFrame()

        conn = sqlite3.connect(db_path)
        
        try:
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(strategies)")
            columns = [info[1] for info in cursor.fetchall()]
            
            query = "SELECT * FROM strategies ORDER BY created_at DESC"
            df = pd.read_sql_query(query, conn)

            if not df.empty:
                # モデルファイルの読み込みを追加
                def load_model_from_path(model_path):
                    if pd.isna(model_path) or not os.path.exists(model_path):
                        return None
                    try:
                        with open(model_path, 'rb') as f:
                            return pickle.load(f)
                    except Exception as e:
                        logger.error(f"モデル読み込みエラー: {e}")
                        return None
                
                if 'model_path' in df.columns:
                    df['model'] = df['model_path'].apply(load_model_from_path)
                
                if 'backtest_log' in columns and 'backtest_log' in df.columns and not df['backtest_log'].isnull().all():
                    def deserialize_log(x):
                        try:
                            return pickle.loads(x) if isinstance(x, bytes) else []
                        except Exception:
                            return []
                    df['backtest_log'] = df['backtest_log'].apply(deserialize_log)

                if 'parameters' in columns and 'parameters' in df.columns and not df['parameters'].isnull().all():
                    def deserialize_params(x):
                        try:
                            return json.loads(x) if isinstance(x, str) else {}
                        except Exception:
                            return {}
                    df['parameters'] = df['parameters'].apply(deserialize_params)
        finally:
            conn.close()
            
        return df
    
    def get_strategy(self, strategy_id):
        """
        特定の戦略を取得
        
        Args:
            strategy_id (str): 戦略ID
            
        Returns:
            dict: 戦略情報
        """
        db_path = self.config['Paths']['database_path']
        if not os.path.exists(db_path):
            return None

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM strategies WHERE strategy_id = ?", (strategy_id,))
            result = cursor.fetchone()
        finally:
            conn.close()
        
        if result:
            strategy_dict = dict(result)
            
            # モデルファイルを読み込み
            model_path = strategy_dict.get('model_path')
            if model_path and os.path.exists(model_path):
                try:
                    with open(model_path, 'rb') as f:
                        strategy_dict['model'] = pickle.load(f)
                    logger.info(f"戦略 {strategy_dict.get('strategy_name')} のモデルを読み込みました")
                except Exception as e:
                    logger.error(f"モデルファイルの読み込みに失敗: {e}")
                    strategy_dict['model'] = None
            else:
                logger.warning(f"モデルファイルが見つかりません: {model_path}")
                strategy_dict['model'] = None
            
            if 'backtest_log' in strategy_dict and strategy_dict['backtest_log'] is not None:
                try:
                    strategy_dict['backtest_log'] = pickle.loads(strategy_dict['backtest_log'])
                except Exception:
                    strategy_dict['backtest_log'] = []
            if 'parameters' in strategy_dict and strategy_dict['parameters'] is not None:
                try:
                    strategy_dict['parameters'] = json.loads(strategy_dict['parameters'])
                except Exception:
                    strategy_dict['parameters'] = {}
            return strategy_dict
        
        return None
    
    def add_new_draw_result(self, draw_number, main_numbers, bonus_number, 
                           set_ball, total_sum=None):
        """
        新しい抽選結果を追加
        
        Args:
            draw_number (int): 回号
            main_numbers (list): 本数字リスト
            bonus_number (int): ボーナス数字
            set_ball (str): セット球
            total_sum (int): 合計値（Noneの場合は自動計算）
        """
        if total_sum is None:
            total_sum = sum(main_numbers)
        
        # 既存データの読み込み
        df = self.load_loto6_history()
        
        # 新しい行のデータを作成
        new_row = {
            '第何回': draw_number,
            '本数字合計': total_sum,
            'セット球': set_ball,
            'ボーナス数字': bonus_number
        }
        
        # 本数字の列を追加
        for i, num in enumerate(main_numbers, 1):
            new_row[f'本数字{i}'] = num
        
        # データフレームに追加
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        
        # ファイルに保存
        csv_path = self.config['Paths']['loto6_history_csv']
        df.to_csv(csv_path, index=False, encoding='shift-jis')
        
        logger.info(f"新しい抽選結果を追加しました: 第{draw_number}回")
