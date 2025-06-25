"""
予測生成画面 (prediction.py)
学習済み戦略を使用したポートフォリオ生成
"""

import streamlit as st
import pandas as pd
import numpy as np
import logging
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from data_manager import DataManager
from strategy import Strategy
from feature_engine import FeatureEngine
from backtester import Backtester

logger = logging.getLogger(__name__)

def show_prediction(data_manager: DataManager):
    """
    予測生成画面の表示
    
    Args:
        data_manager (DataManager): データ管理インスタンス
    """
    # SVG Icons CSS
    st.markdown("""
    <style>
        .prediction-icon { width: 20px; height: 20px; }
        .section-header {
            display: flex;
            align-items: center;
            gap: 8px;
            margin: 20px 0 16px 0;
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('''
    <h2 style="display: flex; align-items: center; gap: 12px; margin-bottom: 20px;">
        <svg class="prediction-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="12" cy="12" r="10"/>
            <path d="m16 12-4-4-4 4"/>
            <path d="m12 16 4-4-4-4"/>
        </svg>
        予測・ポートフォリオ生成
    </h2>
    ''', unsafe_allow_html=True)
    
    # 戦略の選択
    strategies_df = data_manager.get_all_strategies()
    
    if len(strategies_df) == 0:
        st.warning("利用可能な戦略がありません。まず「戦略管理」で戦略を学習してください。")
        return
    
    st.markdown('''
    <h3 class="section-header">
        <svg class="prediction-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="12" cy="12" r="3"/>
            <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1 1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/>
        </svg>
        設定
    </h3>
    ''', unsafe_allow_html=True)
    
    # ステップ1: 戦略の選択
    with st.container():
        st.write("### 1. 戦略の選択")
        
        strategy_options = {}
        for _, row in strategies_df.iterrows():
            display_name = f"{row['strategy_name']} (4等以上: {row['backtest_hit_rate_4']:.1%})"
            strategy_options[display_name] = row['strategy_id']
        
        selected_strategy_display = st.selectbox(
            "使用する戦略を選択",
            list(strategy_options.keys()),
            help="バックテスト結果の良い戦略を選択してください"
        )
        
        selected_strategy_id = strategy_options[selected_strategy_display]
        selected_strategy = data_manager.get_strategy(selected_strategy_id)
        
        # 選択した戦略の詳細表示
        if selected_strategy:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("平均損益", f"{selected_strategy['backtest_profit']:.0f}円")
            with col2:
                st.metric("4等以上的中率", f"{selected_strategy['backtest_hit_rate_4']:.1%}")
            with col3:
                st.metric("3等以上的中率", f"{selected_strategy['backtest_hit_rate_3']:.1%}")
    
    # ステップ2: 購入条件の設定
    with st.container():
        st.write("### 2. 購入条件の設定")
        
        col1, col2 = st.columns(2)
        
        with col1:
            purchase_count = st.number_input(
                "購入口数",
                min_value=1,
                max_value=100,
                value=20,
                help="購入する組み合わせの数"
            )
            
            axis_numbers_input = st.text_input(
                "▶ 軸数字（必須含有数字）",
                placeholder="例: 7, 21, 33",
                help="カンマ区切りで指定。これらの数字を必ず含む組み合わせのみ生成"
            )
        
        with col2:
            exclude_numbers_input = st.text_input(
                "× 除外数字",
                placeholder="例: 4, 15, 28",
                help="カンマ区切りで指定。これらの数字を含む組み合わせを除外"
            )
            
            total_bankroll = st.number_input(
                "¥ 総資金（ケリー基準用）",
                min_value=0,
                value=100000,
                step=10000,
                help="ケリー基準による推奨口数計算に使用"
            )
    
    # ステップ3: 高度なオプション
    with st.expander("3. 高度なオプション"):
        col1, col2 = st.columns(2)
        
        with col1:
            enable_kelly = st.checkbox(
                "ケリー基準で推奨口数を計算",
                help="資金管理理論に基づいた最適な投資額を提案"
            )
            
            enable_contrarian = st.checkbox(
                "逆張り戦略を有効化",
                help="人気のない組み合わせを優遇して期待収益を最大化"
            )
        
        with col2:
            if enable_contrarian:
                contrarian_weight = st.slider(
                    "逆張り強度",
                    min_value=0.0,
                    max_value=2.0,
                    value=0.5,
                    step=0.1,
                    help="大きいほど逆張り効果が強くなる"
                )
            else:
                contrarian_weight = 0.0
    
    # 生成実行ボタン
    if st.button("🚀 ポートフォリオ生成", type="primary"):
        _generate_portfolio(
            data_manager, selected_strategy, purchase_count,
            axis_numbers_input, exclude_numbers_input, 
            enable_kelly, enable_contrarian, contrarian_weight,
            total_bankroll
        )

def _generate_portfolio(data_manager: DataManager, strategy_info: dict, 
                       purchase_count: int, axis_numbers_input: str, 
                       exclude_numbers_input: str, enable_kelly: bool,
                       enable_contrarian: bool, contrarian_weight: float,
                       total_bankroll: float):
    """
    ポートフォリオを生成
    """
    try:
        # 進捗表示
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("データを準備中...")
        progress_bar.progress(20)
        
        # 入力データの解析
        axis_numbers = _parse_number_input(axis_numbers_input)
        exclude_numbers = _parse_number_input(exclude_numbers_input)
        
        # LOTO6履歴データの読み込み
        df_history = data_manager.load_loto6_history()
        
        status_text.text("特徴量を生成中...")
        progress_bar.progress(40)
        
        # 特徴量生成
        feature_engine = FeatureEngine(df_history)
        df_features = feature_engine.run_all()
        
        status_text.text("AIモデルを読み込み中...")
        progress_bar.progress(60)
        
        # 学習済みモデルの読み込み
        backtester = Backtester()
        model = backtester.load_model(strategy_info['model_path'])
        
        # 最新データで予測
        latest_data = df_features.tail(1)
        feature_cols = feature_engine.get_feature_columns()
        X_latest = latest_data[feature_cols].fillna(0)
        
        status_text.text("予測を実行中...")
        progress_bar.progress(80)
        
        # 各数字の出現確率を予測
        predicted_probabilities = model.predict(X_latest)[0]
        
        # 戦略インスタンスの作成
        strategy = Strategy()
        
        # ケリー基準の計算
        kelly_info = None
        if enable_kelly:
            kelly_info = strategy.calculate_kelly_criterion(
                {
                    'hit_rate_4': strategy_info['backtest_hit_rate_4'],
                    'hit_rate_3': strategy_info['backtest_hit_rate_3']
                },
                total_bankroll
            )
        
        # ポートフォリオ生成
        portfolio = strategy.generate_from_probabilities(
            predicted_probabilities,
            purchase_count,
            axis_numbers,
            exclude_numbers,
            enable_contrarian
        )
        
        progress_bar.progress(100)
        status_text.text("完了!")
        
        # 結果表示
        _display_results(portfolio, predicted_probabilities, kelly_info, 
                        strategy_info, axis_numbers, exclude_numbers)
        
    except Exception as e:
        st.error(f"ポートフォリオ生成中にエラーが発生しました: {e}")
        logger.error(f"Portfolio generation error: {e}")
    
    finally:
        if 'progress_bar' in locals():
            progress_bar.empty()
        if 'status_text' in locals():
            status_text.empty()

def _parse_number_input(input_string: str) -> list:
    """
    カンマ区切りの数字文字列をリストに変換
    """
    if not input_string:
        return []
    
    try:
        numbers = [int(x.strip()) for x in input_string.split(',') if x.strip()]
        # 1-43の範囲チェック
        valid_numbers = [n for n in numbers if 1 <= n <= 43]
        return valid_numbers
    except ValueError:
        st.error("数字は1-43の範囲でカンマ区切りで入力してください")
        return []

def _display_results(portfolio: dict, probabilities: np.ndarray, 
                    kelly_info: dict, strategy_info: dict,
                    axis_numbers: list, exclude_numbers: list):
    """
    生成結果を表示
    """
    st.success("ポートフォリオの生成が完了しました！")
    
    # ケリー基準の結果表示
    if kelly_info:
        st.subheader("💰 ケリー基準による推奨投資")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("推奨投資額", f"{kelly_info['recommended_investment']:,.0f}円")
        with col2:
            st.metric("推奨口数", f"{kelly_info['recommended_tickets']}口")
        with col3:
            risk_color = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}
            st.metric("リスクレベル", f"{risk_color.get(kelly_info['risk_level'], '🔴')} {kelly_info['risk_level']}")
    
    # ポートフォリオの表示
    st.subheader("📊 生成されたポートフォリオ")
    
    # 設定条件の確認
    conditions = []
    if axis_numbers:
        conditions.append(f"軸数字: {', '.join(map(str, axis_numbers))}")
    if exclude_numbers:
        conditions.append(f"除外数字: {', '.join(map(str, exclude_numbers))}")
    
    if conditions:
        st.info(f"適用条件: {' | '.join(conditions)}")
    
    # コア戦略の表示
    if 'core' in portfolio and portfolio['core']:
        st.write("### 🎯 コア戦略 (メイン戦略)")
        _display_portfolio_table(portfolio['core'], probabilities, "core")
    
    # サテライト戦略の表示
    if 'satellite' in portfolio and portfolio['satellite']:
        st.write("### 🛰️ サテライト戦略 (補完戦略)")
        _display_portfolio_table(portfolio['satellite'], probabilities, "satellite")
    
    # 全体のコピー・保存機能
    st.subheader("💾 結果の出力")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # クリップボード用テキスト
        clipboard_text = _format_for_clipboard(portfolio)
        st.text_area("クリップボード用テキスト", clipboard_text, height=200)
    
    with col2:
        # CSV保存用データ
        csv_data = _format_for_csv(portfolio, probabilities)
        st.download_button(
            label="📁 CSV形式でダウンロード",
            data=csv_data,
            file_name=f"loto6_portfolio_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )
    
    # AI予測スコア上位数字の表示
    st.subheader("🤖 AI予測スコア上位数字")
    
    top_numbers = np.argsort(probabilities)[-20:][::-1]  # 上位20個を降順
    top_scores = [(num + 1, probabilities[num]) for num in top_numbers]
    
    score_df = pd.DataFrame(top_scores, columns=['数字', 'AIスコア'])
    score_df['順位'] = range(1, len(score_df) + 1)
    score_df = score_df[['順位', '数字', 'AIスコア']]
    score_df['AIスコア'] = score_df['AIスコア'].round(4)
    
    st.dataframe(score_df, use_container_width=True)

def _display_portfolio_table(combinations: list, probabilities: np.ndarray, strategy_type: str):
    """
    ポートフォリオテーブルを表示
    """
    if not combinations:
        st.info("該当する組み合わせがありません")
        return
    
    # テーブル用データの準備
    table_data = []
    
    for i, combo in enumerate(combinations, 1):
        numbers_str = ' - '.join(f"{num:02d}" for num in sorted(combo))
        ai_score = sum(probabilities[num - 1] for num in combo)
        total_sum = sum(combo)
        odd_count = sum(1 for num in combo if num % 2 == 1)
        even_count = 6 - odd_count
        
        table_data.append({
            '番号': i,
            '組み合わせ': numbers_str,
            'AIスコア': round(ai_score, 4),
            '合計': total_sum,
            '奇偶': f"{odd_count}:{even_count}"
        })
    
    df_table = pd.DataFrame(table_data)
    st.dataframe(df_table, use_container_width=True)

def _format_for_clipboard(portfolio: dict) -> str:
    """
    クリップボード用のテキストフォーマット
    """
    lines = []
    lines.append("=== LOTO6 ポートフォリオ ===")
    lines.append(f"生成日時: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    
    for strategy_type, combinations in portfolio.items():
        if combinations:
            type_name = "コア戦略" if strategy_type == "core" else "サテライト戦略"
            lines.append(f"■ {type_name} ({len(combinations)}口)")
            
            for i, combo in enumerate(combinations, 1):
                numbers_str = ' - '.join(f"{num:02d}" for num in sorted(combo))
                lines.append(f"{i:2d}. {numbers_str}")
            
            lines.append("")
    
    return '\n'.join(lines)

def _format_for_csv(portfolio: dict, probabilities: np.ndarray) -> str:
    """
    CSV保存用のデータフォーマット
    """
    rows = []
    rows.append("戦略,番号,数字1,数字2,数字3,数字4,数字5,数字6,AIスコア,合計,奇偶比")
    
    for strategy_type, combinations in portfolio.items():
        strategy_name = "コア戦略" if strategy_type == "core" else "サテライト戦略"
        
        for i, combo in enumerate(combinations, 1):
            sorted_combo = sorted(combo)
            ai_score = sum(probabilities[num - 1] for num in combo)
            total_sum = sum(combo)
            odd_count = sum(1 for num in combo if num % 2 == 1)
            odd_even = f"{odd_count}:{6-odd_count}"
            
            row = [
                strategy_name,
                i,
                *sorted_combo,
                round(ai_score, 4),
                total_sum,
                odd_even
            ]
            
            rows.append(','.join(map(str, row)))
    
    return '\n'.join(rows)
