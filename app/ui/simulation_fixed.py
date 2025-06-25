"""
シミュレーション画面 (simulation.py)
戦略パフォーマンスの将来シミュレーション
"""

import streamlit as st
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from data_manager import DataManager
from simulator import Simulator

logger = logging.getLogger(__name__)

def show_simulation(data_manager: DataManager):
    """
    シミュレーション画面の表示
    
    Args:
        data_manager (DataManager): データ管理インスタンス
    """
    # SVG Icons CSS
    st.markdown("""
    <style>
        .simulation-icon { width: 20px; height: 20px; }
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
        <svg class="simulation-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M3 3v18h18"/>
            <path d="m19 9-5 5-4-4-3 3"/>
        </svg>
        シミュレーション
    </h2>
    ''', unsafe_allow_html=True)
    
    # 戦略一覧の取得
    strategies_df = data_manager.get_all_strategies()
    
    if len(strategies_df) == 0:
        st.warning("利用可能な戦略がありません。まず「戦略管理」で戦略を学習してください。")
        return
    
    # タブ分割
    tab1, tab2 = st.tabs([
        "📊 単一戦略", 
        "🔄 戦略比較"
    ])
    
    with tab1:
        _show_single_strategy_simulation(data_manager, strategies_df)
    
    with tab2:
        _show_strategy_comparison(data_manager, strategies_df)

def _show_single_strategy_simulation(data_manager: DataManager, strategies_df: pd.DataFrame):
    """
    単一戦略のシミュレーション
    """
    st.markdown('''
    <h3 class="section-header">
        <svg class="simulation-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/>
            <line x1="9" y1="9" x2="9" y2="15"/>
            <line x1="15" y1="9" x2="15" y2="15"/>
        </svg>
        単一戦略シミュレーション
    </h3>
    ''', unsafe_allow_html=True)
    
    # 戦略選択
    strategy_options = {}
    for _, row in strategies_df.iterrows():
        display_name = f"{row['strategy_name']} (損益: {row['backtest_profit']:.0f}円)"
        strategy_options[display_name] = row['strategy_id']
    
    selected_strategy_display = st.selectbox(
        "シミュレーションする戦略を選択",
        list(strategy_options.keys())
    )
    
    # パラメータ設定
    col1, col2, col3 = st.columns(3)
    
    with col1:
        purchase_count = st.slider(
            "購入口数",
            min_value=1,
            max_value=100,
            value=20,
            help="1回あたりの購入口数"
        )
    
    with col2:
        simulation_rounds = st.selectbox(
            "シミュレーション回数",
            [100, 500, 1000, 5000],
            index=1,
            help="モンテカルロシミュレーションの試行回数"
        )
    
    with col3:
        draws_per_session = st.slider(
            "投資期間（抽選回数）",
            min_value=10,
            max_value=100,
            value=50,
            help="1回のシミュレーションで何回分の抽選を行うか"
        )
    
    if st.button("🚀 シミュレーション実行", type="primary"):
        _run_single_strategy_simulation(
            data_manager, 
            strategy_options[selected_strategy_display],
            purchase_count,
            simulation_rounds,
            draws_per_session
        )

def _show_strategy_comparison(data_manager: DataManager, strategies_df: pd.DataFrame):
    """
    戦略比較シミュレーション
    """
    st.markdown('''
    <h3 class="section-header">
        <svg class="simulation-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M9 12l2 2 4-4"/>
            <circle cx="21" cy="9" r="1"/>
            <circle cx="3" cy="9" r="1"/>
            <path d="M3 9h18"/>
            <circle cx="21" cy="15" r="1"/>
            <circle cx="3" cy="15" r="1"/>
            <path d="M3 15h18"/>
        </svg>
        戦略比較シミュレーション
    </h3>
    ''', unsafe_allow_html=True)
    
    if len(strategies_df) < 2:
        st.info("比較には2つ以上の戦略が必要です。")
        return
    
    # 比較する戦略の選択
    st.write("比較する戦略を選択してください（複数選択可能）：")
    
    selected_strategies = []
    for _, row in strategies_df.iterrows():
        if st.checkbox(
            f"{row['strategy_name']} (損益: {row['backtest_profit']:.0f}円)",
            key=f"compare_{row['strategy_id']}"
        ):
            selected_strategies.append(row['strategy_id'])
    
    if len(selected_strategies) < 2:
        st.warning("比較には2つ以上の戦略を選択してください。")
        return
    
    # 比較パラメータ
    col1, col2 = st.columns(2)
    
    with col1:
        comparison_rounds = st.selectbox(
            "比較シミュレーション回数",
            [100, 500, 1000],
            index=1
        )
    
    with col2:
        comparison_period = st.slider(
            "比較期間（抽選回数）",
            min_value=20,
            max_value=100,
            value=50
        )
    
    if st.button("📊 戦略比較実行", type="primary"):
        _run_strategy_comparison(
            data_manager,
            selected_strategies,
            comparison_rounds,
            comparison_period
        )

def _run_single_strategy_simulation(data_manager: DataManager, strategy_id: str, 
                                  purchase_count: int, simulation_rounds: int, 
                                  draws_per_session: int):
    """
    単一戦略のシミュレーション実行
    """
    try:
        with st.spinner("シミュレーション実行中..."):
            # シミュレーターの初期化
            simulator = Simulator(data_manager)
            
            # シミュレーション実行
            results = simulator.run_monte_carlo_simulation(
                strategy_id=strategy_id,
                num_simulations=simulation_rounds,
                draws_per_simulation=draws_per_session,
                purchase_count=purchase_count
            )
            
            # 結果の表示
            _display_simulation_results(results, draws_per_session)
            
    except Exception as e:
        st.error(f"シミュレーション実行中にエラーが発生しました: {e}")
        logger.error(f"Simulation error: {e}")

def _run_strategy_comparison(data_manager: DataManager, strategy_ids: list, 
                           simulation_rounds: int, comparison_period: int):
    """
    戦略比較シミュレーション実行
    """
    try:
        with st.spinner("戦略比較シミュレーション実行中..."):
            simulator = Simulator(data_manager)
            
            comparison_results = {}
            
            for strategy_id in strategy_ids:
                results = simulator.run_monte_carlo_simulation(
                    strategy_id=strategy_id,
                    num_simulations=simulation_rounds,
                    draws_per_simulation=comparison_period,
                    purchase_count=20  # 固定値
                )
                
                strategy_info = data_manager.get_strategy_by_id(strategy_id)
                strategy_name = strategy_info['strategy_name'] if strategy_info else f"Strategy_{strategy_id}"
                comparison_results[strategy_name] = results
            
            # 比較結果の表示
            _display_comparison_results(comparison_results, comparison_period)
            
    except Exception as e:
        st.error(f"戦略比較実行中にエラーが発生しました: {e}")
        logger.error(f"Strategy comparison error: {e}")

def _display_simulation_results(results: dict, draws_per_session: int):
    """
    シミュレーション結果の表示
    """
    st.markdown('''
    <h3 class="section-header">
        <svg class="simulation-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/>
            <line x1="9" y1="9" x2="15" y2="15"/>
            <line x1="15" y1="9" x2="9" y2="15"/>
        </svg>
        シミュレーション結果
    </h3>
    ''', unsafe_allow_html=True)
    
    if not results or 'final_profits' not in results:
        st.error("シミュレーション結果が無効です。")
        return
    
    final_profits = results['final_profits']
    
    # 統計情報
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("平均損益", f"{np.mean(final_profits):.0f}円")
    
    with col2:
        st.metric("中央値", f"{np.median(final_profits):.0f}円")
    
    with col3:
        st.metric("標準偏差", f"{np.std(final_profits):.0f}円")
    
    with col4:
        profit_ratio = len([p for p in final_profits if p > 0]) / len(final_profits)
        st.metric("利益確率", f"{profit_ratio:.1%}")
    
    # 損益分布のヒストグラム
    st.subheader("損益分布")
    
    import plotly.express as px
    fig = px.histogram(
        x=final_profits,
        nbins=50,
        title=f"最終損益の分布 ({len(final_profits)}回シミュレーション)",
        labels={'x': '最終損益(円)', 'y': '頻度'},
        color_discrete_sequence=['#3b82f6']
    )
    fig.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="損益分岐点")
    fig.add_vline(x=np.mean(final_profits), line_dash="dash", line_color="green", annotation_text="平均値")
    
    st.plotly_chart(fig, use_container_width=True)
    
    # リスク分析
    st.subheader("リスク分析")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # VaR計算（Value at Risk）
        var_95 = np.percentile(final_profits, 5)
        var_99 = np.percentile(final_profits, 1)
        
        st.write("**VaR (Value at Risk)**")
        st.write(f"- 95% VaR: {var_95:.0f}円")
        st.write(f"- 99% VaR: {var_99:.0f}円")
        
        st.info("VaRは、指定した確率で発生しうる最大損失額を示します。")
    
    with col2:
        # 最大ドローダウン分析
        if 'profit_history' in results:
            max_drawdown = _calculate_max_drawdown(results['profit_history'])
            st.write("**ドローダウン分析**")
            st.write(f"- 最大ドローダウン: {max_drawdown:.0f}円")
        
        # 破産確率
        ruin_probability = len([p for p in final_profits if p < -1000000]) / len(final_profits)
        st.write(f"- 100万円以上の損失確率: {ruin_probability:.1%}")

def _display_comparison_results(comparison_results: dict, comparison_period: int):
    """
    戦略比較結果の表示
    """
    st.markdown('''
    <h3 class="section-header">
        <svg class="simulation-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/>
            <line x1="9" y1="9" x2="15" y2="15"/>
            <line x1="15" y1="9" x2="9" y2="15"/>
        </svg>
        戦略比較結果
    </h3>
    ''', unsafe_allow_html=True)
    
    # 比較統計表
    comparison_stats = []
    
    for strategy_name, results in comparison_results.items():
        final_profits = results['final_profits']
        
        comparison_stats.append({
            '戦略名': strategy_name,
            '平均損益': np.mean(final_profits),
            '中央値': np.median(final_profits),
            '標準偏差': np.std(final_profits),
            '利益確率': len([p for p in final_profits if p > 0]) / len(final_profits),
            'シャープ比': np.mean(final_profits) / np.std(final_profits) if np.std(final_profits) > 0 else 0
        })
    
    df_comparison = pd.DataFrame(comparison_stats)
    
    # 数値フォーマット
    df_display = df_comparison.copy()
    df_display['平均損益'] = df_display['平均損益'].apply(lambda x: f"{x:.0f}円")
    df_display['中央値'] = df_display['中央値'].apply(lambda x: f"{x:.0f}円")
    df_display['標準偏差'] = df_display['標準偏差'].apply(lambda x: f"{x:.0f}円")
    df_display['利益確率'] = df_display['利益確率'].apply(lambda x: f"{x:.1%}")
    df_display['シャープ比'] = df_display['シャープ比'].apply(lambda x: f"{x:.3f}")
    
    st.dataframe(df_display, use_container_width=True)
    
    # 比較チャート
    st.subheader("損益分布比較")
    
    import plotly.graph_objects as go
    fig = go.Figure()
    
    for strategy_name, results in comparison_results.items():
        fig.add_trace(go.Histogram(
            x=results['final_profits'],
            name=strategy_name,
            opacity=0.7,
            nbinsx=30
        ))
    
    fig.update_layout(
        title="戦略別最終損益分布",
        xaxis_title="最終損益(円)",
        yaxis_title="頻度",
        barmode='overlay'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def _calculate_max_drawdown(profit_history: list) -> float:
    """
    最大ドローダウンを計算
    """
    if not profit_history:
        return 0.0
    
    cumulative = np.cumsum(profit_history)
    peak = np.maximum.accumulate(cumulative)
    drawdown = peak - cumulative
    
    return np.max(drawdown)
