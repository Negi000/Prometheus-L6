"""
シミュレーション画面 (simulation.py)
戦略パフォーマンスの将来シミュレーション
"""

import def _show_single_strategy_simulation(data_manager: DataManager, strategies_df: pd.DataFrame):
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
    ''', unsafe_allow_html=True)mlit as st
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
    st.subheader("� 単一戦略シミュレーション")
    
    # 戦略選択
    strategy_options = {}
    for _, row in strategies_df.iterrows():
        display_name = f"{row['strategy_name']} (平均損益: {row['backtest_profit']:.0f}円)"
        strategy_options[display_name] = row['strategy_id']
    
    selected_strategy_display = st.selectbox(
        "シミュレーションする戦略を選択",
        list(strategy_options.keys())
    )
    
    selected_strategy_id = strategy_options[selected_strategy_display]
    strategy_info = data_manager.get_strategy(selected_strategy_id)
    
    # パラメータ設定
    col1, col2, col3 = st.columns(3)
    
    with col1:
        purchase_count = st.slider(
            "購入口数/回",
            min_value=5,
            max_value=100,
            value=20,
            help="1回の抽選での購入口数"
        )
    
    with col2:
        simulation_rounds = st.selectbox(
            "シミュレーション回数",
            [1000, 5000, 10000],
            index=1,
            help="シミュレーションの試行回数"
        )
    
    with col3:
        draws_per_session = st.slider(
            "期間（回数）",
            min_value=4,
            max_value=24,
            value=8,
            help="1セッションあたりの抽選回数（月間想定）"
        )
    
    # シミュレーション実行
    if st.button("🚀 シミュレーション実行", type="primary"):
        _execute_single_simulation(
            strategy_info, purchase_count, simulation_rounds, draws_per_session
        )

def _show_strategy_comparison(data_manager: DataManager, strategies_df: pd.DataFrame):
    """
    戦略比較シミュレーション
    """
    st.subheader("⚖️ 戦略比較シミュレーション")
    
    if len(strategies_df) < 2:
        st.info("比較には2つ以上の戦略が必要です。")
        return
    
    # 戦略選択
    strategy_names = strategies_df['strategy_name'].tolist()
    
    col1, col2 = st.columns(2)
    
    with col1:
        strategy1_name = st.selectbox(
            "戦略1を選択",
            strategy_names,
            key="strategy1"
        )
    
    with col2:
        available_strategy2 = [name for name in strategy_names if name != strategy1_name]
        strategy2_name = st.selectbox(
            "戦略2を選択",
            available_strategy2,
            key="strategy2"
        )
    
    # パラメータ設定
    col1, col2 = st.columns(2)
    
    with col1:
        purchase_count = st.slider(
            "購入口数/回",
            min_value=5,
            max_value=100,
            value=20,
            key="comp_purchase"
        )
    
    with col2:
        simulation_rounds = st.selectbox(
            "シミュレーション回数",
            [1000, 5000, 10000],
            index=1,
            key="comp_simulation"
        )
    
    # 比較実行
    if st.button("⚔️ 比較実行", type="primary"):
        strategy1_info = strategies_df[strategies_df['strategy_name'] == strategy1_name].iloc[0]
        strategy2_info = strategies_df[strategies_df['strategy_name'] == strategy2_name].iloc[0]
        
        _execute_comparison_simulation(
            strategy1_info, strategy2_info, purchase_count, simulation_rounds
        )

def _execute_single_simulation(strategy_info: dict, purchase_count: int, 
                              simulation_rounds: int, draws_per_session: int):
    """
    単一戦略のシミュレーション実行
    """
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("シミュレーションを実行中...")
        progress_bar.progress(50)
        
        # シミュレーター作成
        simulator = Simulator()
        
        # 戦略パフォーマンスの準備
        strategy_performance = {
            'hit_rate_3': strategy_info['backtest_hit_rate_3'],
            'hit_rate_4': strategy_info['backtest_hit_rate_4'],
            'hit_rate_5': strategy_info['backtest_hit_rate_5']
        }
        
        # シミュレーション実行
        results = simulator.run_monte_carlo_simulation(
            strategy_performance,
            purchase_count,
            simulation_rounds,
            draws_per_session
        )
        
        progress_bar.progress(100)
        status_text.text("完了!")
        
        # 結果表示
        _display_simulation_results(results, strategy_info['strategy_name'])
        
    except Exception as e:
        st.error(f"シミュレーション中にエラーが発生しました: {e}")
        logger.error(f"Simulation error: {e}")
    
    finally:
        if 'progress_bar' in locals():
            progress_bar.empty()
        if 'status_text' in locals():
            status_text.empty()

def _execute_comparison_simulation(strategy1_info: dict, strategy2_info: dict,
                                 purchase_count: int, simulation_rounds: int):
    """
    戦略比較シミュレーション実行
    """
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("比較シミュレーションを実行中...")
        progress_bar.progress(50)
        
        # シミュレーター作成
        simulator = Simulator()
        
        # 戦略パフォーマンスの準備
        strategy1_performance = {
            'hit_rate_3': strategy1_info['backtest_hit_rate_3'],
            'hit_rate_4': strategy1_info['backtest_hit_rate_4'],
            'hit_rate_5': strategy1_info['backtest_hit_rate_5']
        }
        
        strategy2_performance = {
            'hit_rate_3': strategy2_info['backtest_hit_rate_3'],
            'hit_rate_4': strategy2_info['backtest_hit_rate_4'],
            'hit_rate_5': strategy2_info['backtest_hit_rate_5']
        }
        
        # 比較シミュレーション実行
        comparison_results = simulator.compare_strategies(
            strategy1_performance,
            strategy2_performance,
            purchase_count,
            simulation_rounds
        )
        
        progress_bar.progress(100)
        status_text.text("完了!")
        
        # 比較結果表示
        _display_comparison_results(
            comparison_results,
            strategy1_info['strategy_name'],
            strategy2_info['strategy_name']
        )
        
    except Exception as e:
        st.error(f"比較シミュレーション中にエラーが発生しました: {e}")
        logger.error(f"Comparison simulation error: {e}")
    
    finally:
        if 'progress_bar' in locals():
            progress_bar.empty()
        if 'status_text' in locals():
            status_text.empty()

def _display_simulation_results(results: dict, strategy_name: str):
    """
    シミュレーション結果の表示
    """
    st.success(f"戦略「{strategy_name}」のシミュレーションが完了しました！")
    
    # 基本統計
    st.subheader("📊 基本統計")
    
    basic_stats = results['basic_stats']
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("平均損益", f"{basic_stats['mean_profit']:,.0f}円")
    
    with col2:
        st.metric("中央値損益", f"{basic_stats['median_profit']:,.0f}円")
    
    with col3:
        roi = results['investment_efficiency']['roi_mean']
        st.metric("平均ROI", f"{roi:.1f}%")
    
    with col4:
        win_prob = results['risk_metrics']['win_probability']
        st.metric("勝率", f"{win_prob:.1%}")
    
    # リスク指標
    st.subheader("⚠️ リスク指標")
    
    risk_metrics = results['risk_metrics']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("損失確率", f"{risk_metrics['loss_probability']:.1%}")
    
    with col2:
        st.metric("VaR (95%)", f"{risk_metrics['var_95']:,.0f}円")
    
    with col3:
        st.metric("VaR (99%)", f"{risk_metrics['var_99']:,.0f}円")
    
    # 当選確率
    st.subheader("🎯 当選確率")
    
    hit_probs = results['hit_probabilities']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("3等以上当選確率", f"{hit_probs['prob_3_or_above']:.1%}")
    
    with col2:
        st.metric("4等以上当選確率", f"{hit_probs['prob_4_or_above']:.1%}")
    
    # 損益分布のヒストグラム
    st.subheader("📈 損益分布")
    
    profits = results['distribution']['profit_distribution']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(profits, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax.axvline(basic_stats['mean_profit'], color='red', linestyle='--', 
               label=f"平均: {basic_stats['mean_profit']:,.0f}円")
    ax.axvline(basic_stats['median_profit'], color='blue', linestyle='--', 
               label=f"中央値: {basic_stats['median_profit']:,.0f}円")
    ax.axvline(0, color='black', linestyle='-', alpha=0.5, label="損益分岐点")
    
    ax.set_xlabel('損益 (円)')
    ax.set_ylabel('頻度')
    ax.set_title('損益分布ヒストグラム')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    
    # パーセンタイル情報
    st.subheader("📊 パーセンタイル情報")
    
    percentiles = results['distribution']['percentiles']
    
    percentile_data = {
        'パーセンタイル': ['10%', '25%', '50% (中央値)', '75%', '90%'],
        '損益 (円)': [
            f"{percentiles['10th']:,.0f}",
            f"{percentiles['25th']:,.0f}",
            f"{basic_stats['median_profit']:,.0f}",
            f"{percentiles['75th']:,.0f}",
            f"{percentiles['90th']:,.0f}"
        ]
    }
    
    st.table(pd.DataFrame(percentile_data))

def _display_comparison_results(comparison_results: dict, strategy1_name: str, strategy2_name: str):
    """
    戦略比較結果の表示
    """
    st.success("戦略比較シミュレーションが完了しました！")
    
    # 推奨結果
    recommendation = comparison_results['recommendation']
    
    if recommendation['recommended_strategy'] == 'strategy1':
        st.info(f"🏆 推奨戦略: **{strategy1_name}**")
    elif recommendation['recommended_strategy'] == 'strategy2':
        st.info(f"🏆 推奨戦略: **{strategy2_name}**")
    else:
        st.info("🤝 両戦略は同等の性能です")
    
    st.write(f"判定理由: {recommendation['reason']}")
    st.write(f"信頼度: {recommendation['confidence']:.1%}")
    
    # 比較表
    st.subheader("📊 戦略比較表")
    
    result1 = comparison_results['strategy1']
    result2 = comparison_results['strategy2']
    
    comparison_data = {
        '指標': [
            '平均損益 (円)',
            '中央値損益 (円)',
            '標準偏差 (円)',
            '勝率 (%)',
            '損失確率 (%)',
            'VaR 95% (円)',
            'ROI (%)',
            'シャープレシオ'
        ],
        strategy1_name: [
            f"{result1['basic_stats']['mean_profit']:,.0f}",
            f"{result1['basic_stats']['median_profit']:,.0f}",
            f"{result1['basic_stats']['std_profit']:,.0f}",
            f"{result1['risk_metrics']['win_probability']:.1%}",
            f"{result1['risk_metrics']['loss_probability']:.1%}",
            f"{result1['risk_metrics']['var_95']:,.0f}",
            f"{result1['investment_efficiency']['roi_mean']:.1f}",
            f"{comparison_results['comparison']['sharpe_ratio_1']:.3f}"
        ],
        strategy2_name: [
            f"{result2['basic_stats']['mean_profit']:,.0f}",
            f"{result2['basic_stats']['median_profit']:,.0f}",
            f"{result2['basic_stats']['std_profit']:,.0f}",
            f"{result2['risk_metrics']['win_probability']:.1%}",
            f"{result2['risk_metrics']['loss_probability']:.1%}",
            f"{result2['risk_metrics']['var_95']:,.0f}",
            f"{result2['investment_efficiency']['roi_mean']:.1f}",
            f"{comparison_results['comparison']['sharpe_ratio_2']:.3f}"
        ]
    }
    
    st.table(pd.DataFrame(comparison_data))
    
    # 損益分布の比較グラフ
    st.subheader("📈 損益分布比較")
    
    profits1 = result1['distribution']['profit_distribution']
    profits2 = result2['distribution']['profit_distribution']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.hist(profits1, bins=30, alpha=0.5, label=strategy1_name, color='blue')
    ax.hist(profits2, bins=30, alpha=0.5, label=strategy2_name, color='red')
    
    ax.axvline(result1['basic_stats']['mean_profit'], color='blue', linestyle='--', alpha=0.8)
    ax.axvline(result2['basic_stats']['mean_profit'], color='red', linestyle='--', alpha=0.8)
    ax.axvline(0, color='black', linestyle='-', alpha=0.5, label="損益分岐点")
    
    ax.set_xlabel('損益 (円)')
    ax.set_ylabel('頻度')
    ax.set_title('戦略別損益分布比較')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)
