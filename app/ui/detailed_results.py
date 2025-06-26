"""
詳細結果表示モジュール (detailed_results.py)
バックテスト結果の視覚的で直感的な表示機能
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

def show_detailed_backtest_results(performance_log: List[Dict], strategy_name: str = "戦略"):
    """
    バックテスト結果の詳細表示（視覚的・直感的）
    
    Args:
        performance_log: バックテスト結果ログ
        strategy_name: 戦略名
    """
    if not performance_log:
        st.warning("表示する結果がありません。")
        return
    
    st.markdown(f"## 📊 {strategy_name} - 詳細バックテスト結果")
    
    # タブ分けで整理
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 パフォーマンス概要", 
        "🎯 回別詳細結果", 
        "🎲 予想vs実際", 
        "💰 損益分析"
    ])
    
    with tab1:
        _show_performance_overview(performance_log)
    
    with tab2:
        _show_detailed_by_draw(performance_log)
    
    with tab3:
        _show_prediction_analysis(performance_log)
    
    with tab4:
        _show_profit_analysis(performance_log)

def _show_performance_overview(performance_log: List[Dict]):
    """パフォーマンス概要の表示"""
    st.markdown("### 📊 全体パフォーマンス")
    
    # 基本統計
    total_profit = sum(log['profit'] for log in performance_log)
    total_cost = sum(log['cost'] for log in performance_log)
    total_winnings = sum(log['winnings'] for log in performance_log)
    win_rate = len([log for log in performance_log if log['profit'] > 0]) / len(performance_log)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("総損益", f"{total_profit:,}円", 
                 delta=f"ROI: {(total_profit/total_cost)*100:.1f}%" if total_cost > 0 else "")
    with col2:
        st.metric("勝率", f"{win_rate:.1%}")
    with col3:
        st.metric("総当選金額", f"{total_winnings:,}円")
    with col4:
        st.metric("実行回数", f"{len(performance_log)}回")
    
    # 損益チャート
    df_chart = pd.DataFrame(performance_log)
    df_chart['cumulative_profit'] = df_chart['profit'].cumsum()
    df_chart['draw_id'] = df_chart['draw_id'].astype(str)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 累積損益チャート
        fig_cumulative = px.line(
            df_chart, x='draw_id', y='cumulative_profit',
            title='📈 累積損益の推移',
            labels={'draw_id': '回号', 'cumulative_profit': '累積損益(円)'}
        )
        fig_cumulative.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig_cumulative.update_traces(line_color='#1f77b4', line_width=3)
        st.plotly_chart(fig_cumulative, use_container_width=True)
    
    with col2:
        # 等級別的中分布
        hit_data = []
        for log in performance_log:
            for rank, count in log['hits_detail'].items():
                if count > 0:
                    hit_data.extend([f'{rank}等'] * count)
        
        if hit_data:
            hit_df = pd.DataFrame(hit_data, columns=['等級'])
            hit_counts = hit_df['等級'].value_counts()
            
            fig_hits = px.pie(
                values=hit_counts.values, 
                names=hit_counts.index,
                title='🎯 等級別的中分布'
            )
            st.plotly_chart(fig_hits, use_container_width=True)
        else:
            st.info("当選実績がありません")

def _show_detailed_by_draw(performance_log: List[Dict]):
    """回別詳細結果の表示"""
    st.markdown("### 🎯 回別詳細結果")
    
    # フィルター機能
    col1, col2, col3 = st.columns(3)
    with col1:
        show_wins_only = st.checkbox("当選回のみ表示")
    with col2:
        show_count = st.selectbox("表示件数", [10, 20, 50, 100, "全て"], index=1)
    with col3:
        sort_by = st.selectbox("並び順", ["新しい順", "古い順", "損益順", "当選順"])
    
    # データの準備
    display_logs = performance_log.copy()
    
    if show_wins_only:
        display_logs = [log for log in display_logs if log['profit'] > 0]
    
    # ソート
    if sort_by == "新しい順":
        display_logs = sorted(display_logs, key=lambda x: x['draw_id'], reverse=True)
    elif sort_by == "古い順":
        display_logs = sorted(display_logs, key=lambda x: x['draw_id'])
    elif sort_by == "損益順":
        display_logs = sorted(display_logs, key=lambda x: x['profit'], reverse=True)
    elif sort_by == "当選順":
        display_logs = sorted(display_logs, key=lambda x: sum(x['hits_detail'].values()), reverse=True)
    
    # 表示件数制限
    if show_count != "全て":
        display_logs = display_logs[:int(show_count)]
    
    if not display_logs:
        st.warning("表示する結果がありません。")
        return
    
    # 結果表示
    for i, log in enumerate(display_logs):
        draw_id = log['draw_id']
        profit = log['profit']
        actual = log['actual_numbers']
        portfolio = log.get('predicted_portfolio', [])
        hits = log['hits_detail']
        
        # 利益に応じた色分け
        if profit > 10000:
            color = "🟢"  # 大勝
        elif profit > 0:
            color = "🔵"  # 小勝
        elif profit == 0:
            color = "⚪"  # 損益なし
        else:
            color = "🔴"  # 損失
        
        # 当選情報
        hit_info = []
        for rank, count in hits.items():
            if count > 0:
                hit_info.append(f"{rank}等×{count}")
        hit_text = " ".join(hit_info) if hit_info else "当選なし"
        
        with st.expander(f"{color} 第{draw_id}回 - 損益: {profit:,}円 - {hit_text}"):
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("#### 🎱 実際の当選番号")
                main_numbers = actual.get('main', [])
                bonus_number = actual.get('bonus')
                
                # 本数字を視覚的に表示
                if main_numbers:
                    main_str = "　".join([f"**{num:02d}**" for num in sorted(main_numbers)])
                    st.markdown(f"**本数字:** {main_str}")
                
                if bonus_number:
                    st.markdown(f"**ボーナス数字:** **{bonus_number:02d}**")
                
                # 当選詳細
                st.markdown("#### 🏆 当選詳細")
                for rank in range(1, 6):
                    count = hits.get(rank, 0)
                    if count > 0:
                        prizes = {1: "2億円", 2: "1000万円", 3: "30万円", 4: "6800円", 5: "1000円"}
                        st.success(f"{rank}等: {count}口 ({prizes[rank]})")
                
                if sum(hits.values()) == 0:
                    st.info("当選なし")
            
            with col2:
                st.markdown("#### 🎯 予想チケット")
                
                if portfolio:
                    # 全ての予想チケットを表示
                    for j, ticket in enumerate(portfolio):
                        ticket_profit = 0
                        ticket_hits = []
                        
                        # このチケットの当選状況を計算
                        if main_numbers:
                            main_set = set(main_numbers)
                            ticket_set = set(ticket)
                            match_main = len(ticket_set.intersection(main_set))
                            match_bonus = 1 if bonus_number and bonus_number in ticket_set else 0
                            
                            # 等級判定
                            if match_main == 6:
                                ticket_hits.append("1等")
                                ticket_profit += 200000000
                            elif match_main == 5 and match_bonus == 1:
                                ticket_hits.append("2等")
                                ticket_profit += 10000000
                            elif match_main == 5:
                                ticket_hits.append("3等")
                                ticket_profit += 300000
                            elif match_main == 4:
                                ticket_hits.append("4等")
                                ticket_profit += 6800
                            elif match_main == 3:
                                ticket_hits.append("5等")
                                ticket_profit += 1000
                        
                        # チケット表示
                        ticket_str = "　".join([f"{num:02d}" for num in sorted(ticket)])
                        hit_status = " ".join(ticket_hits) if ticket_hits else ""
                        
                        if ticket_hits:
                            st.success(f"**チケット{j+1}:** {ticket_str} → {hit_status}")
                        else:
                            st.info(f"**チケット{j+1}:** {ticket_str}")
                
                # 的中数字のハイライト
                if main_numbers and portfolio:
                    st.markdown("#### ✨ 的中数字分析")
                    all_predicted = set()
                    for ticket in portfolio:
                        all_predicted.update(ticket)
                    
                    hit_numbers = set(main_numbers).intersection(all_predicted)
                    if bonus_number and bonus_number in all_predicted:
                        hit_numbers.add(f"{bonus_number}(B)")
                    
                    if hit_numbers:
                        hit_str = "　".join([f"**{num}**" for num in sorted(hit_numbers) if isinstance(num, int)])
                        if bonus_number and bonus_number in all_predicted:
                            hit_str += f"　**{bonus_number}(B)**"
                        st.markdown(f"予想的中: {hit_str}")
                    else:
                        st.info("予想数字の的中なし")

def _show_prediction_analysis(performance_log: List[Dict]):
    """予想vs実際の分析表示"""
    st.markdown("### 🎲 予想vs実際の分析")
    
    # 数字別的中分析
    number_stats = {}
    for i in range(1, 44):
        number_stats[i] = {
            'predicted_count': 0,
            'hit_count': 0,
            'as_bonus_hit': 0
        }
    
    total_predictions = 0
    
    for log in performance_log:
        actual_main = set(log['actual_numbers'].get('main', []))
        actual_bonus = log['actual_numbers'].get('bonus')
        portfolio = log.get('predicted_portfolio', [])
        
        # 予想数字の集計
        predicted_numbers = set()
        for ticket in portfolio:
            predicted_numbers.update(ticket)
            total_predictions += len(ticket)
        
        for num in predicted_numbers:
            if 1 <= num <= 43:
                number_stats[num]['predicted_count'] += 1
                
                # 的中判定
                if num in actual_main:
                    number_stats[num]['hit_count'] += 1
                elif num == actual_bonus:
                    number_stats[num]['as_bonus_hit'] += 1
    
    # 数字別分析表の作成
    analysis_data = []
    for num in range(1, 44):
        stats = number_stats[num]
        predicted = stats['predicted_count']
        hit_rate = (stats['hit_count'] + stats['as_bonus_hit']) / predicted if predicted > 0 else 0
        
        analysis_data.append({
            '数字': num,
            '予想回数': predicted,
            '本数字的中': stats['hit_count'],
            'ボーナス的中': stats['as_bonus_hit'],
            '的中率': f"{hit_rate:.1%}",
            '的中率_数値': hit_rate
        })
    
    df_analysis = pd.DataFrame(analysis_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🎯 数字別予想・的中状況")
        # 予想回数の多い順にソート
        df_display = df_analysis.sort_values('予想回数', ascending=False)
        st.dataframe(
            df_display[['数字', '予想回数', '本数字的中', 'ボーナス的中', '的中率']],
            use_container_width=True,
            height=400
        )
    
    with col2:
        st.markdown("#### 📊 的中率ヒートマップ")
        # 数字を7×7のグリッドに配置（43個なので最後の6個は空）
        heatmap_data = np.zeros((7, 7))
        for i in range(43):
            row = i // 7
            col = i % 7
            heatmap_data[row, col] = df_analysis.iloc[i]['的中率_数値']
        
        # 数字ラベルを作成
        labels = []
        for i in range(7):
            row_labels = []
            for j in range(7):
                num = i * 7 + j + 1
                if num <= 43:
                    rate = df_analysis[df_analysis['数字'] == num]['的中率_数値'].iloc[0]
                    row_labels.append(f"{num}<br>{rate:.1%}")
                else:
                    row_labels.append("")
            labels.append(row_labels)
        
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=heatmap_data,
            text=labels,
            texttemplate="%{text}",
            textfont={"size": 10},
            colorscale="RdYlBu_r",
            hoverongaps=False
        ))
        fig_heatmap.update_layout(
            title="数字別的中率",
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False),
            height=400
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)

def _show_profit_analysis(performance_log: List[Dict]):
    """損益分析の表示"""
    st.markdown("### 💰 損益分析")
    
    df_profit = pd.DataFrame(performance_log)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 損益分布
        fig_hist = px.histogram(
            df_profit, x='profit', nbins=20,
            title='損益分布',
            labels={'profit': '損益(円)', 'count': '回数'}
        )
        fig_hist.add_vline(x=0, line_dash="dash", line_color="red", opacity=0.7)
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # 統計情報
        st.markdown("#### 📈 統計情報")
        stats = df_profit['profit'].describe()
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("平均損益", f"{stats['mean']:,.0f}円")
            st.metric("中央値", f"{stats['50%']:,.0f}円")
        with col_b:
            st.metric("最大利益", f"{stats['max']:,.0f}円")
            st.metric("最大損失", f"{stats['min']:,.0f}円")
    
    with col2:
        # 移動平均損益
        window = min(10, len(df_profit) // 4)
        if window > 1:
            df_profit['moving_avg'] = df_profit['profit'].rolling(window=window).mean()
            
            fig_ma = px.line(
                df_profit, x='draw_id', y=['profit', 'moving_avg'],
                title=f'損益と{window}回移動平均',
                labels={'draw_id': '回号', 'value': '損益(円)'}
            )
            fig_ma.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
            st.plotly_chart(fig_ma, use_container_width=True)
        
        # 連勝・連敗分析
        st.markdown("#### 🔥 連勝・連敗分析")
        streaks = []
        current_streak = 0
        current_type = None
        
        for profit in df_profit['profit']:
            if profit > 0:
                if current_type == 'win':
                    current_streak += 1
                else:
                    if current_streak > 0:
                        streaks.append((current_type, current_streak))
                    current_streak = 1
                    current_type = 'win'
            elif profit < 0:
                if current_type == 'loss':
                    current_streak += 1
                else:
                    if current_streak > 0:
                        streaks.append((current_type, current_streak))
                    current_streak = 1
                    current_type = 'loss'
            else:  # profit == 0
                if current_streak > 0:
                    streaks.append((current_type, current_streak))
                current_streak = 0
                current_type = None
        
        if current_streak > 0:
            streaks.append((current_type, current_streak))
        
        win_streaks = [s[1] for s in streaks if s[0] == 'win']
        loss_streaks = [s[1] for s in streaks if s[0] == 'loss']
        
        col_a, col_b = st.columns(2)
        with col_a:
            if win_streaks:
                st.metric("最大連勝", f"{max(win_streaks)}回")
                st.metric("平均連勝", f"{np.mean(win_streaks):.1f}回")
        with col_b:
            if loss_streaks:
                st.metric("最大連敗", f"{max(loss_streaks)}回")
                st.metric("平均連敗", f"{np.mean(loss_streaks):.1f}回")
