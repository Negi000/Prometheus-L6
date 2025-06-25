"""
ダッシュボード画面 (dashboard.py)
LOTO6データの基本分析とビジュアライゼーション
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging

logger = logging.getLogger(__name__)

def show_dashboard(df_history: pd.DataFrame, df_features: pd.DataFrame = None):
    """
    ダッシュボード画面の表示
    
    Args:
        df_history (pd.DataFrame): LOTO6履歴データ
        df_features (pd.DataFrame): 特徴量データ（オプション）
    """
    # SVG Icons CSS
    st.markdown("""
    <style>
        .dashboard-icon { width: 20px; height: 20px; }
        .metric-icon { width: 16px; height: 16px; margin-right: 8px; }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('''
    <h2 style="display: flex; align-items: center; gap: 12px; margin-bottom: 20px;">
        <svg class="dashboard-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/>
            <line x1="9" y1="9" x2="9" y2="15"/>
            <line x1="15" y1="9" x2="15" y2="15"/>
        </svg>
        LOTO6 データ分析ダッシュボード
    </h2>
    ''', unsafe_allow_html=True)
    
    if df_history is None or len(df_history) == 0:
        st.error("LOTO6履歴データが読み込まれていません。")
        return
    
    # 基本統計情報
    st.markdown('''
    <h3 style="display: flex; align-items: center; gap: 8px; margin-bottom: 16px;">
        <svg class="dashboard-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/>
        </svg>
        基本統計情報
    </h3>
    ''', unsafe_allow_html=True)"■ LOTO6 データ分析ダッシュボード")
    
    if df_history is None or len(df_history) == 0:
        st.error("LOTO6履歴データが読み込まれていません。")
        return
    
    # 基本統計情報
    st.subheader("▶ 基本統計情報")LOTO6データの基本分析とビジュアライゼーション
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging

logger = logging.getLogger(__name__)

def show_dashboard(df_history: pd.DataFrame, df_features: pd.DataFrame = None):
    """
    ダッシュボード画面の表示
    
    Args:
        df_history (pd.DataFrame): LOTO6履歴データ
        df_features (pd.DataFrame): 特徴量データ（オプション）
    """
    st.header("� LOTO6 データ分析ダッシュボード")
    
    if df_history is None or len(df_history) == 0:
        st.error("LOTO6履歴データが読み込まれていません。")
        return
    
    # 基本統計情報
    st.subheader("⚡ 基本統計情報")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("総抽選回数", len(df_history))
    
    with col2:
        if '本数字合計' in df_history.columns:
            avg_sum = df_history['本数字合計'].mean()
            st.metric("平均合計値", f"{avg_sum:.1f}")
    
    with col3:
        if '第何回' in df_history.columns:
            latest_draw = df_history['第何回'].max()
            st.metric("最新回号", f"第{latest_draw}回")
    
    with col4:
        if 'セット球' in df_history.columns:
            set_ball_count = df_history['セット球'].nunique()
            st.metric("セット球種類", f"{set_ball_count}種類")
    
    # 数字出現分析
    st.subheader("● 数字出現分析")
    
    # 本数字の出現回数を計算
    main_cols = [col for col in df_history.columns if '本数字' in col and col != '本数字合計']
    
    if main_cols:
        appearance_data = _calculate_number_appearances(df_history, main_cols)
        
        # 出現回数の棒グラフ
        fig_appearance = px.bar(
            x=list(range(1, 44)),
            y=[appearance_data.get(i, 0) for i in range(1, 44)],
            title="各数字の出現回数",
            labels={'x': '数字', 'y': '出現回数'}
        )
        fig_appearance.update_layout(height=400)
        st.plotly_chart(fig_appearance, use_container_width=True)
        
        # ホット・コールドナンバー分析
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("▲ ホットナンバー（直近20回）")
            hot_numbers = _get_hot_numbers(df_history, main_cols, recent_draws=20)
            if hot_numbers:
                for i, (num, count) in enumerate(hot_numbers[:10], 1):
                    # シンプルで見やすい表示
                    if i <= 3:
                        st.markdown(f"● **{num}** - {count}回出現")
                    elif i <= 6:
                        st.markdown(f"○ **{num}** - {count}回出現")
                    else:
                        st.markdown(f"• **{num}** - {count}回出現")
            else:
                st.write("データ不足")
        
        with col2:
            st.subheader("▼ コールドナンバー（50回以上未出現）")
            cold_numbers = _get_cold_numbers(df_history, main_cols, threshold=50)
            if cold_numbers:
                for i, (num, gap) in enumerate(cold_numbers[:10], 1):
                    st.markdown(f"◦ **{num}** - {gap}回未出現")
            else:
                st.write("該当なし")
    
    # 合計値分析
    if '本数字合計' in df_history.columns:
        st.subheader("▲ 本数字合計値の分布")
        
        fig_sum_dist = px.histogram(
            df_history,
            x='本数字合計',
            nbins=30,
            title="本数字合計値のヒストグラム",
            labels={'本数字合計': '合計値', 'count': '頻度'}
        )
        
        # 統計情報をオーバーレイ
        mean_sum = df_history['本数字合計'].mean()
        median_sum = df_history['本数字合計'].median()
        
        fig_sum_dist.add_vline(x=mean_sum, line_dash="dash", line_color="red", 
                              annotation_text=f"平均: {mean_sum:.1f}")
        fig_sum_dist.add_vline(x=median_sum, line_dash="dash", line_color="blue", 
                              annotation_text=f"中央値: {median_sum:.1f}")
        
        st.plotly_chart(fig_sum_dist, use_container_width=True)
    
    # 奇偶比率分析
    if main_cols:
        st.subheader("◆ 奇偶比率分析")
        
        odd_even_data = _calculate_odd_even_ratio(df_history, main_cols)
        
        fig_odd_even = px.pie(
            values=list(odd_even_data.values()),
            names=list(odd_even_data.keys()),
            title="奇偶比率の分布"
        )
        st.plotly_chart(fig_odd_even, use_container_width=True)
    
    # セット球分析
    if 'セット球' in df_history.columns:
        st.subheader("○ セット球分析")
        
        set_ball_stats = _analyze_set_balls(df_history)
        
        if set_ball_stats:
            df_set_ball = pd.DataFrame(set_ball_stats).T
            df_set_ball.index.name = 'セット球'
            st.dataframe(df_set_ball)
    
    # 時系列トレンド（特徴量データがある場合）
    if df_features is not None and '第何回' in df_features.columns:
        st.subheader("▲ 時系列トレンド分析")
        
        # 最近の傾向を可視化
        recent_data = df_features.tail(100)  # 直近100回
        
        if '本数字合計' in recent_data.columns:
            fig_trend = px.line(
                recent_data,
                x='第何回',
                y='本数字合計',
                title="直近100回の合計値トレンド"
            )
            st.plotly_chart(fig_trend, use_container_width=True)

def _calculate_number_appearances(df: pd.DataFrame, main_cols: list) -> dict:
    """
    各数字の出現回数を計算
    """
    appearance_count = {}
    
    for num in range(1, 44):
        count = 0
        for col in main_cols:
            count += (df[col] == num).sum()
        appearance_count[num] = count
    
    return appearance_count

def _get_hot_numbers(df: pd.DataFrame, main_cols: list, recent_draws: int = 20) -> list:
    """
    ホットナンバー（最近よく出る数字）を取得
    """
    recent_data = df.tail(recent_draws)
    hot_count = {}
    
    for num in range(1, 44):
        count = 0
        for col in main_cols:
            count += (recent_data[col] == num).sum()
        if count > 0:
            hot_count[num] = count
    
    # 出現回数で降順ソート
    return sorted(hot_count.items(), key=lambda x: x[1], reverse=True)

def _get_cold_numbers(df: pd.DataFrame, main_cols: list, threshold: int = 50) -> list:
    """
    コールドナンバー（長期間出現していない数字）を取得
    """
    cold_numbers = []
    
    for num in range(1, 44):
        # 最後に出現した位置を探す
        last_appearance = -1
        for i in range(len(df) - 1, -1, -1):
            row = df.iloc[i]
            if any(row[col] == num for col in main_cols):
                last_appearance = i
                break
        
        if last_appearance == -1:
            gap = len(df)  # 一度も出現していない
        else:
            gap = len(df) - 1 - last_appearance
        
        if gap >= threshold:
            cold_numbers.append((num, gap))
    
    # ギャップで降順ソート
    return sorted(cold_numbers, key=lambda x: x[1], reverse=True)

def _calculate_odd_even_ratio(df: pd.DataFrame, main_cols: list) -> dict:
    """
    奇偶比率を計算
    """
    ratio_count = {
        '6:0 (全偶数)': 0,
        '5:1': 0,
        '4:2': 0,
        '3:3': 0,
        '2:4': 0,
        '1:5': 0,
        '0:6 (全奇数)': 0
    }
    
    for _, row in df.iterrows():
        numbers = [row[col] for col in main_cols if not pd.isna(row[col])]
        odd_count = sum(1 for num in numbers if num % 2 == 1)
        even_count = 6 - odd_count
        
        ratio_key = f"{odd_count}:{even_count}"
        if odd_count == 0:
            ratio_key = '0:6 (全奇数)'
        elif odd_count == 6:
            ratio_key = '6:0 (全偶数)'
        
        if ratio_key in ratio_count:
            ratio_count[ratio_key] += 1
    
    return ratio_count

def _analyze_set_balls(df: pd.DataFrame) -> dict:
    """
    セット球の分析
    """
    if 'セット球' not in df.columns:
        return {}
    
    set_ball_stats = {}
    
    for set_ball in df['セット球'].unique():
        if pd.isna(set_ball):
            continue
            
        subset = df[df['セット球'] == set_ball]
        
        stats = {
            '使用回数': len(subset),
            '平均合計値': subset['本数字合計'].mean() if '本数字合計' in subset.columns else 0,
            '合計値標準偏差': subset['本数字合計'].std() if '本数字合計' in subset.columns else 0
        }
        
        set_ball_stats[set_ball] = stats
    
    return set_ball_stats
