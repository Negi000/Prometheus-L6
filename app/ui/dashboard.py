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
    <h3 class="section-header">
        <svg class="dashboard-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/>
        </svg>
        基本統計情報
    </h3>
    ''', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("総抽選回数", len(df_history))
    
    with col2:
        if '本数字合計' in df_history.columns:
            try:
                # 型安全な平均計算
                sum_data = pd.to_numeric(df_history['本数字合計'], errors='coerce')
                avg_sum = sum_data.mean()
                if pd.notna(avg_sum):
                    st.metric("平均合計値", f"{avg_sum:.1f}")
                else:
                    st.metric("平均合計値", "N/A")
            except Exception as e:
                logger.warning(f"平均合計値計算エラー: {e}")
                st.metric("平均合計値", "N/A")
    
    with col3:
        if '第何回' in df_history.columns:
            try:
                # 型安全な最大回数取得
                round_data = pd.to_numeric(df_history['第何回'], errors='coerce')
                latest_round = round_data.max()
                if pd.notna(latest_round):
                    st.metric("最新回数", f"第{int(latest_round)}回")
                else:
                    st.metric("最新回数", "N/A")
            except Exception as e:
                logger.warning(f"最新回数取得エラー: {e}")
                st.metric("最新回数", "N/A")
    
    with col4:
        # 最新の抽選日
        if '抽選日' in df_history.columns:
            latest_date = df_history['抽選日'].max()
            st.metric("最新抽選日", latest_date)
    
    # 数字出現頻度分析
    st.markdown('''
    <h3 class="section-header">
        <svg class="dashboard-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/>
            <path d="m3 10 18 0"/>
            <path d="m8 21 0-18"/>
            <path d="m16 21 0-18"/>
        </svg>
        数字出現頻度分析
    </h3>
    ''', unsafe_allow_html=True)
    
    # 本数字の出現頻度を計算
    number_columns = [col for col in df_history.columns if col.startswith('本数字') and col != '本数字合計']
    
    if number_columns:
        all_numbers = []
        for col in number_columns:
            all_numbers.extend(df_history[col].dropna().tolist())
        
        # 出現回数の計算
        number_counts = pd.Series(all_numbers).value_counts().sort_index()
        
        # グラフ作成
        fig = px.bar(
            x=number_counts.index,
            y=number_counts.values,
            title="全数字の出現頻度",
            labels={'x': '数字', 'y': '出現回数'},
            color=number_counts.values,
            color_continuous_scale='Blues'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # ホット＆コールドナンバー分析
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('''
            <h4 class="section-header">
                <svg class="dashboard-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M8.5 14.5A2.5 2.5 0 0 0 11 12c0-1.38-.5-2-1-3-1.072-2.143-.224-4.054 2-6 .5 2.5 2 4.9 4 6.5 2 1.6 3 3.5 3 5.5a7 7 0 1 1-14 0c0-1.153.433-2.294 1-3a2.5 2.5 0 0 0 2.5 2.5z"/>
                </svg>
                ホットナンバー (最近よく出る数字)
            </h4>
            ''', unsafe_allow_html=True)
            
            # 最近30回のデータでホットナンバーを計算
            recent_data = df_history.tail(30)
            recent_numbers = []
            for col in number_columns:
                recent_numbers.extend(recent_data[col].dropna().tolist())
            
            hot_numbers = pd.Series(recent_numbers).value_counts().head(6)
            
            for i, (num, count) in enumerate(hot_numbers.items()):
                rank_emoji = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣", "6️⃣"][i]
                st.write(f"{rank_emoji} **{num}** - {count}回出現")
        
        with col2:
            st.markdown('''
            <h4 class="section-header">
                <svg class="dashboard-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/>
                    <path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/>
                </svg>
                コールドナンバー (長期間出現していない数字)
            </h4>
            ''', unsafe_allow_html=True)
            
            # 最後に出現してからの経過回数を計算
            cold_analysis = {}
            
            for num in range(1, 44):  # LOTO6は1-43の数字
                # この数字が最後に出現した回を探す
                last_appearance = None
                for idx, row in df_history.iterrows():
                    for col in number_columns:
                        if row[col] == num:
                            last_appearance = idx
                            break
                    if last_appearance is not None:
                        break
                
                if last_appearance is not None:
                    # 最後に出現してからの経過回数
                    games_since = len(df_history) - last_appearance - 1
                    cold_analysis[num] = games_since
                else:
                    # 一度も出現していない場合
                    cold_analysis[num] = len(df_history)
            
            # コールドナンバー上位6個
            cold_numbers = sorted(cold_analysis.items(), key=lambda x: x[1], reverse=True)[:6]
            
            for i, (num, games_since) in enumerate(cold_numbers):
                rank_emoji = ["❄️", "🧊", "⛄", "🌨️", "🌬️", "💙"][i]
                st.write(f"{rank_emoji} **{num}** - {games_since}回前に最後の出現")
    
    # 合計値の分布
    if '本数字合計' in df_history.columns:
        st.markdown('''
        <h3 class="section-header">
            <svg class="dashboard-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M3 3v18h18"/>
                <path d="m19 9-5 5-4-4-3 3"/>
            </svg>
            合計値分布分析
        </h3>
        ''', unsafe_allow_html=True)
        
        fig = px.histogram(
            df_history,
            x='本数字合計',
            nbins=30,
            title="本数字合計値の分布",
            labels={'本数字合計': '合計値', 'count': '頻度'},
            color_discrete_sequence=['#3b82f6']
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # 合計値の統計
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("平均値", f"{df_history['本数字合計'].mean():.1f}")
        with col2:
            st.metric("中央値", f"{df_history['本数字合計'].median():.1f}")
        with col3:
            st.metric("標準偏差", f"{df_history['本数字合計'].std():.1f}")
    
    # 特徴量データがある場合の追加分析
    if df_features is not None and len(df_features) > 0:
        st.markdown('''
        <h3 class="section-header">
            <svg class="dashboard-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <circle cx="12" cy="12" r="3"/>
                <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1 1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/>
            </svg>
            高度な特徴量分析
        </h3>
        ''', unsafe_allow_html=True)
        
        st.info("特徴量データを基にした高度な分析が利用可能です。戦略管理ページで詳細をご確認ください。")

def get_hot_numbers(df: pd.DataFrame, window: int = 30) -> pd.Series:
    """
    ホットナンバー（最近よく出る数字）を取得
    """
    if len(df) < window:
        window = len(df)
    
    recent_data = df.tail(window)
    number_columns = [col for col in df.columns if col.startswith('本数字') and col != '本数字合計']
    
    all_numbers = []
    for col in number_columns:
        all_numbers.extend(recent_data[col].dropna().tolist())
    
    return pd.Series(all_numbers).value_counts()

def get_cold_numbers(df: pd.DataFrame) -> dict:
    """
    コールドナンバー（長期間出現していない数字）を取得
    """
    number_columns = [col for col in df.columns if col.startswith('本数字') and col != '本数字合計']
    cold_analysis = {}
    
    for num in range(1, 44):  # LOTO6は1-43の数字
        last_appearance = None
        for idx, row in df.iterrows():
            for col in number_columns:
                if row[col] == num:
                    last_appearance = idx
                    break
            if last_appearance is not None:
                break
        
        if last_appearance is not None:
            games_since = len(df) - last_appearance - 1
            cold_analysis[num] = games_since
        else:
            cold_analysis[num] = len(df)
    
    return cold_analysis

def analyze_set_ball_performance(df: pd.DataFrame) -> dict:
    """
    セット球別の当選パターン分析
    """
    if 'セット球' not in df.columns:
        return {}
    
    set_ball_stats = {}
    
    for set_ball in df['セット球'].unique():
        if pd.isna(set_ball):
            continue
        
        subset = df[df['セット球'] == set_ball]
        
        set_ball_stats[set_ball] = {
            'count': len(subset),
            'avg_sum': subset['本数字合計'].mean() if '本数字合計' in subset.columns else 0,
            'most_common_numbers': get_hot_numbers(subset, len(subset)).head(6).to_dict()
        }
    
    return set_ball_stats
