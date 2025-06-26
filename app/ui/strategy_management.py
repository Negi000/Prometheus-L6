"""
戦略管理画面 (strategy_management.py)
学習済み戦略の管理とバックテストの実行
"""

import streamlit as st
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import os
import sys

from ..data_manager import DataManager
from ..backtester import Backtester
from ..feature_engine import FeatureEngine

logger = logging.getLogger(__name__)

def _show_backtest_details(data_manager: DataManager, strategy_name: str):
    """
    選択された戦略のバックテスト詳細を表示
    """
    try:
        # まず戦略名で検索を試行
        strategy_info = data_manager.get_strategy(strategy_name)
        
        if not strategy_info:
            # 戦略名で見つからない場合、戦略一覧から詳細情報を取得を試行
            st.warning(f"戦略「{strategy_name}」のデータベース情報が見つかりません。戦略一覧から情報を取得しています...")
            
            df_strategies = data_manager.get_all_strategies()
            strategy_row = df_strategies[df_strategies['strategy_name'] == strategy_name]
            
            if strategy_row.empty:
                st.error(f"戦略「{strategy_name}」が見つかりません。戦略が削除されているか、名前に誤りがある可能性があります。")
                return
            
            # DataFrameの行を辞書に変換
            strategy_info = strategy_row.iloc[0].to_dict()
        
        # バックテストログの取得
        backtest_log = strategy_info.get('backtest_log', [])
        
        # ログの型チェック
        if not isinstance(backtest_log, list):
            st.warning("バックテストログの形式が正しくありません。")
            backtest_log = []
        
        if not backtest_log:
            st.warning("この戦略にはバックテスト詳細データがありません。")
            
            # 戦略情報だけでも表示
            st.markdown("### 📊 戦略情報")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("モデルタイプ", strategy_info.get('model_type', 'Unknown'))
                st.metric("平均損益", f"{strategy_info.get('backtest_profit', 0):.0f}円")
            with col2:
                st.metric("作成日時", strategy_info.get('created_at', 'Unknown'))
                st.metric("4等以上的中率", f"{strategy_info.get('backtest_hit_rate_4', 0):.1%}")
            return
        
        # 詳細結果表示の新機能を使用
        from .detailed_results import show_detailed_backtest_results
        show_detailed_backtest_results(backtest_log, strategy_name)
        
    except Exception as e:
        st.error(f"バックテスト詳細の表示中にエラーが発生しました: {e}")
        logger.error(f"Backtest details error: {e}", exc_info=True)

def show_strategy_management(data_manager: DataManager):
    """
    戦略管理画面の表示
    
    Args:
        data_manager (DataManager): データ管理インスタンス
    """
    # SVG Icons CSS
    st.markdown("""
    <style>
        .strategy-icon { width: 20px; height: 20px; }
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
        <svg class="strategy-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M12 2L15.09 8.26L22 9L17 14L18.18 21L12 17.77L5.82 21L7 14L2 9L8.91 8.26L12 2Z"/>
        </svg>
        戦略管理
    </h2>
    ''', unsafe_allow_html=True)
    
    # タブ分割
    tab1, tab2, tab3 = st.tabs([
        "📋 戦略一覧", 
        "🚀 新規戦略学習",
        "📈 継続学習"
    ])
    
    with tab1:
        _show_strategy_list(data_manager)
    
    with tab2:
        _show_new_strategy_training(data_manager)
    
    with tab3:
        _show_continuous_learning(data_manager)

def _show_strategy_list(data_manager: DataManager):
    """
    戦略一覧の表示
    """
    st.markdown('''
    <h3 class="section-header">
        <svg class="strategy-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
            <polyline points="14,2 14,8 20,8"/>
            <line x1="16" y1="13" x2="8" y2="13"/>
            <line x1="16" y1="17" x2="8" y2="17"/>
            <polyline points="10,9 9,9 8,9"/>
        </svg>
        保存済み戦略一覧
    </h3>
    ''', unsafe_allow_html=True)
    
    try:
        df_strategies = data_manager.get_all_strategies()
        
        if len(df_strategies) == 0:
            st.info("まだ戦略が保存されていません。「新規戦略学習」タブから戦略を作成してください。")
            return
        
        # 戦略一覧テーブル
        display_df = df_strategies.copy()
        
        # 日付フォーマット
        if 'created_at' in display_df.columns:
            display_df['created_at'] = pd.to_datetime(display_df['created_at']).dt.strftime('%Y-%m-%d %H:%M')
        
        # 数値フォーマット
        if 'backtest_profit' in display_df.columns:
            display_df['backtest_profit'] = display_df['backtest_profit'].round(0).astype(int)
        
        for col in ['backtest_hit_rate_3', 'backtest_hit_rate_4', 'backtest_hit_rate_5']:
            if col in display_df.columns:
                display_df[col] = (display_df[col] * 100).round(1)
        
        # 列名を日本語に変更
        column_mapping = {
            'strategy_name': '戦略名',
            'model_type': 'モデル',
            'created_at': '作成日時',
            'backtest_profit': '平均損益(円)',
            'backtest_hit_rate_3': '3等以上率(%)',
            'backtest_hit_rate_4': '4等以上率(%)',
            'backtest_hit_rate_5': '5等以上率(%)',
            'description': '説明'
        }
        
        display_columns = ['strategy_name', 'model_type', 'created_at', 
                          'backtest_profit', 'backtest_hit_rate_4', 'backtest_hit_rate_3']
        
        display_df = display_df[display_columns].rename(columns=column_mapping)
        
        st.dataframe(display_df, use_container_width=True)
        
        # 戦略の選択と詳細表示
        st.markdown('''
        <h3 class="section-header">
            <svg class="strategy-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/>
                <line x1="9" y1="9" x2="9" y2="15"/>
                <line x1="15" y1="9" x2="15" y2="15"/>
            </svg>
            戦略詳細
        </h3>
        ''', unsafe_allow_html=True)
        
        strategy_names = df_strategies['strategy_name'].tolist()
        selected_strategy = st.selectbox("詳細を表示する戦略を選択", strategy_names)
        
        if selected_strategy:
            strategy_info = df_strategies[df_strategies['strategy_name'] == selected_strategy].iloc[0]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("平均損益", f"{strategy_info['backtest_profit']:.0f}円")
                st.metric("4等以上的中率", f"{strategy_info['backtest_hit_rate_4']:.1%}")
            
            with col2:
                st.metric("3等以上的中率", f"{strategy_info['backtest_hit_rate_3']:.1%}")
                st.metric("5等以上的中率", f"{strategy_info['backtest_hit_rate_5']:.1%}")
            
            if strategy_info['description']:
                st.text_area("説明", strategy_info['description'], disabled=True)
            
            # バックテスト詳細表示オプション
            st.markdown('''
            <h4 class="section-header">
                <svg class="strategy-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <circle cx="12" cy="12" r="10"/>
                    <line x1="12" y1="8" x2="12" y2="12"/>
                    <line x1="12" y1="16" x2="12.01" y2="16"/>
                </svg>
                バックテスト詳細
            </h4>
            ''', unsafe_allow_html=True)
            
            if st.button("🔍 バックテスト詳細を表示", key=f"show_details_{selected_strategy}"):
                with st.spinner("バックテスト詳細を読み込み中..."):
                    _show_backtest_details(data_manager, selected_strategy)
            
            # 戦略削除ボタン
            if st.button(f"⚠️ 戦略「{selected_strategy}」を削除", type="secondary"):
                if st.session_state.get('confirm_delete', False):
                    st.success(f"戦略「{selected_strategy}」を削除しました")
                    st.rerun()
                else:
                    st.session_state.confirm_delete = True
                    st.warning("再度クリックすると削除されます")
        
    except Exception as e:
        st.error(f"戦略一覧の取得中にエラーが発生しました: {e}")
        logger.error(f"Strategy list error: {e}")

def _show_new_strategy_training(data_manager: DataManager):
    """
    新規戦略学習の画面
    """
    st.markdown('''
    <h3 class="section-header">
        <svg class="strategy-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/>
        </svg>
        新規戦略学習
    </h3>
    ''', unsafe_allow_html=True)
    
    with st.form("strategy_training_form"):
        st.markdown("##### ⚙️ 学習パラメータ設定")
        
        # 基本設定
        col1, col2 = st.columns(2)
        
        with col1:
            strategy_name = st.text_input(
                "戦略名",
                value=f"Strategy_{datetime.now().strftime('%Y%m%d_%H%M')}",
                help="この戦略を識別するための名前"
            )
            
            model_type = st.selectbox(
                "🧠 モデルタイプ（プリセット）",
                ["ensemble_balanced", "ensemble_aggressive", "ensemble_full", "xgboost", "lightgbm", "random_forest"],
                index=0, # デフォルトをバランス型に設定
                help="""
                使用する機械学習モデルまたはその組み合わせを選択します。アンサンブル（ensemble）が推奨です。
                - **バランス型 (ensemble_balanced)**: 安定性の高いRandomForestとLightGBMを組み合わせます。
                - **アグレッシブ型 (ensemble_aggressive)**: パフォーマンス重視のLightGBMとXGBoostを組み合わせます。
                - **フルアンサンブル (ensemble_full)**: 3つの主要モデル全てを組み合わせ、多様性を最大化します。
                - **単体モデル**: 個別のモデルの性能を検証する場合に使用します。
                """
            )
            
            backtest_start = st.number_input(
                "バックテスト開始回",
                min_value=1,
                max_value=2008,
                value=1200,
                help="バックテスト開始の回号"
            )
        
        with col2:
            window_size = st.number_input(
                "学習ウィンドウサイズ",
                min_value=50,
                max_value=500,
                value=100,
                help="各予測時点での学習データ範囲"
            )
            
            purchase_count = st.number_input(
                "購入口数",
                min_value=1,
                max_value=100,
                value=20,
                help="1回あたりの購入口数"
            )
            
            backtest_end = st.number_input(
                "バックテスト終了回",
                min_value=1,
                max_value=2008,
                value=1600,
                help="バックテスト終了の回号"
            )

        # 自己補正フィードバック オプション
        st.markdown("---")
        st.markdown("##### 🧠 自己補正フィードバック（モデル育成機能）")
        enable_feedback = st.checkbox(
            "📈 予測誤差を学習し、モデルを継続的に補正する（強く推奨）",
            value=True,
            help="過去の予測の「間違い」をモデル自身に学習させ、次の予測精度を向上させることを目指します。計算負荷が少し増加しますが、モデルを育てる上で重要な機能です。"
        )
        if enable_feedback:
            st.info("✅ 自己補正を有効にすると、モデルが自身の予測のクセを捉え、より賢く成長することが期待できます。")
        
        # 詳細ログオプション
        detailed_log = st.checkbox(
            "📋 詳細ログを記録（推奨）",
            value=True,
            help="バックテストの詳細な実行ログを記録します。透明性向上のため推奨です。"
        )
        
        if detailed_log:
            st.info("✅ 詳細ログを有効にすると、各回の予測内容、実際の当選番号、損益詳細を確認できます。")
        
        # 説明
        description = st.text_area(
            "戦略説明",
            placeholder="この戦略の特徴や用途を記述してください（任意）",
            help="戦略の目的や特徴を記述"
        )
        
        # 学習実行ボタン
        submitted = st.form_submit_button("🚀 戦略学習を開始", type="primary")
        
        if submitted:
            # パラメータ検証
            if backtest_start >= backtest_end:
                st.error("バックテスト開始回は終了回より小さい値を設定してください。")
                return
            
            # 履歴データが学習ウィンドウサイズ以上あるか検証
            # バックテストの最初の回(backtest_start)の学習には、それ以前のwindow_size個のデータが必要
            if backtest_start <= window_size:
                st.error(f"バックテスト開始回({int(backtest_start)})は、学習ウィンドウサイズ({int(window_size)})より大きい必要があります。")
                return
            
            with st.spinner("戦略学習を実行中..."):
                try:
                    # データ準備
                    df_history = data_manager.load_loto6_history()
                    if df_history is None or len(df_history) == 0:
                        st.error("LOTO6履歴データが読み込まれていません。")
                        return
                    
                    st.info("特徴量を生成中...")
                    
                    # 特徴量エンジンで特徴量生成
                    feature_engine = FeatureEngine(df_history)
                    df_features = feature_engine.run_all()
                    
                    if df_features is None:
                        st.error("特徴量の生成に失敗しました。")
                        return
                    
                    st.info(f"バックテストを実行中... (第{backtest_start}回〜第{backtest_end}回)")
                    
                    # 進行状況表示用のプレースホルダー
                    progress_bar = st.progress(0)
                    progress_text = st.empty()
                    
                    def update_progress(progress, current_draw, total_draws):
                        progress_bar.progress(progress)
                        progress_text.text(f"バックテスト進行状況: 第{current_draw}回 ({progress:.1%} 完了)")
                    
                    # バックテスター初期化
                    backtester = Backtester(model_type=model_type)
                    
                    # バックテスト実行
                    result = backtester.run(
                        df_features, 
                        start_draw=int(backtest_start),
                        end_draw=int(backtest_end),
                        window_size=int(window_size),
                        purchase_count=int(purchase_count),
                        detailed_log=detailed_log,
                        enable_feedback=enable_feedback, # フィードバックオプションを渡す
                        progress_callback=update_progress
                    )
                    
                    # 返り値の数に応じて処理
                    if len(result) == 3:
                        model, performance_log, detailed_result = result
                    else:
                        model, performance_log = result
                        detailed_result = None
                    
                    if not performance_log:
                        st.error("バックテスト結果が空です。パラメータを確認してください。")
                        return
                    
                    # プログレスバーとテキストをクリア
                    progress_bar.empty()
                    progress_text.empty()
                    
                    # 結果の集計
                    total_profit = sum(log['profit'] for log in performance_log)
                    avg_profit = total_profit / len(performance_log)
                    
                    hit_rates = backtester._calculate_hit_rates(performance_log)
                    
                    # 戦略として保存
                    strategy_data = {
                        'strategy_name': strategy_name,
                        'model_type': model_type,
                        'model': model,
                        'created_at': datetime.now(),
                        'description': description,
                        'backtest_profit': avg_profit,
                        'backtest_hit_rate_3': hit_rates['hit_rate_3'],
                        'backtest_hit_rate_4': hit_rates['hit_rate_4'],
                        'backtest_hit_rate_5': hit_rates['hit_rate_5'],
                        'backtest_log': performance_log,  # 全ての結果を保存
                        'parameters': {
                            'window_size': window_size,
                            'purchase_count': purchase_count,
                            'backtest_start': backtest_start,
                            'backtest_end': backtest_end,
                            'detailed_log': detailed_log,
                            'enable_feedback': enable_feedback # パラメータも保存
                        }
                    }
                    
                    # データベースに保存
                    data_manager.save_strategy(strategy_data)
                    
                    # 結果表示
                    st.success(f"✅ 戦略「{strategy_name}」の学習が完了しました！")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("平均損益", f"{avg_profit:.0f}円")
                    with col2:
                        st.metric("4等以上的中率", f"{hit_rates['hit_rate_4']:.1%}")
                    with col3:
                        st.metric("3等以上的中率", f"{hit_rates['hit_rate_3']:.1%}")
                    
                    st.info(f"📊 バックテスト実行: {len(performance_log)}回")
                    
                    # 詳細分析結果の表示
                    if detailed_log and performance_log:
                        st.markdown("---")
                        
                        # 詳細結果表示の新機能を使用
                        from .detailed_results import show_detailed_backtest_results
                        show_detailed_backtest_results(performance_log, strategy_name)
                    
                    if detailed_log:
                        st.success("🔍 詳細ログが記録されました。戦略一覧タブから詳細を確認できます。")

                    # 詳細分析結果の表示
                    if detailed_result is not None and not detailed_result.empty:
                        st.markdown("---")
                        st.markdown("##### 🔬 バックテスト詳細分析")
                        
                        with st.expander("詳細な予測結果と特徴量を表示（先頭10件）"):
                            st.dataframe(detailed_result.head(10))
                        
                        # 特徴量の重要度を可視化
                        # 'feature_importance' 列が存在し、中身が空でない行があるか確認
                        if 'feature_importance' in detailed_result.columns and detailed_result['feature_importance'].notna().any():
                            st.markdown("##### ✨ 特徴量の重要度 (Feature Importance)")
                            
                            try:
                                all_importances = []
                                # dropna()でNaN（特徴量重要度がない行）をスキップ
                                for imp_list in detailed_result['feature_importance'].dropna():
                                    # imp_listがリスト形式であることを確認
                                    if isinstance(imp_list, list):
                                        all_importances.extend(imp_list)
                                
                                if all_importances:
                                    df_imp = pd.DataFrame(all_importances)
                                    # 特徴量ごとに重要度を平均化
                                    avg_imp = df_imp.groupby('feature')['importance'].mean().sort_values(ascending=False)
                                    
                                    st.info("バックテスト期間全体での特徴量の平均重要度（上位20）")
                                    st.bar_chart(avg_imp.head(20))
                                    
                                    with st.expander("全ての特徴量の重要度を表示"):
                                        st.dataframe(avg_imp)
                                else:
                                    st.warning("特徴量の重要度データが見つかりませんでした。")

                            except Exception as ex:
                                st.error(f"特徴量の重要度の表示中にエラーが発生しました: {ex}")
                    
                except Exception as e:
                    st.error(f"戦略学習中にエラーが発生しました: {e}")
                    logger.error(f"Strategy training error: {e}")
                    
                    # デバッグ情報
                    if st.checkbox("デバッグ情報を表示"):
                        st.code(f"エラー詳細: {str(e)}")

def _show_continuous_learning(data_manager: DataManager):
    """
    継続学習の画面（既存戦略の追加学習）
    """
    st.markdown('''
    <h3 class="section-header">
        <svg class="strategy-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
            <polyline points="17,8 12,3 7,8"/>
            <line x1="12" y1="3" x2="12" y2="15"/>
        </svg>
        継続学習（既存戦略の改善）
    </h3>
    ''', unsafe_allow_html=True)
    
    try:
        df_strategies = data_manager.get_all_strategies()
        
        if len(df_strategies) == 0:
            st.info("継続学習を行うには、まず「新規戦略学習」で戦略を作成してください。")
            return
        
        st.info("💡 継続学習では、既存の戦略モデルを最新のデータで追加学習し、性能を向上させることができます。")
        st.warning("🧠 **真の継続学習**: 戦略を上書き更新することで、同じ戦略を何度でも成長させることができます。")
        
        with st.form("continuous_learning_form"):
            st.markdown("##### 🎯 継続学習設定")
            
            # 継続学習モード選択
            learning_mode = st.radio(
                "📈 継続学習モード",
                ["🔄 戦略を上書き更新（推奨：真の継続学習）", "🆕 新しい戦略として保存"],
                help="上書き更新: 同じ戦略を継続的に成長させます / 新戦略保存: バージョン管理のため別名で保存"
            )
            
            # 戦略選択
            strategy_names = df_strategies['strategy_name'].tolist()
            selected_strategy = st.selectbox(
                "📊 継続学習する戦略を選択",
                strategy_names,
                help="既存の戦略から選択してください。選択した戦略のモデルを基に追加学習を行います。"
            )
            
            if selected_strategy:
                strategy_info = df_strategies[df_strategies['strategy_name'] == selected_strategy].iloc[0]
                
                # 現在の戦略情報を表示
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("現在の平均損益", f"{strategy_info['backtest_profit']:.0f}円")
                with col2:
                    st.metric("現在の4等以上的中率", f"{strategy_info['backtest_hit_rate_4']:.1%}")
                with col3:
                    st.metric("現在の3等以上的中率", f"{strategy_info['backtest_hit_rate_3']:.1%}")
                
                # 学習パラメータ設定
                st.markdown("---")
                st.markdown("##### ⚙️ 追加学習パラメータ")
                
                # 元の戦略のパラメータを取得
                original_params = strategy_info.get('parameters', {})
                
                # 戦略の学習範囲情報を横並びで表示
                original_end = original_params.get('backtest_end', 1600)
                original_start = original_params.get('backtest_start', 1000)
                
                info_col1, info_col2 = st.columns(2)
                with info_col1:
                    st.info(f"💡 **元の戦略**: 第{original_start}回〜第{original_end}回で学習済み")
                with info_col2:
                    st.success(f"🔄 **継続学習**: 任意の範囲で追加パターン学習が可能です")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    continue_start = st.number_input(
                        "継続学習開始回",
                        min_value=1000,
                        max_value=2008,
                        value=max(1800, original_end - 100),
                        help="任意の範囲を選択可能。過去データでの再学習も含めてパターン強化を行います。"
                    )
                    
                    window_size = st.number_input(
                        "学習ウィンドウサイズ",
                        min_value=50,
                        max_value=500,
                        value=original_params.get('window_size', 100),
                        help="各予測時点での学習データ範囲"
                    )
                
                with col2:
                    # 継続学習終了回を動的に設定
                    min_end = continue_start + 10
                    default_end = min(continue_start + 100, 2008)
                    
                    continue_end = st.number_input(
                        "継続学習終了回",
                        min_value=min_end,
                        max_value=2008,
                        value=default_end,
                        help="継続学習の終了回号"
                    )
                    
                    purchase_count = st.number_input(
                        "購入口数",
                        min_value=1,
                        max_value=100,
                        value=original_params.get('purchase_count', 20),
                        help="1回あたりの購入口数"
                    )
                
                # 継続学習オプション
                st.markdown("---")
                st.markdown("##### 🧠 継続学習オプション")
                
                enable_feedback = st.checkbox(
                    "📈 自己補正フィードバックを継続",
                    value=original_params.get('enable_feedback', True),
                    help="元の戦略の自己補正機能を継続します。"
                )
                
                detailed_log = st.checkbox(
                    "📋 詳細ログを記録",
                    value=True,
                    help="継続学習の詳細ログを記録します。"
                )
                
                # 学習率調整係数の詳細説明
                st.markdown("**🎯 学習率調整係数** - 継続学習の慎重さを制御")
                
                # 推奨値の説明を表示
                with st.expander("💡 学習率調整係数について詳しく"):
                    st.markdown("""
                    **学習率調整係数**は、継続学習時にモデルがどれだけ積極的に新しいパターンを学習するかを制御します。
                    
                    **📈 推奨値とその効果:**
                    - **0.5-0.7 (慎重)**: 既存の知識を重視し、新しいデータによる変化を抑制
                    - **0.8 (推奨)**: バランスの取れた学習。安定性と適応性の良いバランス
                    - **0.9-1.0 (標準)**: 通常の学習率。新しいパターンを積極的に学習
                    - **1.1-1.5 (積極的)**: より敏感に新しいトレンドに適応。不安定になるリスクあり
                    - **1.6-2.0 (非推奨)**: 過学習のリスクが高い。既存の知識を破壊する可能性
                    
                    **💭 選択の指針:**
                    - 既存戦略が安定して良好 → **0.6-0.8** (慎重な改善)
                    - 性能改善を期待 → **0.8-1.0** (バランス型)
                    - 最近のトレンド変化に対応 → **1.0-1.2** (適応型)
                    """)
                
                learning_rate_factor = st.slider(
                    "学習率調整係数",
                    min_value=0.1,
                    max_value=2.0,
                    value=0.8,
                    step=0.1,
                    help="0.8推奨: 慎重で安定した継続学習。1.0=標準、0.6=超慎重、1.2=積極的"
                )
                
                # 選択された値に応じた説明を表示
                if learning_rate_factor < 0.7:
                    st.info("🛡️ **慎重モード**: 既存の知識を重視し、安定性を優先します")
                elif learning_rate_factor <= 1.0:
                    st.success("⚖️ **バランスモード**: 安定性と適応性の良いバランスです")
                elif learning_rate_factor <= 1.3:
                    st.warning("🚀 **積極モード**: 新しいトレンドに敏感に反応します")
                else:
                    st.error("⚠️ **危険モード**: 過学習のリスクがあります。注意して使用してください")
                
                # 戦略名の設定（モードに応じて）
                update_mode = learning_mode.startswith("🔄")
                
                if update_mode:
                    st.info(f"✅ 戦略「{selected_strategy}」を上書き更新します（真の継続学習）")
                    new_strategy_name = selected_strategy  # 同じ名前で上書き
                    strategy_name_input = st.empty()  # 名前入力フィールドを非表示
                else:
                    new_strategy_name = st.text_input(
                        "継続学習後の戦略名",
                        value=f"{selected_strategy}_v2",
                        help="継続学習後の戦略に付ける新しい名前"
                    )
                
                description = st.text_area(
                    "継続学習の説明",
                    placeholder=f"「{selected_strategy}」の継続学習。第{continue_start}回〜第{continue_end}回のデータで追加学習。",
                    help="継続学習の内容や改善点を記述"
                )
                
                # 継続学習実行ボタン
                if update_mode:
                    submitted = st.form_submit_button("🔄 戦略を継続成長させる", type="primary")
                else:
                    submitted = st.form_submit_button("🆕 新しい戦略として学習", type="primary")
                
                if submitted:
                    # 元の戦略の終了回を取得
                    original_end = original_params.get('backtest_end', 1600)
                    
                    # シンプルなパラメータ検証
                    validation_errors = []
                    
                    # 基本的な順序チェック
                    if continue_start >= continue_end:
                        validation_errors.append("継続学習開始回は終了回より小さい値を設定してください。")
                    
                    # 学習ウィンドウサイズのチェック
                    if continue_start <= window_size:
                        validation_errors.append(f"継続学習開始回({int(continue_start)})は、学習ウィンドウサイズ({int(window_size)})より大きい必要があります。")
                    
                    # 戦略名の重複チェック（新戦略保存の場合のみ）
                    if not update_mode and new_strategy_name in strategy_names:
                        validation_errors.append(f"戦略名「{new_strategy_name}」は既に存在します。別の名前を選択してください。")
                    
                    # エラーがある場合は表示して終了
                    if validation_errors:
                        for error in validation_errors:
                            st.error(error)
                        return
                    
                    with st.spinner("継続学習を実行中..."):
                        try:
                            # データ準備
                            df_history = data_manager.load_loto6_history()
                            if df_history is None or len(df_history) == 0:
                                st.error("LOTO6履歴データが読み込まれていません。")
                                return
                            
                            st.info("特徴量を生成中...")
                            
                            # 特徴量エンジンで特徴量生成
                            feature_engine = FeatureEngine(df_history)
                            df_features = feature_engine.run_all()
                            
                            if df_features is None:
                                st.error("特徴量の生成に失敗しました。")
                                return
                            
                            st.info(f"継続学習を実行中... (第{continue_start}回〜第{continue_end}回)")
                            
                            # 元の戦略からモデルを安全に取得
                            if 'model' not in strategy_info or strategy_info['model'] is None:
                                st.error("選択された戦略にモデルデータが含まれていません。この戦略では継続学習を実行できません。")
                                return
                            
                            base_model = strategy_info['model']
                            
                            # 進行状況表示用のプレースホルダー
                            progress_bar = st.progress(0)
                            progress_text = st.empty()
                            
                            def update_progress(progress, current_draw, total_draws):
                                progress_bar.progress(progress)
                                progress_text.text(f"継続学習進行状況: 第{current_draw}回 ({progress:.1%} 完了)")
                            
                            model_type = strategy_info.get('model_type', 'xgboost')
                            
                            logger.info(f"継続学習: 元の戦略={selected_strategy}, モデルタイプ={model_type}")
                            
                            # バックテスター初期化（継続学習モード）
                            backtester = Backtester(model_type=model_type)
                            
                            # 継続学習実行
                            try:
                                result = backtester.run_continuous_learning(
                                    df_features,
                                    base_model=base_model,
                                    start_draw=int(continue_start),
                                    end_draw=int(continue_end),
                                    window_size=int(window_size),
                                    purchase_count=int(purchase_count),
                                    detailed_log=detailed_log,
                                    enable_feedback=enable_feedback,
                                    learning_rate_factor=learning_rate_factor,
                                    progress_callback=update_progress
                                )
                            except Exception as e:
                                st.error(f"継続学習の実行中にエラーが発生しました: {e}")
                                logger.error(f"Continuous learning execution error: {e}")
                                return
                            
                            # 返り値の数に応じて処理
                            if len(result) == 3:
                                updated_model, performance_log, detailed_result = result
                            else:
                                updated_model, performance_log = result
                                detailed_result = None
                            

                            if not performance_log:
                                st.error("継続学習の結果が空です。パラメータを確認してください。")
                                return
                            
                            # プログレスバーとテキストをクリア
                            progress_bar.empty()
                            progress_text.empty()
                            
                            # 結果の集計（型安全なキーアクセス）
                            def safe_get_profit(log):
                                if isinstance(log, dict):
                                    return log.get('profit', 0)
                                elif isinstance(log, (list, tuple)) and len(log) > 1:
                                    return log[1] if len(log) > 1 else 0
                                return 0
                            
                            def has_profit(log):
                                if isinstance(log, dict):
                                    return 'profit' in log
                                elif isinstance(log, (list, tuple)):
                                    return len(log) > 1
                                return False
                            
                            total_profit = sum(safe_get_profit(log) for log in performance_log if has_profit(log))
                            valid_logs = [log for log in performance_log if has_profit(log)]
                            
                            if not valid_logs:
                                st.error("有効な継続学習結果が見つかりません。ログエントリにprofitデータが存在しません。")
                                return
                            
                            avg_profit = total_profit / len(valid_logs)
                            
                            hit_rates = backtester._calculate_hit_rates(performance_log)
                            
                            # 改善度を計算
                            profit_improvement = avg_profit - strategy_info['backtest_profit']
                            hit_rate_improvement = hit_rates['hit_rate_4'] - strategy_info['backtest_hit_rate_4']
                            
                            # 継続学習後の戦略として保存
                            strategy_data = {
                                'strategy_name': new_strategy_name,
                                'model_type': model_type,
                                'model': updated_model,
                                'created_at': datetime.now(),
                                'description': description,
                                'backtest_profit': avg_profit,
                                'backtest_hit_rate_3': hit_rates['hit_rate_3'],
                                'backtest_hit_rate_4': hit_rates['hit_rate_4'],
                                'backtest_hit_rate_5': hit_rates['hit_rate_5'],
                                'backtest_log': performance_log,  # 全ての結果を保存
                                'parameters': {
                                    'window_size': window_size,
                                    'purchase_count': purchase_count,
                                    'backtest_start': continue_start,
                                    'backtest_end': continue_end,
                                    'detailed_log': detailed_log,
                                    'enable_feedback': enable_feedback,
                                    'learning_rate_factor': learning_rate_factor,
                                    'base_strategy': selected_strategy,  # 元の戦略名を記録
                                    'is_continuous_learning': True
                                }
                            }
                            
                            # データベースに保存（モードに応じて）
                            if update_mode:
                                # 戦略を上書き更新（真の継続学習）
                                data_manager.update_strategy(new_strategy_name, strategy_data)
                                st.success(f"🔄 戦略「{new_strategy_name}」を継続成長させました！")
                                st.info("💡 この戦略は今後も継続的に学習を重ねることができます。")
                            else:
                                # 新しい戦略として保存
                                data_manager.save_strategy(strategy_data)
                                st.success(f"🆕 継続学習が完了しました！戦略「{new_strategy_name}」として保存されました。")
                            
                            # 改善度を表示
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                improvement_color = "green" if profit_improvement > 0 else "red"
                                st.metric(
                                    "平均損益", 
                                    f"{avg_profit:.0f}円",
                                    delta=f"{profit_improvement:+.0f}円"
                                )
                            with col2:
                                st.metric(
                                    "4等以上的中率", 
                                    f"{hit_rates['hit_rate_4']:.1%}",
                                    delta=f"{hit_rate_improvement:+.1%}"
                                )
                            with col3:
                                st.metric("3等以上的中率", f"{hit_rates['hit_rate_3']:.1%}")
                            

                            st.info(f"📊 継続学習実行: {len(performance_log)}回")
                            
                            if profit_improvement > 0:
                                st.success(f"🎉 戦略が改善されました！平均損益が {profit_improvement:.0f}円 向上しました。")
                            elif profit_improvement < -100:
                                st.warning(f"⚠️ 今回の継続学習では性能が低下しました。パラメータを調整して再試行することをお勧めします。")
                            else:
                                st.info("📊 性能は横ばいです。さらなる改善のため、異なるパラメータでの継続学習を検討してください。")
                            

                            if detailed_log:
                                st.success("🔍 詳細ログが記録されました。戦略一覧タブから詳細を確認できます。")
                                
                                # 継続学習の詳細結果も表示
                                st.markdown("---")
                                from .detailed_results import show_detailed_backtest_results
                                show_detailed_backtest_results(performance_log, new_strategy_name)
                            
                        except Exception as e:
                            st.error(f"継続学習中にエラーが発生しました: {e}")
                            logger.error(f"Continuous learning error: {e}")
                            
                            # デバッグ情報
                            if st.checkbox("デバッグ情報を表示", key="debug_continuous"):
                                st.code(f"エラー詳細: {str(e)}")
    
    except Exception as e:
        st.error(f"継続学習画面の表示中にエラーが発生しました: {e}")
        logger.error(f"Continuous learning page error: {e}")