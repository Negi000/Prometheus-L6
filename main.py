"""
Prometheus-L6 メインアプリケーション
LOTO6戦略的投資支援システム
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import logging
from configparser import ConfigParser

# アプリケーションモジュールのインポート
from app.data_manager import DataManager
from app.feature_engine import FeatureEngine
from app.ui.dashboard import show_dashboard
import app.ui.strategy_management
from app.ui.prediction import show_prediction
from app.ui.simulation import show_simulation

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Streamlit設定
st.set_page_config(
    page_title="Prometheus-L6",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# カスタムCSS with Modern SVG Icons
st.markdown("""
<style>
    /* Base Styles */
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1e3a8a;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 12px;
    }
    .subtitle {
        font-size: 1.2rem;
        text-align: center;
        color: #64748b;
        margin-bottom: 2rem;
    }
    .sidebar-content {
        padding: 1rem;
    }
    .metric-container {
        background-color: #f8fafc;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #3b82f6;
    }
    
    /* SVG Icon Styles */
    .icon-svg {
        width: 20px;
        height: 20px;
        display: inline-block;
        vertical-align: middle;
    }
    .icon-header {
        width: 32px;
        height: 32px;
    }
    .icon-nav {
        width: 18px;
        height: 18px;
        margin-right: 8px;
    }
    
    /* Navigation Styles */
    .nav-item {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 8px 12px;
        border-radius: 6px;
        margin: 2px 0;
        transition: all 0.2s ease;
        cursor: pointer;
    }
    .nav-item:hover {
        background-color: #f3f4f6;
        transform: translateX(2px);
    }
    
    /* Status Indicators */
    .status-success { color: #10b981; }
    .status-warning { color: #f59e0b; }
    .status-error { color: #ef4444; }
    .status-info { color: #3b82f6; }
    
    /* Card Styles */
    .info-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

def main():
    """
    メインアプリケーション
    """
    # ヘッダー
    st.markdown('''
    <div class="main-header">
        <svg class="icon-header" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/>
        </svg>
        Prometheus-L6
    </div>
    ''', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">LOTO6戦略的投資支援システム - AI駆動型分析・意思決定支援ツール</p>', unsafe_allow_html=True)
    
    # サイドバー
    with st.sidebar:
        st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
        
        # ナビゲーションメニュー
        st.markdown('''
        <div style="margin-bottom: 20px;">
            <h3 style="display: flex; align-items: center; gap: 8px; margin-bottom: 16px;">
                <svg class="icon-svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <line x1="3" y1="6" x2="21" y2="6"/>
                    <line x1="3" y1="12" x2="21" y2="12"/>
                    <line x1="3" y1="18" x2="21" y2="18"/>
                </svg>
                メニュー
            </h3>
        </div>
        ''', unsafe_allow_html=True)
        
        # カスタムラジオボタンの代替
        menu_options = [
            ("dashboard", "ダッシュボード", '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><line x1="9" y1="9" x2="9" y2="15"/><line x1="15" y1="9" x2="15" y2="15"/></svg>'),
            ("strategy", "戦略管理", '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2L15.09 8.26L22 9L17 14L18.18 21L12 17.77L5.82 21L7 14L2 9L8.91 8.26L12 2Z"/></svg>'),
            ("prediction", "予測生成", '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="m16 12-4-4-4 4"/><path d="m12 16 4-4-4-4"/></svg>'),
            ("simulation", "シミュレーション", '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 3v18h18"/><path d="m19 9-5 5-4-4-3 3"/></svg>'),
            ("system", "システム情報", '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1 1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>')
        ]
        
        page = st.radio(
            "機能を選択してください",
            [f"{name}" for _, name, _ in menu_options],
            index=0,
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # システム状態
        st.markdown('''
        <h3 style="display: flex; align-items: center; gap: 8px; margin: 20px 0 16px 0;">
            <svg class="icon-svg status-info" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M9 12l2 2 4-4"/>
                <circle cx="12" cy="12" r="9"/>
            </svg>
            システム状態
        </h3>
        ''', unsafe_allow_html=True)
        
        try:
            # データマネージャーの初期化テスト
            data_manager = DataManager()
            st.markdown('''
            <div style="display: flex; align-items: center; gap: 8px; color: #10b981; margin: 8px 0;">
                <svg class="icon-svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M9 12l2 2 4-4"/>
                    <circle cx="12" cy="12" r="9"/>
                </svg>
                <span>システム正常</span>
            </div>
            ''', unsafe_allow_html=True)
            
            # データファイルの存在確認
            try:
                df_history = data_manager.load_loto6_history()
                st.markdown(f'''
                <div style="display: flex; align-items: center; gap: 8px; color: #3b82f6; margin: 8px 0;">
                    <svg class="icon-svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                        <polyline points="14,2 14,8 20,8"/>
                        <line x1="16" y1="13" x2="8" y2="13"/>
                        <line x1="16" y1="17" x2="8" y2="17"/>
                        <polyline points="10,9 9,9 8,9"/>
                    </svg>
                    <span>履歴データ: {len(df_history)}件</span>
                </div>
                ''', unsafe_allow_html=True)
                data_status = "success"
            except UnicodeDecodeError as e:
                st.markdown(f'''
                <div style="display: flex; align-items: center; gap: 8px; color: #ef4444; margin: 8px 0;">
                    <svg class="icon-svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <circle cx="12" cy="12" r="10"/>
                        <line x1="15" y1="9" x2="9" y2="15"/>
                        <line x1="9" y1="9" x2="15" y2="15"/>
                    </svg>
                    <span>文字エンコーディングエラー</span>
                </div>
                ''', unsafe_allow_html=True)
                st.code(f"エラー詳細: {str(e)}")
                data_status = "encoding_error"
            except FileNotFoundError:
                st.markdown('''
                <div style="display: flex; align-items: center; gap: 8px; color: #ef4444; margin: 8px 0;">
                    <svg class="icon-svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
                        <line x1="12" y1="9" x2="12" y2="13"/>
                        <line x1="12" y1="17" x2="12.01" y2="17"/>
                    </svg>
                    <span>データファイルが見つかりません</span>
                </div>
                ''', unsafe_allow_html=True)
                data_status = "file_not_found"
            except Exception as e:
                st.markdown(f'''
                <div style="display: flex; align-items: center; gap: 8px; color: #f59e0b; margin: 8px 0;">
                    <svg class="icon-svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
                        <line x1="12" y1="9" x2="12" y2="13"/>
                        <line x1="12" y1="17" x2="12.01" y2="17"/>
                    </svg>
                    <span>データ読み込み注意: {str(e)}</span>
                </div>
                ''', unsafe_allow_html=True)
                data_status = "error"
            
            # 戦略数の表示
            try:
                strategies = data_manager.get_all_strategies()
                st.markdown(f'''
                <div style="display: flex; align-items: center; gap: 8px; color: #3b82f6; margin: 8px 0;">
                    <svg class="icon-svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M12 2L15.09 8.26L22 9L17 14L18.18 21L12 17.77L5.82 21L7 14L2 9L8.91 8.26L12 2Z"/>
                    </svg>
                    <span>学習済み戦略: {len(strategies)}個</span>
                </div>
                ''', unsafe_allow_html=True)
            except Exception:
                st.markdown('''
                <div style="display: flex; align-items: center; gap: 8px; color: #6b7280; margin: 8px 0;">
                    <svg class="icon-svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M12 2L15.09 8.26L22 9L17 14L18.18 21L12 17.77L5.82 21L7 14L2 9L8.91 8.26L12 2Z"/>
                    </svg>
                    <span>学習済み戦略: 0個</span>
                </div>
                ''', unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"✕ システムエラー: {str(e)}")
            data_manager = None
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # メインコンテンツ
    if data_manager is None:
        st.error("システムの初期化に失敗しました。設定ファイルとデータファイルを確認してください。")
        
        with st.expander("▶ トラブルシューティング"):
            st.markdown("""
            **必要なファイル:**
            1. `config.ini` - システム設定ファイル
            2. `data/LOTO6情報.csv` - LOTO6履歴データ
            3. `data/allloto6.txt` - 全組み合わせデータ（オプション）
            
            **確認事項:**
            - ファイルが正しい場所に配置されているか
            - ファイルの読み取り権限があるか
            - CSVファイルの文字エンコーディングが正しいか（Shift-JIS推奨）
            
            **ディレクトリ構造:**
            ```
            Prometheus-L6/
            ├── main.py
            ├── config.ini
            ├── data/
            │   ├── LOTO6情報.csv
            │   └── allloto6.txt
            └── app/
                └── ...
            ```
            """)
        
        st.stop()
    
    # ページルーティング
    try:
        if page == "ダッシュボード":
            show_dashboard_page(data_manager)
        
        elif page == "戦略管理":
            app.ui.strategy_management.show_strategy_management(data_manager)
        
        elif page == "予測生成":
            show_prediction(data_manager)
        
        elif page == "シミュレーション":
            show_simulation(data_manager)
        
        elif page == "システム情報":
            show_system_info(data_manager)
            
    except Exception as e:
        st.error(f"ページの表示中にエラーが発生しました: {e}")
        logger.error(f"Page display error: {e}")
        
        if st.button("↻ ページをリロード"):
            st.rerun()

def show_dashboard_page(data_manager: DataManager):
    """
    ダッシュボードページの表示
    """
    st.markdown('''
    <h2 style="display: flex; align-items: center; gap: 12px; margin-bottom: 20px;">
        <svg class="icon-svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/>
            <line x1="9" y1="9" x2="9" y2="15"/>
            <line x1="15" y1="9" x2="15" y2="15"/>
        </svg>
        ダッシュボード
    </h2>
    ''', unsafe_allow_html=True)
    try:
        # データ読み込み（エラーハンドリング強化）
        try:
            df_history = data_manager.load_loto6_history()
        except UnicodeDecodeError as e:
            st.markdown('''
            <div style="display: flex; align-items: center; gap: 8px; color: #ef4444; margin: 16px 0;">
                <svg class="icon-svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <circle cx="12" cy="12" r="10"/>
                    <line x1="15" y1="9" x2="9" y2="15"/>
                    <line x1="9" y1="9" x2="15" y2="15"/>
                </svg>
                <span>文字エンコーディングエラーが発生しました</span>
            </div>
            ''', unsafe_allow_html=True)
            
            with st.expander("🔧 修正方法"):
                st.markdown("""
                **エンコーディングエラーの解決法:**
                
                1. **ファイルの文字コードを確認**
                   - CSVファイルがShift-JIS形式であることを確認
                   - UTF-8で保存されている場合は、Shift-JISに変換
                
                2. **手動変換方法**
                   - Excelで開いて「名前を付けて保存」→「CSV (カンマ区切り)」
                   - または、テキストエディタで「Shift-JIS」として保存
                
                3. **別の解決法**
                   - ファイルをUTF-8で保存し直す
                   - または、システムが自動的に複数エンコーディングを試行
                """)
                
                st.code(f"詳細エラー: {str(e)}")
            
            return
            
        except FileNotFoundError:
            st.error("✕ LOTO6履歴データファイルが見つかりません")
            st.markdown("**必要なファイル:** `data/LOTO6情報.csv`")
            return
            
        except Exception as e:
            st.error(f"✕ データ読み込みエラー: {str(e)}")
            return
        
        # 特徴量データの生成（キャッシュがあれば使用）
        features_cache_path = "output/cache/features.csv"
        
        if os.path.exists(features_cache_path):
            try:
                df_features = pd.read_csv(features_cache_path)
                st.markdown('''
                <div style="display: flex; align-items: center; gap: 8px; color: #3b82f6; margin: 8px 0;">
                    <svg class="icon-svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M19 21V5a2 2 0 0 0-2-2H7a2 2 0 0 0-2 2v16m14 0h2m-2 0h-5m-9 0H3m2 0h5M9 7h1m-1 4h1m4-4h1m-1 4h1m-5 8v-1a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v1"/>
                    </svg>
                    <span>キャッシュされた特徴量データを使用しています</span>
                </div>
                ''', unsafe_allow_html=True)
            except Exception:
                df_features = None
                st.markdown('''
                <div style="display: flex; align-items: center; gap: 8px; color: #f59e0b; margin: 8px 0;">
                    <svg class="icon-svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
                        <line x1="12" y1="9" x2="12" y2="13"/>
                        <line x1="12" y1="17" x2="12.01" y2="17"/>
                    </svg>
                    <span>特徴量キャッシュの読み込みに失敗しました</span>
                </div>
                ''', unsafe_allow_html=True)
        else:
            df_features = None
            st.markdown('''
            <div style="display: flex; align-items: center; gap: 8px; color: #3b82f6; margin: 8px 0;">
                <svg class="icon-svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <circle cx="12" cy="12" r="10"/>
                    <line x1="12" y1="6" x2="12" y2="12"/>
                    <line x1="16" y1="16" x2="12" y2="12"/>
                </svg>
                <span>初回実行時は特徴量生成に時間がかかる場合があります</span>
            </div>
            ''', unsafe_allow_html=True)
        
        # ダッシュボード表示
        show_dashboard(df_history, df_features)
        
        # 特徴量生成ボタン
        if df_features is None:
            st.markdown("---")
            st.markdown('''
            <div style="display: flex; align-items: center; gap: 8px; margin: 16px 0;">
                <svg class="icon-svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <circle cx="12" cy="12" r="10"/>
                    <path d="m16 12-4-4-4 4"/>
                    <path d="m12 16 4-4-4-4"/>
                </svg>
                <span>初回セットアップ</span>
            </div>
            ''', unsafe_allow_html=True)
            
            if st.button("特徴量を生成（初回セットアップ）"):
                with st.spinner("特徴量を生成中..."):
                    feature_engine = FeatureEngine(df_history)
                    df_features = feature_engine.run_all()
                    
                    # キャッシュに保存
                    os.makedirs(os.path.dirname(features_cache_path), exist_ok=True)
                    df_features.to_csv(features_cache_path, index=False)
                    
                    st.success("✅ 特徴量生成が完了しました！ページをリロードしてください。")
                    st.rerun()
        
    except Exception as e:
        st.error(f"ダッシュボードの表示中にエラーが発生しました: {e}")
        logger.error(f"Dashboard error: {e}")

def show_system_info(data_manager: DataManager):
    """
    システム情報ページ
    """
    st.markdown('''
    <h2 style="display: flex; align-items: center; gap: 12px; margin-bottom: 20px;">
        <svg class="icon-svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="12" cy="12" r="3"/>
            <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1 1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/>
        </svg>
        システム情報
    </h2>
    ''', unsafe_allow_html=True)
    
    # システム概要
    st.markdown('''
    <h3 style="display: flex; align-items: center; gap: 8px; margin-bottom: 16px;">
        <svg class="icon-svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="12" cy="12" r="10"/>
            <line x1="12" y1="6" x2="12" y2="12"/>
            <line x1="16" y1="16" x2="12" y2="12"/>
        </svg>
        システム概要
    </h3>
    ''', unsafe_allow_html=True)
    
    st.markdown("""
    **Prometheus-L6** は、LOTO6を戦略的投資対象として扱うための包括的な分析・意思決定支援ツールです。
    
    ### 🎯 主要機能
    
    1. **📊 データ分析ダッシュボード**
       - 過去の当選データの詳細分析
       - 数字出現パターンの可視化
       - ホット/コールドナンバーの特定
    
    2. **⚙️ 戦略管理**
       - AI機械学習モデルの学習と管理
       - バックテストによる戦略検証
       - 複数戦略の性能比較
    
    3. **🎯 予測生成**
       - 学習済みAIによるポートフォリオ生成
       - ケリー基準による最適投資額算出
       - 逆張り戦略による期待収益最大化
    
    4. **📈 シミュレーション**
       - モンテカルロ法による将来予測
       - 戦略のリスク・リターン分析
       - 投資判断の定量的支援
    """)
    
    # 技術情報
    st.markdown('''
    <h3 style="display: flex; align-items: center; gap: 8px; margin: 20px 0 16px 0;">
        <svg class="icon-svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <polygon points="12 2 15.09 8.26 22 9 17 14 18.18 21 12 17.77 5.82 21 7 14 2 9 8.91 8.26 12 2"/>
        </svg>
        技術情報
    </h3>
    ''', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **使用技術:**
        - Python 3.9+
        - Streamlit (UI)
        - XGBoost (機械学習)
        - Pandas/NumPy (データ処理)
        - Plotly (可視化)
        """)
    
    with col2:
        st.markdown("""
        **アルゴリズム:**
        - 適応学習・バックテスト
        - ケリー基準
        - モンテカルロシミュレーション
        - 特徴量エンジニアリング
        """)
    
    # ファイル状態
    st.markdown('''
    <h3 style="display: flex; align-items: center; gap: 8px; margin: 20px 0 16px 0;">
        <svg class="icon-svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
            <polyline points="14,2 14,8 20,8"/>
            <line x1="16" y1="13" x2="8" y2="13"/>
            <line x1="16" y1="17" x2="8" y2="17"/>
            <polyline points="10,9 9,9 8,9"/>
        </svg>
        ファイル状態
    </h3>
    ''', unsafe_allow_html=True)
    
    file_info = []
    
    # データファイル
    try:
        df_history = data_manager.load_loto6_history()
        file_size = len(df_history)
        file_info.append({
            "ファイル": "LOTO6情報.csv",
            "状態": "✅ 正常",
            "サイズ": f"{file_size}件",
            "最終更新": "N/A"
        })
    except Exception as e:
        file_info.append({
            "ファイル": "LOTO6情報.csv", 
            "状態": f"✕ エラー: {str(e)[:30]}...",
            "サイズ": "N/A",
            "最終更新": "N/A"
        })
    
    # 戦略データ
    try:
        strategies = data_manager.get_all_strategies()
        file_info.append({
            "ファイル": "strategies.db",
            "状態": "✅ 正常",
            "サイズ": f"{len(strategies)}戦略",
            "最終更新": "N/A"
        })
    except Exception as e:
        file_info.append({
            "ファイル": "strategies.db",
            "状態": f"✕ エラー: {str(e)[:30]}...",
            "サイズ": "N/A", 
            "最終更新": "N/A"
        })
    
    # 特徴量キャッシュ
    features_path = "output/cache/features.csv"
    if os.path.exists(features_path):
        file_size = os.path.getsize(features_path) / 1024 / 1024  # MB
        file_info.append({
            "ファイル": "features.csv",
            "状態": "✅ 正常",
            "サイズ": f"{file_size:.1f}MB",
            "最終更新": "N/A"
        })
    else:
        file_info.append({
            "ファイル": "features.csv",
            "状態": "○ 未生成",
            "サイズ": "N/A",
            "最終更新": "N/A"
        })
    
    # ファイル状態テーブル
    df_files = pd.DataFrame(file_info)
    st.dataframe(df_files, use_container_width=True)
    
    # 免責事項
    st.markdown('''
    <h3 style="display: flex; align-items: center; gap: 8px; margin: 20px 0 16px 0;">
        <svg class="icon-svg status-warning" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
            <line x1="12" y1="9" x2="12" y2="13"/>
            <line x1="12" y1="17" x2="12.01" y2="17"/>
        </svg>
        重要な注意事項
    </h3>
    ''', unsafe_allow_html=True)
    
    st.warning("""
    **免責事項:**
    - 本ツールは統計的分析に基づく意思決定支援が目的です
    - **当選は保証されません**
    - 投資は自己責任で実行してください
    - 過去の実績は将来の結果を保証しません
    
    **推奨事項:**
    - 必ず少額から開始してください
    - 複数戦略の分散投資を心がけてください
    - 定期的な戦略見直しを行ってください
    """)

if __name__ == "__main__":
    main()
