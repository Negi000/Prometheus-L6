"""
Prometheus-L6 メインアプリケーション
LOTO6戦略的投資支援ツール

このツールは以下の機能を提供します：
- 過去データの詳細分析
- AI を使った適応学習シス        # ナビゲーションメニュー
        st.markdown("### ▶ メニュー")
        
        page = st.radio(
            "機能を        if page == "■ ダッシュボード":
            show_dashboard_page(data_manager)
        
        elif page == def show_system_info(data_manager: DataManager):
    """
    システム情報ページ
    """
    st.header("▼ システム情報")
    
    # システム概要
    st.subheader("▶ システム概要"):
            show_strategy_management(data_manager)
        
        elif page == "◆ 予測生成":
            show_prediction(data_manager)
        
        elif page == "▲ シミュレーション":
            show_simulation(data_manager)
        
        elif page == "▼ システム情報":
            show_system_info(data_manager)          [
                "■ ダッシュボード",
                "● 戦略管理", 
                "◆ 予測生成",
                "▲ シミュレーション",
                "▼ システム情報"
            ],
            index=0
        )
        
        st.markdown("---")
        
        # システム状態
        st.markdown("### ◆ システム状態") ポートフォリオ生成と最適化
- モンテカルロシミュレーション
"""

import streamlit as st
import sys
import os
import logging
import pandas as pd

# アプリのルートディレクトリをパスに追加
app_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(app_root)
sys.path.append(os.path.join(app_root, 'app'))

# ローカルモジュールのインポート
try:
    from app.data_manager import DataManager
    from app.feature_engine import FeatureEngine
    from app.ui.dashboard import show_dashboard
    from app.ui.strategy_management import show_strategy_management
    from app.ui.prediction import show_prediction
    from app.ui.simulation import show_simulation
except ImportError as e:
    st.error(f"モジュールのインポートに失敗しました: {e}")
    st.stop()

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('prometheus_l6.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# ページ設定
st.set_page_config(
    page_title="Prometheus-L6",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# カスタムCSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1e3a8a;
        margin-bottom: 1rem;
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
    }
</style>
""", unsafe_allow_html=True)

def main():
    """
    メインアプリケーション
    """
    # ヘッダー
    st.markdown('<h1 class="main-header">🎯 Prometheus-L6</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">LOTO6 戦略的投資支援システム</p>', unsafe_allow_html=True)
    
    # サイドバー
    with st.sidebar:
        st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
        
        st.image("https://via.placeholder.com/200x100/1e3a8a/ffffff?text=Prometheus-L6", 
                caption="AI-Powered LOTO6 Strategy Tool")
        
        st.markdown("---")
        
        # ナビゲーションメニュー
        st.markdown("### ⚡ メニュー")
        
        page = st.radio(
            "機能を選択してください",
            [
                "� ダッシュボード",
                "⚙️ 戦略管理", 
                "� 予測生成",
                "🧪 シミュレーション",
                "🔍 システム情報"
            ],
            index=0
        )
        
        st.markdown("---")
        
        # システム状態
        st.markdown("### � システム状態")
        
        try:
            # データマネージャーの初期化テスト
            data_manager = DataManager()
            st.success("✅ システム正常")
            
            # データファイルの存在確認
            try:
                df_history = data_manager.load_loto6_history()
                st.info(f"📄 履歴データ: {len(df_history)}件")
                data_status = "success"
            except UnicodeDecodeError as e:
                st.error(f"🚨 文字エンコーディングエラー: ファイルの文字コードを確認してください")
                st.code(f"エラー詳細: {str(e)}")
                data_status = "encoding_error"
            except FileNotFoundError:
                st.error("🚨 データファイルが見つかりません")
                data_status = "file_not_found"
            except Exception as e:
                st.warning(f"⚠️ データ読み込み注意: {str(e)}")
                data_status = "error"
            
            # 戦略数の表示
            try:
                strategies = data_manager.get_all_strategies()
                st.info(f"🎯 学習済み戦略: {len(strategies)}個")
            except Exception:
                st.info("🎯 学習済み戦略: 0個")
                
        except Exception as e:
            st.error(f"❌ システムエラー: {str(e)[:50]}...")
            data_manager = None
        
        st.markdown("---")
        
        # バージョン情報
        st.markdown("### 📋 バージョン情報")
        st.text("Version: 1.0.0")
        st.text("Build: 2025.06.20")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # メインコンテンツ
    if data_manager is None:
        st.error("システムの初期化に失敗しました。設定ファイルとデータファイルを確認してください。")
        
        with st.expander("🔧 トラブルシューティング"):
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
        if page == "� ダッシュボード":
            show_dashboard_page(data_manager)
        
        elif page == "⚙️ 戦略管理":
            show_strategy_management(data_manager)
        
        elif page == "� 予測生成":
            show_prediction(data_manager)
        
        elif page == "🧪 シミュレーション":
            show_simulation(data_manager)
        
        elif page == "🔍 システム情報":
            show_system_info(data_manager)
            
    except Exception as e:
        st.error(f"ページの表示中にエラーが発生しました: {e}")
        logger.error(f"Page display error: {e}")
        
        if st.button("🔄 ページをリロード"):
            st.rerun()

def show_dashboard_page(data_manager: DataManager):
    """
    ダッシュボードページの表示
    """
    try:
        # データ読み込み（エラーハンドリング強化）
        try:
            df_history = data_manager.load_loto6_history()
        except UnicodeDecodeError as e:
            st.error("🚨 文字エンコーディングエラーが発生しました")
            
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
            st.error("🚨 LOTO6履歴データファイルが見つかりません")
            st.markdown("**必要なファイル:** `data/LOTO6情報.csv`")
            return
            
        except Exception as e:
            st.error(f"🚨 データ読み込みエラー: {str(e)}")
            return
        
        # 特徴量データの生成（キャッシュがあれば使用）
        features_cache_path = "output/cache/features.csv"
        
        if os.path.exists(features_cache_path):
            try:
                df_features = pd.read_csv(features_cache_path)
                st.info("💾 キャッシュされた特徴量データを使用しています")
            except Exception:
                df_features = None
                st.warning("⚠️ 特徴量キャッシュの読み込みに失敗しました")
        else:
            df_features = None
            st.info("💡 初回実行時は特徴量生成に時間がかかる場合があります")
        
        # ダッシュボード表示
        show_dashboard(df_history, df_features)
        
        # 特徴量生成ボタン
        if df_features is None:
            st.markdown("---")
            if st.button("🔧 特徴量を生成（初回セットアップ）"):
                with st.spinner("特徴量を生成中..."):
                    feature_engine = FeatureEngine(df_history)
                    df_features = feature_engine.run_all()
                    
                    # キャッシュに保存
                    os.makedirs(os.path.dirname(features_cache_path), exist_ok=True)
                    df_features.to_csv(features_cache_path, index=False)
                    
                    st.success("✅ 特徴量生成が完了しました！")
                    st.rerun()
    
    except Exception as e:
        st.error(f"ダッシュボード表示中にエラーが発生しました: {e}")
        logger.error(f"Dashboard error: {e}")

def show_system_info(data_manager: DataManager):
    """
    システム情報ページ
    """
    st.header("🔍 システム情報")
    
    # システム概要
    st.subheader("� システム概要")
    
    st.markdown("""
    **Prometheus-L6** は、LOTO6を戦略的投資対象として扱うための包括的な分析・意思決定支援ツールです。
    
    ### ⚡ 主要機能
    
    1. **� データ分析ダッシュボード**
       - 過去の当選データの詳細分析
       - 数字出現パターンの可視化
       - ホット/コールドナンバーの特定
    
    2. **⚙️ 戦略管理**
       - AI機械学習モデルの学習と管理
       - バックテストによる戦略検証
       - 複数戦略の性能比較
    
    3. **� 予測生成**
       - 学習済みAIによるポートフォリオ生成
       - ケリー基準による最適投資額算出
       - 逆張り戦略による期待収益最大化
    
    4. **🧪 シミュレーション**
       - モンテカルロ法による将来予測
       - 戦略のリスク・リターン分析
       - 投資判断の定量的支援
    """)
    
    # 技術情報
    st.subheader("�️ 技術情報")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **使用技術:**
        - Python 3.9+
        - Streamlit (UI)
        - XGBoost (機械学習)
        - Pandas/NumPy (データ処理)
        - Plotly (可視化)
        - SQLite (データベース)
        """)
    
    with col2:
        st.markdown("""
        **アルゴリズム:**
        - 適応学習システム
        - スライディングウィンドウバックテスト
        - ケリー基準最適投資
        - モンテカルロシミュレーション
        - 逆張り戦略最適化
        """)
    
    # ファイル情報
    st.subheader("📁 ファイル情報")
    
    file_info = []
    
    # データファイル
    try:
        df_history = data_manager.load_loto6_history()
        file_info.append({
            "ファイル": "LOTO6情報.csv",
            "状態": "✅ 正常",
            "サイズ": f"{len(df_history)}行",
            "最終更新": "N/A"
        })
    except Exception as e:
        file_info.append({
            "ファイル": "LOTO6情報.csv", 
            "状態": f"❌ エラー: {str(e)[:30]}...",
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
            "状態": f"❌ エラー: {str(e)[:30]}...",
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
            "状態": "⚠️ 未生成",
            "サイズ": "N/A",
            "最終更新": "N/A"
        })
    
    st.table(pd.DataFrame(file_info))
    
    # 使用方法
    st.subheader("📚 使用方法")
    
    with st.expander("💡 基本的な使い方"):
        st.markdown("""
        ### ステップ1: 初期セットアップ
        1. 「ダッシュボード」で過去データの概要を確認
        2. 初回は特徴量生成を実行（数分かかります）
        
        ### ステップ2: 戦略作成
        1. 「戦略管理」→「新規戦略学習」
        2. バックテスト期間とパラメータを設定
        3. 学習実行（10-30分程度）
        
        ### ステップ3: 予測生成
        1. 「予測生成」で学習済み戦略を選択
        2. 購入条件（口数、軸数字等）を設定
        3. ポートフォリオ生成実行
        
        ### ステップ4: リスク評価
        1. 「シミュレーション」で戦略の将来性能を評価
        2. 複数戦略を比較検討
        3. 最適な戦略を選択
        """)
    
    # 免責事項
    st.subheader("⚠️ 重要な注意事項")
    
    st.warning("""
    **免責事項:**
    - 本ツールは統計的分析に基づく意思決定支援を目的としており、当選を保証するものではありません
    - 投資は自己責任で行ってください
    - 過去の実績は将来の結果を保証するものではありません
    - ギャンブル依存症にご注意ください
    """)

if __name__ == "__main__":
    main()
