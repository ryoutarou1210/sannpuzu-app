import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import math
import numpy as np
import io
import japanize_matplotlib

# ---------------------------------------------------------
# ユーティリティ関数
# ---------------------------------------------------------
def get_auto_scale_info(max_val):
    if max_val == 0 or pd.isna(max_val):
        return 1.0, "", 0
    exponent = math.floor(math.log10(max_val) / 3) * 3
    if exponent == 0:
        return 1.0, "", 0
    si_prefixes = {
        -12: 'p', -9: 'n', -6: r'$\mu$', -3: 'm', 
        0: '', 3: 'k', 6: 'M', 9: 'G', 12: 'T'
    }
    scale_factor = 10 ** (-exponent)
    prefix = si_prefixes.get(exponent, "")
    return scale_factor, prefix, exponent

def scientific_formatter(x, pos):
    """軸目盛り用のフォーマッター"""
    if x == 0:
        return "0"
    exponent = int(math.floor(math.log10(abs(x))))
    mantissa = x / (10 ** exponent)
    return f"${mantissa:.2f} \\times 10^{{{exponent}}}$"

def to_latex_sci(x):
    """
    数値をLaTeX形式の文字列に変換する関数
    - 指数が -1, 0, 1 の場合は通常の小数表記にする
    - それ以外は a \times 10^b の形式にする
    """
    if x == 0:
        return "0"
    
    exponent = int(math.floor(math.log10(abs(x))))
    
    if exponent in [-1, 0, 1]:
        return f"{x:.3g}"

    mantissa = x / (10 ** exponent)
    return f"{mantissa:.2f} \\times 10^{{{exponent}}}"

# ---------------------------------------------------------
# メインアプリ
# ---------------------------------------------------------
def main():
    st.set_page_config(page_title="散布図作成ツール", layout="wide", page_icon="📈")

    # ==========================================
    # 1. サイドバー（データ読み込み）
    # ==========================================
    st.sidebar.header("📂 データ読み込み")
    uploaded_file = st.sidebar.file_uploader("ファイルを選択", type=["csv", "xlsx"])

    df = None
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.xlsx'):
                xl = pd.ExcelFile(uploaded_file)
                sheet_names = xl.sheet_names
                st.sidebar.subheader("シート選択")
                if len(sheet_names) > 1:
                    selected_sheet = st.sidebar.selectbox("対象のシート", sheet_names)
                else:
                    selected_sheet = sheet_names[0]
                df = pd.read_excel(uploaded_file, sheet_name=selected_sheet)
            else:
                df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.sidebar.error(f"読み込みエラー: {e}")

    # ==========================================
    # メインエリア
    # ==========================================
    st.title("📈 散布図作成ツール")

    if df is None:
        st.info("👈 左側のサイドバーからファイルをアップロードしてください。")
        return

    # --- 列選択 ---
    st.markdown("##### 1. 列の選択")
    event = st.dataframe(
        df,
        use_container_width=True,
        on_select="rerun",
        selection_mode="multi-column",
        height=300
    )
    selected_cols = event.selection.get("columns", [])
    
    if len(selected_cols) == 0 or len(selected_cols) % 2 != 0:
        st.info("👆 表からグラフにしたい列を **偶数個** 選択してください。")
        return

    # --- データ抽出処理 ---
    series_list = [] 
    all_x_values = []
    all_y_values = []

    for i in range(0, len(selected_cols), 2):
        col_x = selected_cols[i]
        col_y = selected_cols[i+1]
        
        clean_x = pd.to_numeric(df[col_x], errors='coerce')
        clean_y = pd.to_numeric(df[col_y], errors='coerce')
        pair_df = pd.DataFrame({'X': clean_x, 'Y': clean_y}).dropna()
        pair_df = pair_df.sort_values(by='X')

        if not pair_df.empty:
            series_list.append({
                "x": pair_df['X'],
                "y": pair_df['Y'],
                "col_x_name": col_x, 
                "col_y_name": col_y, 
                "label_name": col_x  
            })
            all_x_values.extend(pair_df['X'].abs().tolist())
            all_y_values.extend(pair_df['Y'].abs().tolist())

    if not series_list:
        st.error("有効なデータがありません。")
        return

    # --- グラフ設定 ---
    st.divider()
    st.markdown("##### 2. グラフ設定")

    global_max_x = max(all_x_values) if all_x_values else 0
    global_max_y = max(all_y_values) if all_y_values else 0
    
    col_ui1, col_ui2 = st.columns(2)
    
    # X軸設定
    with col_ui1:
        st.markdown("**X軸設定**")
        auto_scale_x = st.checkbox("自動スケーリング (X)", value=False)
        x_scale_factor, x_prefix = 1.0, ""
        if auto_scale_x:
            x_scale_factor, x_prefix, x_exp = get_auto_scale_info(global_max_x)
            if x_scale_factor != 1.0:
                st.info(f"💡 スケール: **{x_prefix}** ($10^{{{x_exp}}}$)")
        x_label = st.text_input("X軸ラベル (TeX形式は$で囲む)", value=series_list[0]['col_x_name'])


    # Y軸設定
    with col_ui2:
        st.markdown("**Y軸設定**")
        auto_scale_y = st.checkbox("自動スケーリング (Y)", value=True)
        y_scale_factor, y_prefix = 1.0, ""
        if auto_scale_y:
            y_scale_factor, y_prefix, y_exp = get_auto_scale_info(global_max_y)
            if y_scale_factor != 1.0:
                st.info(f"💡 スケール: **{y_prefix}** ($10^{{{y_exp}}}$)")
        y_label = st.text_input("Y軸ラベル (TeX形式は$で囲む)", value=series_list[0]['col_y_name'])

    # ==========================================
    # 3. 近似直線設定（多重対応）
    # ==========================================
    st.divider()
    st.markdown("##### 3. 近似直線の設定")

    col_fit_setting, col_fit_sliders = st.columns([1, 2])
    
    fit_configs = [] 

    with col_fit_setting:
        enable_fitting = st.checkbox("近似直線を追加する", value=False)
        
        if enable_fitting:
            num_fits = st.number_input("直線の本数", min_value=1, max_value=5, value=1)
            extend_full = st.checkbox("線をグラフ全体に延長", value=True, help="OFFにすると、選択範囲の少し外側までしか線を描画しません。")
    
    if enable_fitting:
        all_scaled_x = []
        for s in series_list:
             all_scaled_x.extend((s['x'] * (x_scale_factor if auto_scale_x else 1.0)).tolist())
        
        if all_scaled_x:
            min_val = float(min(all_scaled_x))
            max_val = float(max(all_scaled_x))
            margin_val = (max_val - min_val) * 0.05 if max_val != min_val else 1.0
            
            with col_fit_sliders:
                for i in range(num_fits):
                    st.markdown(f"**近似直線 {i+1} の範囲**")
                    f_range = st.slider(
                        f"Fit {i+1} 範囲指定",
                        min_value=min_val - margin_val,
                        max_value=max_val + margin_val,
                        value=(min_val, max_val),
                        step=(max_val - min_val) / 200 if max_val != min_val else 0.1,
                        key=f"fit_slider_{i}",
                        label_visibility="collapsed"
                    )
                    fit_configs.append(f_range)

    # ==========================================
    # 4. 凡例編集
    # ==========================================
    st.divider()
    st.markdown("##### 4. 凡例の設定")
    cols = st.columns(len(series_list))
    for i, s in enumerate(series_list):
        new_label = cols[i].text_input(f"データ {i+1} 名前", value=s['col_x_name'], key=f"legend_{i}")
        series_list[i]['label_name'] = new_label

    # ==========================================
    # プロット描画処理
    # ==========================================
    st.divider()
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # --- 副目盛りを有効化 ---
    ax.minorticks_on() 
    # ---------------------

    ax.tick_params(direction="in", top=True, right=True, which="both")
    
    colors = ['black', 'blue', 'red', 'orange', 'green', 'purple', 'brown']
    markers = ['o', 's', '^', 'D', 'v', '<', '>']
    linestyles = ['--', '-.', ':', '--', '-.']

    plot_x_min_all = []
    plot_x_max_all = []
    
    # --- Y軸範囲固定のためのデータ収集用リスト ---
    plot_y_min_all = []
    plot_y_max_all = []
    # ----------------------------------------
    
    is_fit_plotted = False

    for idx, s in enumerate(series_list):
        x_plot = s['x'] * (x_scale_factor if auto_scale_x else 1.0)
        y_plot = s['y'] * (y_scale_factor if auto_scale_y else 1.0)
        
        plot_x_min_all.append(x_plot.min())
        plot_x_max_all.append(x_plot.max())
        
        # --- データ点のみの最小・最大を記録 ---
        plot_y_min_all.append(y_plot.min())
        plot_y_max_all.append(y_plot.max())
        # ----------------------------------
        
        base_color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]

        # 生データのプロット
        ax.plot(x_plot, y_plot, label=s['label_name'], color=base_color, 
                marker=marker, linestyle='-', linewidth=0, markersize=4, alpha=1)

        # 近似直線のプロット
        if enable_fitting:
            for fit_idx, (f_min, f_max) in enumerate(fit_configs):
                mask = (x_plot >= f_min) & (x_plot <= f_max)
                x_fit = x_plot[mask]
                y_fit = y_plot[mask]

                if len(x_fit) > 1:
                    try:
                        coeffs = np.polyfit(x_fit, y_fit, 1)
                        poly_func = np.poly1d(coeffs)
                        
                        if extend_full:
                            x_line_min = x_plot.min()
                            x_line_max = x_plot.max()
                            padding = (x_line_max - x_line_min) * 0.1
                            x_line = np.linspace(x_line_min - padding, x_line_max + padding, 100)
                        else:
                            padding = (f_max - f_min) * 0.2
                            x_line = np.linspace(f_min - padding, f_max + padding, 100)

                        y_line = poly_func(x_line)
                        
                        # 数値を変換
                        slope = coeffs[0]
                        intercept = coeffs[1]
                        
                        slope_latex = to_latex_sci(slope)
                        intercept_latex = to_latex_sci(abs(intercept))
                        sign = "+" if intercept >= 0 else "-"
                        
                        fit_label = f"Fit{fit_idx+1}: $y = {slope_latex}x {sign} {intercept_latex}$"
                        
                        ls = linestyles[fit_idx % len(linestyles)]
                        
                        ax.plot(x_line, y_line, color=base_color, linestyle=ls, 
                                linewidth=1.5, label=fit_label, alpha=0.9)
                        
                        is_fit_plotted = True

                    except Exception as e:
                        pass

    # 軸フォーマット設定
    if not auto_scale_x:
        if global_max_x > 1000 or (global_max_x < 0.001 and global_max_x > 0):
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(scientific_formatter))
    if not auto_scale_y:
        if global_max_y > 1000 or (global_max_y < 0.001 and global_max_y > 0):
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(scientific_formatter))

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    
    # X軸範囲設定
    if plot_x_min_all and plot_x_max_all:
        x_all_min = min(plot_x_min_all)
        x_all_max = max(plot_x_max_all)
        margin_x = (x_all_max - x_all_min) * 0.05 if x_all_max != x_all_min else 1.0
        ax.set_xlim(x_all_min - margin_x, x_all_max + margin_x)

    # --- Y軸範囲設定（データ点に合わせて固定） ---
    if plot_y_min_all and plot_y_max_all:
        y_all_min = min(plot_y_min_all)
        y_all_max = max(plot_y_max_all)
        diff = y_all_max - y_all_min
        # マージンを10%程度とる
        margin_y = diff * 0.1 if diff != 0 else (abs(y_all_max) * 0.1 if y_all_max != 0 else 1.0)
        ax.set_ylim(y_all_min - margin_y, y_all_max + margin_y)
    # ---------------------------------------

    # 凡例表示ロジック
    if len(series_list) > 1 or is_fit_plotted:
        ax.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0, fontsize=6)

    _, col_center, _ = st.columns([1, 5, 1]) 
    with col_center:
        st.pyplot(fig, use_container_width=False)

    # 画像保存
    st.divider()
    col_save_input, col_save_btn = st.columns([3, 1])
    with col_save_input:
        file_name_input = st.text_input("保存ファイル名", value="multi_fit_plot")
    with col_save_btn:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
        buf.seek(0)
        st.download_button(
            label="画像を保存 (PNG)",
            data=buf,
            file_name=f"{file_name_input}.png",
            mime="image/png",
            type="primary"
        )

if __name__ == "__main__":
    main()