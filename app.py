import streamlit as st
import pandas as pd
import io
import re
from datetime import datetime

# 尝试导入音频分析库（可选）
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

try:
    import soundfile as sf
    HAS_SOUND_FILE = True
except ImportError:
    HAS_SOUND_FILE = False

# ========== 工具函数 ==========

MONTHS = [f"{i}月新歌" for i in range(1, 13)]

# 过度电子化的风格关键词（高难歌排除）
ELECTRONIC_KEYWORDS = ["电音", "电子", "EDM", "Synth", "合成器", "Techno", "House", "Trance"]
# 鼓点明确/摇摆性强的风格关键词（高难歌优先）
GOOD_STYLE_KEYWORDS = ["嘻哈", "嘻哈说唱", "R&B", "Pop", "Dance", "放克", "迪斯科", "Funk", "Disco", "Hip", "Dance-Pop"]

def norm(s):
    if pd.isna(s):
        return ""
    s = str(s).strip().lower()
    s = re.sub(r'\([^)]*\)', '', s)
    s = re.sub(r'（[^）]*）', '', s)
    s = re.sub(r'[\s\-_.,!?！？、，。]+', '', s)
    return s

def style_is_too_electronic(style_str):
    """判断风格是否过度电子化"""
    if pd.isna(style_str):
        return False
    s = str(style_str)
    return any(kw in s for kw in ELECTRONIC_KEYWORDS)

def style_is_good_for_hard(style_str):
    """判断风格是否适合高难（鼓点明确/摇摆性强）"""
    if pd.isna(style_str):
        return True  # 无风格信息时不过滤
    s = str(style_str)
    return any(kw in s for kw in GOOD_STYLE_KEYWORDS)

def select_mobile_songs(records_df, exclude_japanese=True, dedup_set=None):
    """
    手游选歌：35常规(BPM>=75) + 10高难(BPM>=140)
    高难歌额外排除过度电子化，优先鼓点明确的风格
    """
    df = records_df.copy()
    name_col = 2
    singer_col = 3
    lang_col = 4
    bpm_col = 5
    style_col = 7  # 风格特点

    if exclude_japanese and lang_col < len(df.columns):
        df = df[df.iloc[:, lang_col] != "日语"]

    def valid_bpm(x, min_bpm):
        try:
            bpm = float(x)
            return bpm >= min_bpm and bpm < 300
        except:
            return False

    if dedup_set:
        def not_in_dedup(row):
            nm = norm(row.iloc[name_col])
            sg = norm(row.iloc[singer_col])
            return (nm, sg) not in dedup_set
        df = df[df.apply(not_in_dedup, axis=1)]

    # 常规 35 首（BPM>=75，优先接近100）
    df_reg = df[df.iloc[:, bpm_col].apply(lambda x: valid_bpm(x, 75))].copy()
    df_reg['_score'] = df_reg.iloc[:, bpm_col].apply(lambda x: abs(float(x) - 100) if pd.notna(x) else 999)
    df_reg = df_reg.sort_values('_score').head(35).drop(columns=['_score'])

    # 高难 10 首（BPM>=140，排除过度电子化，优先适合的风格）
    df_hard_pool = df[df.iloc[:, bpm_col].apply(lambda x: valid_bpm(x, 140))].copy()

    # 标记风格
    if style_col < len(df_hard_pool.columns):
        df_hard_pool['_too_electronic'] = df_hard_pool.iloc[:, style_col].apply(style_is_too_electronic)
        df_hard_pool['_good_style'] = df_hard_pool.iloc[:, style_col].apply(style_is_good_for_hard)
    else:
        df_hard_pool['_too_electronic'] = False
        df_hard_pool['_good_style'] = True

    # 排序：不过度电子化优先 > 适合风格优先 > BPM接近150
    df_hard_pool['_sort'] = (
        df_hard_pool['_too_electronic'].apply(lambda x: 0 if not x else 1) * 1000 +
        df_hard_pool['_good_style'].apply(lambda x: 0 if x else 1) * 100 +
        df_hard_pool.iloc[:, bpm_col].apply(lambda x: abs(float(x) - 150) if pd.notna(x) else 999)
    )
    df_hard = df_hard_pool.sort_values('_sort').head(10).drop(columns=['_too_electronic', '_good_style', '_sort'])

    return df_reg, df_hard


def select_pc_songs(records_df, exclude_japanese=True, dedup_set=None):
    """
    端游选歌：炫舞1=9首常规，炫舞2=18首常规（8-9首复用炫舞1）
    只选常规，BPM>=100
    """
    df = records_df.copy()
    name_col = 2
    singer_col = 3
    lang_col = 4
    bpm_col = 5

    if exclude_japanese and lang_col < len(df.columns):
        df = df[df.iloc[:, lang_col] != "日语"]

    def valid_bpm(x):
        try:
            bpm = float(x)
            return bpm >= 100 and bpm < 300
        except:
            return False

    if dedup_set:
        def not_in_dedup(row):
            nm = norm(row.iloc[name_col])
            sg = norm(row.iloc[singer_col])
            return (nm, sg) not in dedup_set
        df = df[df.apply(not_in_dedup, axis=1)]

    # 炫舞1：9首常规，BPM优先接近120
    df_x51 = df[df.iloc[:, bpm_col].apply(valid_bpm)].copy()
    df_x51['_score'] = df_x51.iloc[:, bpm_col].apply(lambda x: abs(float(x) - 120) if pd.notna(x) else 999)
    df_x51 = df_x51.sort_values('_score').head(9).drop(columns=['_score'])

    # 炫舞2：18首常规
    df_x52 = df[df.iloc[:, bpm_col].apply(valid_bpm)].copy()
    df_x52['_score'] = df_x52.iloc[:, bpm_col].apply(lambda x: abs(float(x) - 120) if pd.notna(x) else 999)
    df_x52 = df_x52.sort_values('_score').head(18).drop(columns=['_score'])

    return df_x51, df_x52


def build_dedup_set_from_file(uploaded_file):
    try:
        df = pd.read_excel(uploaded_file, engine='openpyxl', header=None)
    except Exception as e:
        return set(), f"读取查重表失败: {e}"

    dedup = set()
    for i in range(len(df)):
        row = df.iloc[i]
        if pd.isna(row.iloc[0]) and pd.isna(row.iloc[1]):
            continue
        nm = norm(row.iloc[0]) if len(row) > 0 and pd.notna(row.iloc[0]) else ""
        sg = norm(row.iloc[1]) if len(row) > 1 and pd.notna(row.iloc[1]) else ""
        if nm or sg:
            dedup.add((nm, sg))

    return dedup, f"查重表加载成功，共 {len(df)} 行，构建查重 {len(dedup)} 条"


def analyze_bpm(audio_bytes, filename):
    """分析音频BPM（需要librosa）"""
    if not HAS_LIBROSA:
        return None, "未安装音频分析库(librosa)，此功能不可用", ""

    try:
        import tempfile
        import os

        suffix = os.path.splitext(filename)[1] or '.mp3'
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        y, sr = librosa.load(tmp_path, sr=None, duration=300)
        os.unlink(tmp_path)

        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        bpm = float(tempo)

        segment_duration = 30
        duration = len(y) / sr
        num_segments = max(1, int(duration / segment_duration))

        segment_bpms = []
        for i in range(num_segments):
            start = int(i * segment_duration * sr)
            end = int(min((i+1) * segment_duration * sr, len(y)))
            if end - start < sr * 10:
                continue
            seg_tempo, _ = librosa.beat.beat_track(y=y[start:end], sr=sr)
            segment_bpms.append(float(seg_tempo))

        if len(segment_bpms) <= 1:
            stability = "稳定（无法分段检测）"
            detail = f"总BPM: {bpm:.2f}"
        else:
            bpm_std = np.std(segment_bpms)
            detail = (f"总BPM: {bpm:.2f}，"
                      f"分段BPM: {[f'{x:.2f}' for x in segment_bpms]}，"
                      f"标准差: {bpm_std:.2f}")
            if bpm_std < 1:
                stability = "稳定"
            elif bpm_std < 3:
                stability = "轻微波动"
            else:
                stability = "不稳定（变速）"

        return round(bpm, 2), stability, detail

    except Exception as e:
        return None, f"分析失败: {e}", ""


def make_mobile_output(df_regular, df_hard, online_month):
    """生成手游输出格式"""
    rows = []

    # 常规 35 首
    for i, (_, row) in enumerate(df_regular.iterrows(), 1):
        rows.append({
            '序号': i,
            '产品': '手游',
            '关卡类别': '常规',
            '歌名': row.iloc[2] if len(row) > 2 else "",
            '歌手': row.iloc[3] if len(row) > 3 else "",
            'BPM': row.iloc[5] if len(row) > 5 else "",
            '歌曲难度（制作后填）': '',
            '来源': '巨匠',
            '制作状态': '',
            '上线时间': online_month,
            '查重情况': '已通过查重',
            '备注': ''
        })

    # 高难 10 首
    for i, (_, row) in enumerate(df_hard.iterrows(), len(rows) + 1):
        rows.append({
            '序号': i,
            '产品': '手游',
            '关卡类别': '高难歌曲',
            '歌名': row.iloc[2] if len(row) > 2 else "",
            '歌手': row.iloc[3] if len(row) > 3 else "",
            'BPM': row.iloc[5] if len(row) > 5 else "",
            '歌曲难度（制作后填）': '',
            '来源': '巨匠',
            '制作状态': '',
            '上线时间': online_month,
            '查重情况': '已通过查重',
            '备注': ''
        })

    return pd.DataFrame(rows)


def make_pc_output(x51_df, x52_df, online_month):
    """生成端游输出格式"""
    x51_names = set(norm(x) for x in x51_df.iloc[:, 2])
    rows_x51 = []

    for i, (_, row) in enumerate(x51_df.iterrows(), 1):
        rows_x51.append({
            '序号': i,
            '产品': '炫舞1',
            '关卡类别': '常规',
            '歌名': row.iloc[2] if len(row) > 2 else "",
            '歌手': row.iloc[3] if len(row) > 3 else "",
            'BPM': row.iloc[5] if len(row) > 5 else "",
            '歌曲难度（制作后填）': '',
            '来源': '巨匠',
            '制作状态': '',
            '上线时间': online_month,
            '查重情况': '已通过查重',
            '备注': ''
        })
    df_x51_out = pd.DataFrame(rows_x51)

    rows_x52 = []
    for i, (_, row) in enumerate(x52_df.iterrows(), 1):
        is_reuse = norm(row.iloc[2]) in x51_names
        rows_x52.append({
            '序号': i,
            '产品': '炫舞2',
            '关卡类别': '常规',
            '歌名': row.iloc[2] if len(row) > 2 else "",
            '歌手': row.iloc[3] if len(row) > 3 else "",
            'BPM': row.iloc[5] if len(row) > 5 else "",
            '歌曲难度（制作后填）': '',
            '来源': '复用炫舞1' if is_reuse else '巨匠',
            '制作状态': '',
            '上线时间': online_month,
            '查重情况': '已通过查重',
            '备注': '复用炫舞1' if is_reuse else ''
        })
    df_x52_out = pd.DataFrame(rows_x52)

    return df_x51_out, df_x52_out


# ========== 主程序 ==========

def main():
    st.set_page_config(page_title="炫舞选歌工具", layout="wide")

    defaults = {
        'step': 1,
        'selected_songs': None,
        'bpm_check_results': None,
        '_game_type': None,
        '_exclude_japanese': True,
        '_dedup_set': set(),
        '_jujiang_df': None,
        '_online_month': '7月新歌',
        '_df_regular': None,
        '_df_hard': None,
        '_x51_df': None,
        '_x52_df': None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    with st.sidebar:
        st.title("选歌流程")
        steps = [
            "① 上传文件",
            "② 选择类型",
            "③ 开始选歌",
            "④ 查看结果",
            "⑤ 上传音频验证",
            "⑥ 查看验证结果",
            "⑦ 替换问题歌曲",
            "⑧ 最终输出"
        ]
        for i, s in enumerate(steps, 1):
            if i == st.session_state.step:
                st.markdown(f"**→ {s}**")
            elif i < st.session_state.step:
                st.markdown(f"~~{s}~~")
            else:
                st.markdown(f"  {s}")
        st.divider()
        if st.button("重置所有步骤", use_container_width=True, key="btn_reset"):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

    st.title("炫舞选歌工具 v0.7")

    # ========== 步骤1 ==========
    if st.session_state.step == 1:
        st.header("① 上传文件")
        st.caption("请上传巨匠音乐歌单.xlsx，可选上传查重表")

        jujiang_file = st.file_uploader(
            "上传巨匠音乐歌单.xlsx",
            type=["xlsx"],
            key="upload_jujiang"
        )

        dedup_file = st.file_uploader(
            "上传查重表（可选）",
            type=["xlsx"],
            key="upload_dedup"
        )

        if jujiang_file is not None:
            st.success(f"已上传：{jujiang_file.name}")
            try:
                jujiang_file.seek(0)
                xl = pd.ExcelFile(jujiang_file, engine='openpyxl')
                target_sheet = None
                for i, name in enumerate(xl.sheet_names):
                    if "选歌记录" in name and "神同步" not in name:
                        target_sheet = name
                        break
                if target_sheet:
                    jujiang_file.seek(0)
                    df = pd.read_excel(jujiang_file, sheet_name=target_sheet, engine='openpyxl', header=0)
                    st.session_state._jujiang_df = df
                    st.success(f"已加载「{target_sheet}」，共 {len(df)} 行")
                    with st.expander("查看列名"):
                        st.write(df.columns.tolist())
                else:
                    st.error("未找到「选歌记录」sheet")
            except Exception as e:
                st.error(f"读取文件失败: {e}")

        if dedup_file is not None:
            st.success(f"已上传查重表：{dedup_file.name}")
            dedup_set, msg = build_dedup_set_from_file(dedup_file)
            st.session_state._dedup_set = dedup_set
            st.info(msg)

        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.button("← 上一步", disabled=True, use_container_width=True, key="btn_prev_1")
        with col2:
            if st.button("下一步 →", type="primary", use_container_width=True, key="btn_next_1"):
                if st.session_state._jujiang_df is not None:
                    st.session_state.step = 2
                    st.rerun()
                else:
                    st.error("请先上传巨匠音乐歌单")

    # ========== 步骤2 ==========
    elif st.session_state.step == 2:
        st.header("② 选择游戏类型")

        game_type = st.selectbox(
            "选择游戏类型",
            ["手游选歌", "端游选歌（X51+X52）"],
            key="sel_game_type"
        )

        online_month = st.selectbox(
            "选择上线月份",
            MONTHS,
            index=6,
            key="sel_month"
        )
        st.session_state._online_month = online_month

        if game_type == "手游选歌":
            st.caption("手游：35常规（BPM≥75）+ 10高难（BPM≥140）= 45首")
            st.caption("常规优先 BPM>100，高难排除过度电子化、优先鼓点明确的风格")
        else:
            st.info("端游选歌：只选「常规」类型歌曲")
            st.caption("炫舞1：9首常规（BPM≥100）")
            st.caption("炫舞2：18首常规（BPM≥100），其中8-9首复用炫舞1")

        exclude_japanese = st.checkbox("排除日语歌", value=True, key="cb_exclude")

        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← 上一步", use_container_width=True, key="btn_prev_2"):
                st.session_state.step = 1
                st.rerun()
        with col2:
            if st.button("下一步 →", type="primary", use_container_width=True, key="btn_next_2"):
                st.session_state._game_type = game_type
                st.session_state._exclude_japanese = exclude_japanese
                st.session_state.step = 3
                st.rerun()

    # ========== 步骤3 ==========
    elif st.session_state.step == 3:
        st.header("③ 开始选歌")

        gt = st.session_state._game_type
        om = st.session_state._online_month
        ej = st.session_state._exclude_japanese
        ds = st.session_state._dedup_set
        df = st.session_state._jujiang_df

        if gt == "手游选歌":
            st.info(f"规则：35常规（BPM≥75）+ 10高难（BPM≥140）\n"
                     f"高难：排除过度电子化，优先鼓点明确/摇摆性强的风格\n"
                     f"排除日语歌：{ej}\n查重条目：{len(ds)} 条\n上线时间：{om}")
        else:
            st.info(f"规则：炫舞1=9首常规，炫舞2=18首常规（复用炫舞1）\n"
                     f"排除日语歌：{ej}\n查重条目：{len(ds)} 条\n上线时间：{om}")

        if st.button("开始选歌", type="primary", use_container_width=True, key="btn_start_select"):
            with st.spinner("正在选歌..."):
                if gt == "手游选歌":
                    df_reg, df_hard = select_mobile_songs(df, exclude_japanese=ej, dedup_set=ds)
                    st.session_state._df_regular = df_reg
                    st.session_state._df_hard = df_hard
                    st.session_state.selected_songs = {'regular': df_reg, 'hard': df_hard}
                    st.success(f"选歌完成！常规 {len(df_reg)} 首，高难 {len(df_hard)} 首")
                else:
                    x51_df, x52_df = select_pc_songs(df, exclude_japanese=ej, dedup_set=ds)
                    st.session_state._x51_df = x51_df
                    st.session_state._x52_df = x52_df
                    reuse = sum(1 for _, row in x52_df.iterrows()
                              if norm(row.iloc[2]) in set(norm(x) for x in x51_df.iloc[:, 2]))
                    st.session_state.selected_songs = {'x51': x51_df, 'x52': x52_df, 'reuse': reuse}
                    st.success(f"选歌完成！炫舞1: {len(x51_df)} 首，炫舞2: {len(x52_df)} 首，复用: {reuse} 首")

                st.session_state.step = 4
                st.rerun()

        st.divider()
        if st.button("← 上一步", use_container_width=True, key="btn_prev_3"):
            st.session_state.step = 2
            st.rerun()

    # ========== 步骤4 ==========
    elif st.session_state.step == 4:
        st.header("④ 查看选歌结果")

        if st.session_state.selected_songs is None:
            st.warning("暂无选歌结果，请先完成选歌")
        else:
            gt = st.session_state._game_type

            if gt == "手游选歌":
                data = st.session_state.selected_songs
                df_reg = data['regular']
                df_hard = data['hard']

                st.subheader(f"常规歌曲（{len(df_reg)} 首）")
                st.dataframe(df_reg, use_container_width=True)

                st.subheader(f"高难歌曲（{len(df_hard)} 首）")
                st.dataframe(df_hard, use_container_width=True)

                # 下载
                df_out = make_mobile_output(df_reg, df_hard, st.session_state._online_month)
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    df_out.to_excel(writer, sheet_name='选歌结果', index=False)
                output.seek(0)

                col_dl1, col_dl2 = st.columns(2)
                with col_dl1:
                    st.download_button(
                        "下载选歌结果（手游）.xlsx",
                        output,
                        file_name=f"手游选歌结果_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                        key="dl_mobile"
                    )
            else:
                data = st.session_state.selected_songs
                x51_df = data['x51']
                x52_df = data['x52']

                st.subheader(f"炫舞1（{len(x51_df)} 首常规）")
                st.dataframe(x51_df, use_container_width=True)

                st.subheader(f"炫舞2（{len(x52_df)} 首常规，复用 {data['reuse']} 首）")
                st.dataframe(x52_df, use_container_width=True)

                x51_out, x52_out = make_pc_output(x51_df, x52_df, st.session_state._online_month)
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    x51_out.to_excel(writer, sheet_name='炫舞1', index=False)
                    x52_out.to_excel(writer, sheet_name='炫舞2', index=False)
                output.seek(0)

                col_dl1, col_dl2 = st.columns(2)
                with col_dl1:
                    st.download_button(
                        "下载选歌结果（端游）.xlsx",
                        output,
                        file_name=f"端游选歌结果_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                        key="dl_pc"
                    )

        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← 上一步", use_container_width=True, key="btn_prev_4"):
                st.session_state.step = 3
                st.rerun()
        with col2:
            if st.button("下一步 →", type="primary", use_container_width=True, key="btn_next_4"):
                st.session_state.step = 5
                st.rerun()

    # ========== 步骤5 ==========
    elif st.session_state.step == 5:
        st.header("⑤ 上传音频文件进行BPM验证")
        st.caption("支持 MP3、WAV、FLAC、OGG 等常见音频格式")
        st.caption("BPM 检测结果将精确到小数点后两位")

        if not HAS_LIBROSA:
            st.warning("音频分析功能需要安装 librosa 库。当前为云端版本，暂不支持音频检测。")
            st.info("建议：在本地版本中使用此功能（双击 start.bat 启动）")
        else:
            audio_files = st.file_uploader(
                "上传音频文件（可多选）",
                type=["mp3", "wav", "flac", "ogg", "m4a", "aac"],
                accept_multiple_files=True,
                key="upload_audio"
            )

            if audio_files and st.button("开始BPM检测", type="primary", use_container_width=True, key="btn_start_bpm"):
                results = []
                progress = st.progress(0, text="正在检测BPM...")
                for i, audio_file in enumerate(audio_files):
                    st.write(f"正在检测：{audio_file.name}")
                    audio_bytes = audio_file.read()
                    bpm, stability, detail = analyze_bpm(audio_bytes, audio_file.name)
                    results.append({
                        '文件名': audio_file.name,
                        '检测BPM': f"{bpm:.2f}" if bpm else "失败",
                        '稳定性': stability,
                        '详情': detail
                    })
                    progress.progress((i + 1) / len(audio_files), text=f"已完成 {i+1}/{len(audio_files)}")

                progress.empty()
                st.session_state.bpm_check_results = results
                st.success("BPM检测完成！")
                st.session_state.step = 6
                st.rerun()

        st.divider()
        if st.button("← 上一步", use_container_width=True, key="btn_prev_5"):
            st.session_state.step = 4
            st.rerun()

    # ========== 步骤6 ==========
    elif st.session_state.step == 6:
        st.header("⑥ BPM验证结果")

        if st.session_state.bpm_check_results is None:
            st.warning("暂无验证结果，请先上传音频文件")
        else:
            results = st.session_state.bpm_check_results
            df_results = pd.DataFrame(results)
            st.dataframe(df_results, use_container_width=True)

            stable = sum(1 for r in results if "稳定" in r['稳定性'] and "波动" not in r['稳定性'] and "变速" not in r['稳定性'])
            suspicious = sum(1 for r in results if "波动" in r['稳定性'] or "变速" in r['稳定性'] or "失败" in r['稳定性'])

            col1, col2, col3 = st.columns(3)
            col1.metric("总计", len(results))
            col2.metric("稳定", stable)
            col3.metric("有问题", suspicious)

            output = io.BytesIO()
            df_results.to_excel(output, index=False, engine='xlsxwriter')
            output.seek(0)
            st.download_button(
                "下载验证结果.xlsx",
                output,
                file_name=f"BPM验证结果_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                key="dl_bpm"
            )

        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← 上一步", use_container_width=True, key="btn_prev_6"):
                st.session_state.step = 5
                st.rerun()
        with col2:
            if st.button("下一步 →", type="primary", use_container_width=True, key="btn_next_6"):
                st.session_state.step = 7
                st.rerun()

    # ========== 步骤7 ==========
    elif st.session_state.step == 7:
        st.header("⑦ 替换问题歌曲")
        st.info("此功能正在开发中，当前版本请手动替换问题歌曲")

        if st.session_state.bpm_check_results:
            problem_songs = [r for r in st.session_state.bpm_check_results
                             if "波动" in r['稳定性'] or "变速" in r['稳定性'] or "失败" in r['稳定性']]
            if problem_songs:
                st.subheader("检测到以下可疑/不稳定歌曲：")
                for song in problem_songs:
                    st.write(f"- {song['文件名']}: BPM={song['检测BPM']}, {song['稳定性']}")

        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← 上一步", use_container_width=True, key="btn_prev_7"):
                st.session_state.step = 6
                st.rerun()
        with col2:
            if st.button("下一步 →", type="primary", use_container_width=True, key="btn_next_7"):
                st.session_state.step = 8
                st.rerun()

    # ========== 步骤8 ==========
    elif st.session_state.step == 8:
        st.header("⑧ 最终输出")
        st.success("选歌流程已完成！请下载最终结果。")

        gt = st.session_state._game_type
        om = st.session_state._online_month

        if gt == "手游选歌":
            df_reg = st.session_state._df_regular
            df_hard = st.session_state._df_hard
            df_out = make_mobile_output(df_reg, df_hard, om)

            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df_out.to_excel(writer, sheet_name='最终歌单', index=False)
            output.seek(0)

            st.download_button(
                "下载最终歌单（手游）.xlsx",
                output,
                file_name=f"手游最终歌单_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                key="dl_final_mobile"
            )
        else:
            x51_df = st.session_state._x51_df
            x52_df = st.session_state._x52_df
            x51_out, x52_out = make_pc_output(x51_df, x52_df, om)

            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                x51_out.to_excel(writer, sheet_name='炫舞1', index=False)
                x52_out.to_excel(writer, sheet_name='炫舞2', index=False)
            output.seek(0)

            st.download_button(
                "下载最终歌单（端游）.xlsx",
                output,
                file_name=f"端游最终歌单_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                key="dl_final_pc"
            )

        st.divider()
        if st.button("← 上一步", use_container_width=True, key="btn_prev_8"):
            st.session_state.step = 7
            st.rerun()


if __name__ == "__main__":
    main()
