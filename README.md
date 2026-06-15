# 炫舞选歌工具

AI 辅助选歌工具，用于炫舞手游/端游每月新歌筛选。

## 功能

- 上传巨匠音乐歌单，自动按规则筛选歌曲
- 支持手游（常规35首 + 高难10首）和端游（X51/X52）选歌
- BPM 稳定性检测（基于 librosa 音频分析）
- 排除日语歌、查重过滤
- 输出格式对齐内部月度汇总表

## 本地运行

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 在线使用

访问：https://xuanywu-selector.streamlit.app（部署后更新）

## 技术栈

- Streamlit — 前端界面
- pandas + calamine — Excel 读取
- librosa + soundfile — 音频 BPM 检测
- xlsxwriter — Excel 输出
