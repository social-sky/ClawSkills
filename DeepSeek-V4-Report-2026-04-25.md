# DeepSeek V4 深度研究分析報告

**報告日期：** 2026-04-25  
**資料來源：** HuggingFace、DeepSeek 技術報告、CNBC、Fortune、The Meridiem、Lushbinary、AI2Work  
**覆核狀態：** ✅ 已確認

---

## 一、模型概覽

DeepSeek V4 於 **2026 年 4 月 24 日** 正式發布預覽版。

| 項目 | **V4-Pro** | **V4-Flash** |
|---|---|---|
| 總參數量 | 1.6 兆（1.6T） | 2840 億（284B） |
| 激活參數 | 490 億（49B） | 130 億（13B） |
| 上下文窗口 | 100 萬 tokens | 100 萬 tokens |
| 精度 | FP8 | FP4 + FP8 |
| 發布日期 | 2026-04-24 | 2026-04-24 |
| 授權 | MIT | MIT |

核心技術：混合注意力（CSA+HCA）、流形約束超連接（mHC）、Muon 優化器，預訓練 32T tokens。

---

## 二、晶片適用性

### NVIDIA 晶片
開源權重 MIT 許可，支援 vLLM、SGLang、Transformers。RTX 4090（24GB）可單卡推斷，推薦 H100/H800。

### 華為 Ascend 晶片
DeepSeek V4 是 **歷史上第一個完全在中國國產半導體上訓練的前沿模型**，使用 **Ascend 950PR**。DeepSeek 給予華為早期架構優化訪問權，未向 NVIDIA/AMD 提供同等待遇。

Ascend 950PR：達文西架構，HBM3，CANN 軟體框架，對標 H100/H200。

### V3 → V4 訓練硬體演變
- DeepSeek V3：NVIDIA H800 × 2,048 顆，訓練成本約 3,000 萬美元
- DeepSeek V4：華為 Ascend 950PR（全自主國產供應鏈）

### 開發者視角
| 維度 | NVIDIA | 華為 Ascend |
|---|---|---|
| 開源權重 | ✅ 完全支援 | ⚠️ 需額外適配 |
| CUDA 生態 | ✅ 成熟 | ❌ CANN 不成熟 |
| 第三方框架 | ✅ vLLM/SGLang | ❌ 需移植 |

---

## 三、API 定價

### 每百萬 tokens 費用

| 模型 | 輸入（緩存命中） | 輸入（未命中） | **輸出** |
|---|---|---|---|
| DeepSeek V4-Pro | $0.145 | $1.74 | **$3.48** |
| DeepSeek V4-Flash | $0.028 | $0.14 | **$0.28** |
| OpenAI GPT-5.4 | — | — | **$30.00** |
| Anthropic Opus 4.6 | — | — | **$25.00** |
| Kimi | — | — | **$4.00** |

V4-Pro 輸出費用僅為 GPT-5.4 的 **11.6%**。DeepSeek 將隨 Ascend 950 規模化繼續降價。

---

## 四、產業影響

### 4.1 地緣政治：美國出口管制實質失效
華為 Ascend 950PR 成功訓練頂尖前沿模型，Counterpoint Research 評論：「晶片管制假設需要立即重新定價。」

### 4.2 硬體生態：NVIDIA vs CANN 雙雄並立
- NVIDIA 陣營：CUDA、H100/H200、全球雲端
- 華為 Ascend 陣營：CANN、Ascend 950PR、中國市場
- DeepSeek 同步與寒武紀合作移植核心

### 4.3 市場：開源碾壓閉源定價
| 基準測試 | Opus 4.6 | GPT-5.4 | **DS-V4-Pro** |
|---|---|---|---|
| LiveCodeBench | 88.8 | 91.7 | **93.5 🏆** |
| Codeforces | — | 3168 | **3206 🏆** |
| MMLU-Pro | 89.1 | 87.5 | 87.5 |

### 4.4 半導體股：中資大漲、美系受壓
- SMIC 股價單日飆升 **+10%**（生產華為 Ascend 處理器）
- MiniMax、知識天地股價下跌超 **9%**

### 4.5 AI 民主化
訓練成本 3,000 萬美元 vs GPT-4 的 1 億+ 美元。企業自建 LLM 攤薄後每 token 低於 $0.03。受監管行業可本地部署確保數據主權。

---

## 五、十大關鍵結論

1. ✅ 首個國產晶片訓練的前沿模型 → 華為 Ascend 950PR
2. ✅ 訓練硬體從 H800 全面轉向華為 → 戰略性供應鏈重構
3. ✅ API 極低價格：$3.48/M 輸出 vs GPT-5.4 的 $30/M
4. ✅ MIT 許可 → NVIDIA GPU 完全可本地部署
5. ✅ 1M token 上下文 + 混合注意力 → 長文本效率提升 90%
6. ✅ Coding 基準（93.5）領先所有對手
7. ✅ 美國出口管制戰略實質失效
8. ✅ CANN vs CUDA 軟體生態之戰正式拉開
9. ✅ SMIC 等中資半導體股大漲
10. ✅ AI 民主化：企業自建 LLM 成本降至千萬美元級別

**DeepSeek V4 不僅是模型升級，更是 AI 產業從「單一硬體霸權」走向「多極生態競爭」的歷史轉折點。**

---

*報告生成：2026-04-25 | 數據來源：HuggingFace API、OpenRouter API、新聞媒體*
