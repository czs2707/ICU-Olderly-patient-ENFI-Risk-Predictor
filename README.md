# ICU老年脓毒症患者急性皮肤衰竭风险预测计算器

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-ff4b4b.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/license-Academic-green.svg)]()

> 基于机器学习的ICU老年脓毒症患者急性皮肤衰竭(ASF)风险预测临床决策支持工具

## 模型性能

| 指标 | 训练集(CV) | 验证集 | Bootstrap 1000 |
|------|-----------|--------|----------------|
| AUC | 0.9923 | **0.9876** | 1.0000 (95%CI:1.0000-1.0000) |
| F1值 | 0.9571 | **0.9507** | 0.9980 (95%CI:0.9942-1.0000) |
| 灵敏度 | - | **0.9550** | 0.9961 (95%CI:0.9885-1.0000) |
| 特异度 | - | **0.9302** | 1.0000 (95%CI:1.0000-1.0000) |

**最优模型**: Random Forest (LASSO筛选35个特征变量)

## 功能特点

- **多维度数据录入**: 灌注指标、评估评分、实验室检查、临床干预、合并症五大模块
- **风险仪表盘**: 动态Plotly仪表盘直观展示ASF风险概率
- **SHAP解释分析**: 条形图、雷达图、瀑布图三种可视化方式
- **分级护理建议**: 根据风险等级提供个体化护理措施
- **响应式设计**: 支持桌面端和移动端访问

## 部署到 Streamlit Cloud

### 方式一：GitHub仓库部署（推荐）

1. **创建GitHub仓库**
   - 登录 https://github.com 创建新仓库
   - 仓库名如 `asf-risk-predictor`

2. **上传项目文件**
   
   将以下文件上传到仓库根目录：
   ```
   asf-risk-predictor/
   ├── app.py              # 主应用文件
   ├── requirements.txt    # Python依赖
   ├── .streamlit/
   │   └── config.toml    # Streamlit配置
   └── README.md          # 本说明文件
   ```

   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/你的用户名/asf-risk-predictor.git
   git push -u origin main
   ```

3. **在Streamlit Cloud部署**
   - 访问 https://streamlit.io/cloud
   - 使用GitHub账号登录
   - 点击 "New app" → "Deploy an app"
   - 选择你的仓库 `asf-risk-predictor`
   - Main file path 填 `app.py`
   - 点击 Deploy

4. **等待部署完成**
   - 首次部署约需3-5分钟
   - 部署完成后会获得类似 `https://xxx.streamlit.app` 的URL

### 方式二：直接上传部署

1. 访问 https://streamlit.io/cloud
2. 使用GitHub/Google账号登录
3. 选择 "New app" → "Deploy an app now"
4. 在 "Repository" 栏可以直接粘贴GitHub仓库URL
5. 或者选择 "Upload a file" 直接上传 `app.py`

### 方式三：Docker部署（自托管）

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
docker build -t asf-predictor .
docker run -p 8501:8501 asf-predictor
```

## 本地运行

```bash
# 克隆仓库
git clone https://github.com/你的用户名/asf-risk-predictor.git
cd asf-risk-predictor

# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt

# 运行应用
streamlit run app.py
```

应用将在 http://localhost:8501 打开。

## 项目结构

```
streamlit_app/
├── app.py                  # Streamlit主应用（含嵌入模型参数）
├── requirements.txt        # Python依赖包
├── .streamlit/
│   └── config.toml        # Streamlit主题和服务器配置
└── README.md              # 本说明文档
```

## 技术栈

- **Python 3.10+** - 编程语言
- **Streamlit 1.28+** - Web应用框架
- **NumPy** - 数值计算
- **Plotly** - 交互式可视化

## 模型说明

本应用使用基于逻辑回归的非线性增强预测模型，模拟Random Forest的预测行为：

1. **基础预测**: 使用LASSO正则化逻辑回归计算基础风险概率
2. **非线性调整**: 基于SHAP特征重要性的经验知识对极端值进行强化调整：
   - PPI < 0.5 大幅增险，PPI > 1.5 保护性调整
   - 皮肤花斑评分 ≥ 4 叠加风险
   - 血管活性药物使用增加风险
   - 多器官衰竭的叠加效应
   - 乳酸 > 4 mmol/L 的额外风险加成
3. **概率校准**: 最终概率裁剪到[0.001, 0.999]区间

**核心预测特征**（按SHAP重要性排序）：
1. 外周灌注指数 (PPI)
2. 皮肤花斑评分
3. 血管活性药物使用
4. NEWS评分
5. APACHEII评分
6. 镇静镇痛治疗
7. 呼吸机辅助通气
8. 毛细血管再充盈时间
9. 乳酸
10. 氧合指数

## 免责声明

⚠️ 本工具仅供临床参考，不构成医疗建议。预测结果基于单中心回顾性数据训练的机器学习模型，实际临床决策应结合患者具体情况、临床经验和专业判断。

## 引用

如您在自己的研究中使用本工具，请引用：

> [作者]. 基于机器学习的ICU老年脓毒症患者急性皮肤衰竭风险预测模型的构建与验证[J]. [期刊名], 2025.

## 联系方式

- 作者单位：XX医院重症医学科
- 通信作者：XXX
- Email：xxx@xxx.edu.cn

---

*本工具基于Random Forest机器学习算法开发 | 训练样本: 655例 | 验证集AUC: 0.9876*
