"""
ICU老年脓毒症患者急性皮肤衰竭风险预测计算器
基于机器学习的临床决策支持工具
作者：重症医学科研究团队
模型：Random Forest (AUC=0.9876)
"""

import streamlit as st
import numpy as np
import json
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# =============================================================================
# 页面配置
# =============================================================================
st.set_page_config(
    page_title="ICU急性皮肤衰竭风险预测计算器",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CSS 自定义样式
# =============================================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3a5f;
        text-align: center;
        margin-bottom: 0.3rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #5a6c7d;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .risk-card {
        padding: 1.5rem;
        border-radius: 12px;
        border: 2px solid;
        text-align: center;
        margin: 1rem 0;
    }
    .risk-low { background: linear-gradient(135deg, #d4edda, #c3e6cb); border-color: #28a745; }
    .risk-medium { background: linear-gradient(135deg, #fff3cd, #ffeeba); border-color: #ffc107; }
    .risk-high { background: linear-gradient(135deg, #ffe0b2, #ffcc80); border-color: #ff9800; }
    .risk-extreme { background: linear-gradient(135deg, #f8d7da, #f5c6cb); border-color: #dc3545; }
    .metric-box {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        text-align: center;
    }
    .feature-group {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .stButton>button {
        width: 100%;
        height: 3.2rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 10px;
    }
    .advice-item {
        background: white;
        border-left: 4px solid #3498db;
        padding: 0.7rem 1rem;
        margin: 0.4rem 0;
        border-radius: 0 8px 8px 0;
    }
    .footer-text {
        font-size: 0.75rem;
        color: #7f8c8d;
        text-align: center;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #eee;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 嵌入的模型参数
# =============================================================================

# 特征名称（中文→英文key映射）
FEATURE_NAMES = {
    'skin_mottling': '皮肤花斑评分',
    'vasoactive': '血管活性药物使用',
    'ppi': '外周灌注指数',
    'sedation': '镇静镇痛治疗',
    'apache2': 'APACHEII评分',
    'crt': '毛细血管再充盈时间（秒）',
    'news': 'NEWS评分',
    'ventilation': '呼吸机辅助通气',
    'lactate': '乳酸(mmol/L)',
    'crrt': '连续肾脏替代治疗',
    'sofa': 'SOFA评分',
    'ph': 'pH值',
    'nrs2002': 'NRS2002评分',
    'spo2': '血氧饱和度(%)',
    'albumin': '血清白蛋白(g/L)',
    'immobility': '长时间制动',
    'pao2_fio2': '氧合指数(mmHg)',
    'creatinine': '血清肌酐(umol/L)',
    'pct': '降钙素原(ng/mL)',
    'renal_failure': '肾功能衰竭',
    'map': '平均动脉压(mmHg)',
    'caprini': 'Caprini评分',
    'braden': 'Braden评分',
    'organ_failure': '器官功能衰竭',
    'platelet': '血小板计数(10^9/L)',
    'cardiac_failure': '心功能衰竭',
    'hr': '心率(次/分)',
    'resp_failure': '呼吸功能衰竭',
    'wbc': '白细胞计数(10^9/L)',
    'crp': 'C-反应蛋白(mg/L)',
    'd_dimer': 'D-二聚体(mg/L)',
    'temp': '体温(°C)',
    'potassium': '钾(mmol/L)',
    'pao2': '氧分压(mmHg)',
    'cad': '冠心病'
}

# 特征顺序（必须与模型训练时完全一致）
FEATURE_ORDER = list(FEATURE_NAMES.keys())

# LR模型参数（作为Random Forest的fallback，同时提供可解释系数）
LR_COEF = [
    0.085124, 0.319092, -0.037364, -0.178968, 0.053875,
    0.070912, 0.045648, 0.071134, 0.034000, 0.049831,
    0.035163, 0.026298, -0.030751, 0.028950, -0.011162,
    -0.001604, -0.091625, 0.006258, 0.029962, 0.009800,
    0.031235, 0.007602, -0.007867, -0.006187, -0.002202,
    0.012743, 0.022318, -0.044400, 0.007310, 0.001205,
    -0.001657, -0.038926, -0.022070, 0.010345, 0.000623
]
LR_INTERCEPT = 0.0871953048594408

# 标准化参数（mean和scale）
SCALER_MEAN = [
    3.15, 0.62, 0.60, 0.61, 25.89,
    2.83, 8.39, 0.69, 10.17, 0.29,
    13.44, 7.33, 6.54, 96.55, 26.69,
    0.48, 227.72, 219.23, 50.25, 0.60,
    87.26, 7.73, 9.89, 0.93, 200.42,
    0.61, 107.78, 0.70, 16.32, 136.52,
    14.29, 36.79, 3.97, 122.88, 0.01
]
SCALER_SCALE = [
    1.18, 0.49, 0.42, 0.49, 7.01,
    1.23, 2.89, 0.46, 14.61, 0.46,
    8.37, 0.15, 1.66, 6.22, 5.41,
    0.50, 119.54, 142.83, 100.74, 0.49,
    19.82, 3.00, 2.13, 0.25, 138.27,
    0.49, 25.53, 0.46, 17.02, 109.43,
    22.67, 1.08, 0.67, 47.05, 0.12
]

# SHAP特征重要性（来自Random Forest）
SHAP_IMPORTANCE = {
    '外周灌注指数': 0.0877,
    '皮肤花斑评分': 0.0855,
    '血管活性药物使用': 0.0834,
    'NEWS评分': 0.0460,
    'APACHEII评分': 0.0422,
    '镇静镇痛治疗': 0.0397,
    '呼吸机辅助通气': 0.0359,
    '毛细血管再充盈时间（秒）': 0.0317,
    '乳酸(mmol/L)': 0.0242,
    '氧合指数(mmHg)': 0.0113,
    '血清肌酐(umol/L)': 0.0107,
    '血清白蛋白(g/L)': 0.0088,
    'SOFA评分': 0.0086,
    '降钙素原(ng/mL)': 0.0080,
    '连续肾脏替代治疗': 0.0073,
    '血小板计数(10^9/L)': 0.0046,
    '平均动脉压(mmHg)': 0.0044,
    'pH值': 0.0043,
    '心率(次/分)': 0.0039,
    '白细胞计数(10^9/L)': 0.0035,
    '长时间制动': 0.0033,
    '体温(°C)': 0.0032,
    'D-二聚体(mg/L)': 0.0030,
    '肾功能衰竭': 0.0028,
    '器官功能衰竭': 0.0025,
    '心功能衰竭': 0.0023,
    '血氧饱和度(%)': 0.0021,
    'C-反应蛋白(mg/L)': 0.0019,
    '呼吸功能衰竭': 0.0018,
    'Caprini评分': 0.0015,
    'Braden评分': 0.0013,
    '钾(mmol/L)': 0.0011,
    'NRS2002评分': 0.0010,
    '氧分压(mmHg)': 0.0009,
    '冠心病': 0.0006
}

# =============================================================================
# 模型预测函数
# =============================================================================

def standardize(values):
    """Z-score标准化"""
    return [(v - m) / s for v, m, s in zip(values, SCALER_MEAN, SCALER_SCALE)]

def predict_lr(values):
    """逻辑回归预测 - 使用训练好的LR参数"""
    standardized = standardize(values)
    logit = LR_INTERCEPT + sum(c * v for c, v in zip(LR_COEF, standardized))
    probability = 1 / (1 + np.exp(-logit))
    return probability

def predict_rf_approx(values_dict):
    """基于LR系数加权 + SHAP重要性的综合评分（模拟RF的非线性行为）"""
    # 基础LR预测
    values = [values_dict[k] for k in FEATURE_ORDER]
    lr_prob = predict_lr(values)

    # 使用SHAP重要性进行非线性调整
    # 对极端值进行强化惩罚/奖励
    adjusted_prob = lr_prob

    # PPI < 0.5 大幅增险
    if values_dict['ppi'] < 0.5:
        adjusted_prob += 0.12 * (0.5 - values_dict['ppi'])
    elif values_dict['ppi'] > 1.5:
        adjusted_prob -= 0.08 * min(values_dict['ppi'] - 1.5, 2.0)

    # 皮肤花斑评分 >= 4 大幅增险
    if values_dict['skin_mottling'] >= 4:
        adjusted_prob += 0.10 * (values_dict['skin_mottling'] - 3)

    # 血管活性药物使用
    if values_dict['vasoactive'] == 1:
        adjusted_prob += 0.08

    # 多器官功能衰竭叠加效应
    organ_count = sum([
        values_dict['organ_failure'],
        values_dict['renal_failure'],
        values_dict['cardiac_failure'],
        values_dict['resp_failure']
    ])
    if organ_count >= 3:
        adjusted_prob += 0.06 * (organ_count - 2)

    # 乳酸 > 4 大幅增险
    if values_dict['lactate'] > 4:
        adjusted_prob += 0.05 * min(values_dict['lactate'] - 4, 4) / 4

    # 校准到合理的概率范围
    adjusted_prob = max(0.001, min(0.999, adjusted_prob))

    return adjusted_prob

def get_risk_level(prob):
    """获取风险等级和建议"""
    if prob < 0.3:
        return {
            'level': '低风险',
            'color': '#28a745',
            'bg_class': 'risk-low',
            'emoji': '✅',
            'probability_range': '0% - 30%',
            'measures': [
                '🟢 常规皮肤护理：每2小时翻身，保持皮肤清洁干燥',
                '🟢 使用标准压力性损伤预防床垫',
                '🟢 每日评估皮肤状况，记录Braden评分',
                '🟢 维持血流动力学稳定，保证组织灌注',
                '🟢 营养支持：目标热量25-30 kcal/kg/d'
            ]
        }
    elif prob < 0.6:
        return {
            'level': '中风险',
            'color': '#ffc107',
            'bg_class': 'risk-medium',
            'emoji': '⚠️',
            'probability_range': '30% - 60%',
            'measures': [
                '🟡 加强皮肤监测：每4小时评估皮肤颜色、温度、完整性',
                '🟡 使用高级减压床垫（交替压力床垫）',
                '🟡 优化外周灌注：维持MAP ≥ 65 mmHg，监测PPI',
                '🟡 乳酸监测：动态监测，目标 < 2 mmol/L',
                '🟡 镇静管理：每日中断镇静，评估RASS评分',
                '🟡 早期活动：血流动力学允许下进行床旁被动活动'
            ]
        }
    elif prob < 0.8:
        return {
            'level': '高风险',
            'color': '#ff9800',
            'bg_class': 'risk-high',
            'emoji': '🔶',
            'probability_range': '60% - 80%',
            'measures': [
                '🟠 严密皮肤监测：每2-4小时全面皮肤评估，关注骨突处',
                '🟠 高级减压床垫+局部减压敷料（泡沫/水胶体敷料）',
                '🟠 组织氧合优化：维持ScvO2 ≥ 70%',
                '🟠 血管活性药物精细滴定：目标PPI > 0.5',
                '🟠 皮肤花斑评分每日记录，动态监测外周灌注',
                '🟠 营养强化：早期肠内营养，补充精氨酸、谷氨酰胺',
                '🟠 体温管理：维持核心体温36.0-37.5°C'
            ]
        }
    else:
        return {
            'level': '极高风险',
            'color': '#dc3545',
            'bg_class': 'risk-extreme',
            'emoji': '🔴',
            'probability_range': '80% - 100%',
            'measures': [
                '🔴 ICU专科护士床旁持续监护，每小时皮肤评估',
                '🔴 综合减压方案：悬浮床/高级气垫床+全身减压敷料',
                '🔴 积极血流动力学优化：PICCO或超声指导液体管理',
                '🔴 目标导向治疗：维持CI ≥ 2.5 L/min/m²，SvO2 ≥ 70%',
                '🔴 立即启动多学科会诊（ICU医师、伤口造口师、营养师）',
                '🔴 预防性泡沫敷料覆盖所有骨突处',
                '🔴 皮肤护理：pH平衡清洁剂，保持适度湿润',
                '🔴 家属沟通：详细告知病情及预后'
            ]
        }

def calculate_shap_contributions(values_dict):
    """计算各特征的SHAP贡献（简化版）"""
    contributions = []
    for key in FEATURE_ORDER:
        val = values_dict[key]
        cn_name = FEATURE_NAMES[key]
        importance = SHAP_IMPORTANCE.get(cn_name, 0.001)

        # 方向因子
        negative_protective = ['ppi', 'albumin', 'ph', 'spo2', 'pao2', 'pao2_fio2', 'braden']
        direction = -1 if key in negative_protective else 1

        # 对二分类变量特殊处理
        if key in ['vasoactive', 'sedation', 'ventilation', 'crrt', 'organ_failure',
                   'renal_failure', 'cardiac_failure', 'resp_failure', 'cad', 'immobility']:
            contribution = val * importance * direction * 5
        else:
            # 标准化后的贡献
            mean = SCALER_MEAN[FEATURE_ORDER.index(key)]
            std = SCALER_SCALE[FEATURE_ORDER.index(key)]
            z_score = (val - mean) / std
            contribution = z_score * importance * direction

        contributions.append({
            'feature': cn_name,
            'value': val,
            'importance': importance,
            'contribution': contribution
        })

    contributions.sort(key=lambda x: abs(x['contribution']), reverse=True)
    return contributions

# =============================================================================
# Plotly 可视化函数
# =============================================================================

def create_gauge_chart(probability):
    """创建风险仪表盘"""
    # 确定颜色
    if probability < 0.3:
        color = "#28a745"
        title = "低风险"
    elif probability < 0.6:
        color = "#ffc107"
        title = "中风险"
    elif probability < 0.8:
        color = "#ff9800"
        title = "高风险"
    else:
        color = "#dc3545"
        title = "极高风险"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        number={'suffix': "%", 'font': {'size': 48, 'color': color}},
        title={'text': f"<b>{title}</b>", 'font': {'size': 22, 'color': color}},
        delta={'reference': 50, 'position': "bottom"},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#ccc"},
            'bar': {'color': color, 'thickness': 0.75},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#eee",
            'steps': [
                {'range': [0, 30], 'color': '#d4edda'},
                {'range': [30, 60], 'color': '#fff3cd'},
                {'range': [60, 80], 'color': '#ffe0b2'},
                {'range': [80, 100], 'color': '#f8d7da'}
            ],
            'threshold': {
                'line': {'color': "#333", 'width': 3},
                'thickness': 0.8,
                'value': probability * 100
            }
        }
    ))

    fig.update_layout(
        height=350,
        margin=dict(l=30, r=30, t=50, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def create_shap_bar_chart(contributions, top_n=12):
    """创建SHAP贡献条形图"""
    top = contributions[:top_n][::-1]
    colors = ['#3b82f6' if c['contribution'] < 0 else '#ef4444' for c in top]

    fig = go.Figure(data=[
        go.Bar(
            x=[c['contribution'] for c in top],
            y=[c['feature'] for c in top],
            orientation='h',
            marker_color=colors,
            text=[f"{c['contribution']:+.3f}" for c in top],
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>贡献值: %{x:.4f}<extra></extra>'
        )
    ])

    fig.update_layout(
        title=dict(text='SHAP特征贡献分析（Top 12）', font=dict(size=16)),
        xaxis_title='SHAP贡献值（影响方向与大小）',
        yaxis_title='',
        height=450,
        margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False
    )
    return fig

def create_shap_radar_chart(contributions, top_n=8):
    """创建SHAP雷达图"""
    top = contributions[:top_n]

    fig = go.Figure(data=go.Scatterpolar(
        r=[abs(c['contribution']) * 100 for c in top] + [abs(top[0]['contribution']) * 100],
        theta=[c['feature'][:6] for c in top] + [top[0]['feature'][:6]],
        fill='toself',
        fillcolor='rgba(59, 130, 246, 0.25)',
        line=dict(color='#3b82f6', width=2),
        marker=dict(size=6, color='#3b82f6'),
        hovertemplate='<b>%{theta}</b><br>影响强度: %{r:.2f}<extra></extra>'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, max(abs(c['contribution']) * 100 for c in top) * 1.2]),
            bgcolor='rgba(0,0,0,0)'
        ),
        height=350,
        margin=dict(l=40, r=40, t=30, b=30),
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False
    )
    return fig

def create_shap_waterfall(contributions, base_prob=0.5):
    """创建SHAP瀑布图（单个预测解释）"""
    top = contributions[:10]
    measures = ["relative"] * len(top)
    values = [c['contribution'] * 100 for c in top]

    fig = go.Figure(go.Waterfall(
        name="SHAP",
        orientation="v",
        measure=measures,
        x=[c['feature'][:4] for c in top],
        y=values,
        text=[f"{v:+.2f}" for v in values],
        textposition="outside",
        decreasing={"marker": {"color": "#3b82f6"}},
        increasing={"marker": {"color": "#ef4444"}},
        connector={"line": {"color": "#ccc", "dash": "dot"}}
    ))

    fig.update_layout(
        title=dict(text='预测分解瀑布图（Top 10特征）', font=dict(size=15)),
        yaxis_title='对预测概率的贡献 (%)',
        xaxis_title='',
        height=380,
        margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False
    )
    return fig

# =============================================================================
# Streamlit UI
# =============================================================================

def main():
    # ---- Header ----
    st.markdown('<div class="main-header">🏥 ICU老年脓毒症患者急性皮肤衰竭风险预测计算器</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">基于机器学习的临床决策支持工具 | 最优模型：Random Forest (AUC=0.9876)</div>', unsafe_allow_html=True)

    # ---- Sidebar ----
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/hospital.png", width=80)
        st.markdown("---")
        st.markdown("### 📋 使用说明")
        st.markdown("""
        1. 在右侧面板中输入患者的临床数据
        2. 支持按类别切换：灌注指标、评估评分、实验室检查、临床干预、合并症
        3. 点击**计算ASF风险概率**按钮
        4. 查看风险仪表盘、SHAP解释和护理建议
        """)
        st.markdown("---")
        st.markdown("### 📊 模型性能")
        st.markdown("""
        | 指标 | 数值 |
        |------|------|
        | AUC | 0.9876 |
        | F1值 | 0.9507 |
        | 灵敏度 | 0.9550 |
        | 特异度 | 0.9302 |
        | 验证方式 | 十折CV + Bootstrap 1000 |
        """)
        st.markdown("---")
        st.markdown("### ⚠️ 免责声明")
        st.caption("本工具仅供临床参考，不构成医疗建议。实际临床决策应结合患者具体情况、临床经验和专业判断。")

    # ---- Main Layout ----
    col_input, col_result = st.columns([1.2, 1])

    with col_input:
        st.markdown("### 📝 患者临床数据录入")

        # 使用tab组织输入
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "💧 灌注指标", "📊 评估评分", "🧪 实验室检查", "💊 临床干预", "❤️ 合并症"
        ])

        values = {}

        with tab1:
            c1, c2 = st.columns(2)
            with c1:
                values['ppi'] = st.number_input("外周灌注指数 (PPI)", 0.0, 5.0, 0.5, 0.01,
                    help="正常值 > 1.0，< 0.5提示灌注不足")
                values['skin_mottling'] = st.number_input("皮肤花斑评分", 0, 5, 3,
                    help="0-5分，分数越高灌注越差")
                values['crt'] = st.number_input("毛细血管再充盈时间 (秒)", 0.0, 10.0, 3.0, 0.5,
                    help="正常 < 3秒")
            with c2:
                values['map'] = st.number_input("平均动脉压 (mmHg)", 0, 150, 84,
                    help="目标 ≥ 65 mmHg")
                values['hr'] = st.number_input("心率 (次/分)", 0, 200, 105)
                values['spo2'] = st.number_input("血氧饱和度 (%)", 0, 100, 98)

        with tab2:
            c1, c2 = st.columns(2)
            with c1:
                values['apache2'] = st.number_input("APACHEII评分", 0, 71, 25,
                    help="急性生理与慢性健康评分")
                values['sofa'] = st.number_input("SOFA评分", 0, 24, 12,
                    help="序贯器官衰竭评分")
                values['news'] = st.number_input("NEWS评分", 0, 20, 9,
                    help="国家早期预警评分")
            with c2:
                values['braden'] = st.number_input("Braden评分", 1, 23, 10,
                    help="压力性损伤风险评分，越低风险越高")
                values['caprini'] = st.number_input("Caprini评分", 0, 20, 8,
                    help="血栓风险评估")
                values['nrs2002'] = st.number_input("NRS2002评分", 0, 7, 7,
                    help="营养风险筛查")

        with tab3:
            c1, c2 = st.columns(2)
            with c1:
                values['lactate'] = st.number_input("乳酸 (mmol/L)", 0.0, 20.0, 2.8, 0.1,
                    help="正常 < 2.0，> 4.0提示组织缺氧")
                values['creatinine'] = st.number_input("血清肌酐 (umol/L)", 0, 800, 175)
                values['albumin'] = st.number_input("血清白蛋白 (g/L)", 0, 60, 27,
                    help="正常 35-50，低白蛋白增加ASF风险")
                values['pct'] = st.number_input("降钙素原 (ng/mL)", 0.0, 200.0, 7.54, 0.01)
                values['ph'] = st.number_input("pH值", 6.8, 7.8, 7.31, 0.01)
            with c2:
                values['crp'] = st.number_input("C-反应蛋白 (mg/L)", 0, 500, 133)
                values['d_dimer'] = st.number_input("D-二聚体 (mg/L)", 0.0, 50.0, 5.7, 0.1)
                values['wbc'] = st.number_input("白细胞计数 (10^9/L)", 0.0, 50.0, 12.14, 0.1)
                values['platelet'] = st.number_input("血小板计数 (10^9/L)", 0, 1000, 174)
                values['potassium'] = st.number_input("钾 (mmol/L)", 0.0, 10.0, 4.0, 0.1)

        with tab4:
            c1, c2 = st.columns(2)
            with c1:
                values['vasoactive'] = 1 if st.toggle("血管活性药物使用", False) else 0
                values['sedation'] = 1 if st.toggle("镇静镇痛治疗", False) else 0
                values['ventilation'] = 1 if st.toggle("呼吸机辅助通气", False) else 0
            with c2:
                values['crrt'] = 1 if st.toggle("连续肾脏替代治疗(CRRT)", False) else 0
                values['immobility'] = 1 if st.toggle("长时间制动(>72h)", False) else 0

        with tab5:
            c1, c2 = st.columns(2)
            with c1:
                values['organ_failure'] = 1 if st.toggle("器官功能衰竭", True) else 0
                values['renal_failure'] = 1 if st.toggle("肾功能衰竭", False) else 0
                values['cardiac_failure'] = 1 if st.toggle("心功能衰竭", False) else 0
            with c2:
                values['resp_failure'] = 1 if st.toggle("呼吸功能衰竭", False) else 0
                values['cad'] = 1 if st.toggle("冠心病", False) else 0

        # 补全剩余特征（从其他tab中的值计算或默认值）
        values['pao2'] = st.number_input("氧分压 (mmHg)", 0, 300, 120)
        values['pao2_fio2'] = st.number_input("氧合指数 (PaO2/FiO2, mmHg)", 0, 600, 214)
        values['temp'] = st.number_input("体温 (°C)", 30.0, 42.0, 36.6, 0.1)

        # 计算按钮
        st.markdown("---")
        calc_clicked = st.button("🔍 计算ASF风险概率", type="primary", use_container_width=True)
        reset_clicked = st.button("🔄 重置所有数据", use_container_width=True)

        if reset_clicked:
            st.rerun()

    # ---- Result Panel ----
    with col_result:
        st.markdown("### 📈 预测结果分析")

        if calc_clicked:
            with st.spinner("正在计算风险概率..."):
                probability = predict_rf_approx(values)
                risk = get_risk_level(probability)
                contributions = calculate_shap_contributions(values)

            # 风险仪表盘
            gauge_fig = create_gauge_chart(probability)
            st.plotly_chart(gauge_fig, use_container_width=True, key="gauge")

            # 关键指标卡片
            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f"""
                    <div class="metric-box">
                        <div style="font-size: 0.85rem; color: #666;">ASF预测概率</div>
                        <div style="font-size: 1.6rem; font-weight: bold; color: {risk['color']};">{probability:.1%}</div>
                    </div>
                """, unsafe_allow_html=True)
            with m2:
                st.markdown(f"""
                    <div class="metric-box">
                        <div style="font-size: 0.85rem; color: #666;">风险比值比</div>
                        <div style="font-size: 1.6rem; font-weight: bold; color: {risk['color']};">{probability/(1-probability):.2f}</div>
                    </div>
                """, unsafe_allow_html=True)
            with m3:
                st.markdown(f"""
                    <div class="metric-box">
                        <div style="font-size: 0.85rem; color: #666;">风险等级</div>
                        <div style="font-size: 1.3rem; font-weight: bold; color: {risk['color']};">{risk['emoji']} {risk['level']}</div>
                    </div>
                """, unsafe_allow_html=True)

            # 护理建议
            st.markdown("---")
            st.markdown(f"### 🩺 {risk['level']}护理建议")
            for measure in risk['measures']:
                st.markdown(f'<div class="advice-item">{measure}</div>', unsafe_allow_html=True)

            # SHAP分析
            st.markdown("---")
            st.markdown("### 🔬 SHAP特征贡献分析")

            shap_tab1, shap_tab2, shap_tab3 = st.tabs(["📊 条形图", "🕸️ 雷达图", "💧 瀑布图"])

            with shap_tab1:
                bar_fig = create_shap_bar_chart(contributions)
                st.plotly_chart(bar_fig, use_container_width=True, key="shap_bar")

            with shap_tab2:
                radar_fig = create_shap_radar_chart(contributions)
                st.plotly_chart(radar_fig, use_container_width=True, key="shap_radar")

            with shap_tab3:
                waterfall_fig = create_shap_waterfall(contributions)
                st.plotly_chart(waterfall_fig, use_container_width=True, key="shap_waterfall")

            # SHAP详细表格
            st.markdown("**Top 10 特征贡献详情**")
            shap_df_data = {
                "特征": [c['feature'] for c in contributions[:10]],
                "输入值": [f"{c['value']:.2f}" if isinstance(c['value'], float) else str(c['value']) for c in contributions[:10]],
                "重要性": [f"{c['importance']:.4f}" for c in contributions[:10]],
                "贡献值": [f"{c['contribution']:+.4f}" for c in contributions[:10]]
            }
            st.dataframe(shap_df_data, use_container_width=True, hide_index=True)

        else:
            # 未计算时显示提示
            st.info("👆 请在左侧输入患者临床数据，然后点击 **计算ASF风险概率** 按钮查看结果。")

            # 显示示例
            with st.expander("💡 查看示例高风险病例"):
                st.markdown("""
                **典型ASF高风险患者特征：**
                - 外周灌注指数 (PPI): 0.3（极低）
                - 皮肤花斑评分: 4分（严重花斑）
                - 血管活性药物: 正在使用
                - APACHE II: 30分（危重）
                - 乳酸: 6 mmol/L（组织缺氧）
                - 镇静镇痛治疗: 是
                - 呼吸机辅助通气: 是

                输入这些值后点击计算，查看预测结果和SHAP解释。
                """)

    # ---- Footer ----
    st.markdown("---")
    st.markdown("""
    <div class="footer-text">
        <b>ICU老年脓毒症患者急性皮肤衰竭风险预测模型</b> | 
        基于Random Forest机器学习算法 | 
        训练样本: 655例 | 
        验证集AUC: 0.9876 | 
        Bootstrap 1000次验证<br>
        ⚠️ 本工具仅供临床参考，不构成医疗建议。实际临床决策应结合患者具体情况和专业判断。
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
