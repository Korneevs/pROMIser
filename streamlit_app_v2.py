import plotly.express as px
import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go
from tqdm import tqdm
from MILP_2 import CampaignOptimizer
import numpy as np

def mid_CI(plans):
        min_CI_width = 200e6
        index = 101
        for idx, plan in enumerate(plans):
            widths = [elem['high'] - elem['low'] for elem in plan]
            avg_width = np.sqrt(np.sum((np.array(widths)/2) ** 2))/len(widths)
            if avg_width < min_CI_width:
                min_CI_width = avg_width
                index = idx
        return index

def enumerate_plans(optimizer, B_tot, K, **opt_kwargs):
        plans = []
        forbidden = []
        progress_bar = st.progress(0, text="Оптимизация...")
        for i in tqdm(range(K)):
            plan, chosen_x = optimizer.optimize(B_tot, forbidden_plans=forbidden, **opt_kwargs)
            if not plan:
                break
            plans.append(plan)
            forbidden.append(chosen_x)
            progress = (i + 1) / K
            progress_bar.progress(progress, text=f"{i+1}%")
        progress_bar.empty()
        return plans

@st.cache_data
def load_data(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

full_df = load_data('CE_prediction_dict_goods_3.pkl')

lookup = {}
for key, value in full_df.items():
    category, month_str = key.rsplit('_', 1)
    month = int(month_str)
    lookup.setdefault(category, {})[month] = value[[
        'TRP', 'DTB_pred', 'SOV', 'budget', 'revenue',
        'ROMI', 'logcats', 'low', 'high'
    ]]

categories = list(lookup.keys())
optimizer = CampaignOptimizer(categories=categories, lookup=lookup)

st.title("Оптимизация рекламных кампаний (млн)")

st.subheader("Общий бюджет")
budget_mln = st.number_input("Бюджет на все категории (млн)", value=2848, step=100)

st.subheader("Ограничения по категориям")
default_df = pd.DataFrame({
    "Категория": categories,
    "Макс. бюджет (млн)": [2795] * len(categories),
    "Мин. кампаний": [1] * len(categories),
    "Макс. кампаний": [3] * len(categories)
})
edited_df = st.data_editor(
    default_df,
    use_container_width=True,
    num_rows="fixed",
    hide_index=True, 
    column_config={
        "Макс. бюджет (млн)": st.column_config.NumberColumn(step=100),
        "Мин. кампаний": st.column_config.NumberColumn(step=1),
        "Макс. кампаний": st.column_config.NumberColumn(step=1)
    }
)
max_cat = {row["Категория"]: 1e6 * row["Макс. бюджет (млн)"] for _, row in edited_df.iterrows()}
min_campaigns = {row["Категория"]: int(row["Мин. кампаний"]) for _, row in edited_df.iterrows()}
max_campaigns = {row["Категория"]: int(row["Макс. кампаний"]) for _, row in edited_df.iterrows()}

every_month = st.checkbox("Кампании есть в каждом месяце", value=True)
rk_sale = st.checkbox("Обязательные распродажи", value=True)

# ===== Кнопка запуска базовой оптимизации =====
if st.button("Запустить оптимизацию"):
    total_budget = budget_mln * 1e6

    best_plans = enumerate_plans(
        optimizer, 
        B_tot=total_budget,
        K=100,
        min_campaigns=min_campaigns,
        max_campaigns=max_campaigns,
        b_max=max_cat,
        scenario=3,
        every_month=every_month,
        rk_sale=rk_sale
    )

    index = mid_CI(best_plans)
    df_romi = pd.DataFrame(best_plans[index])
    df_romi['duration'] = df_romi['end_month'] - df_romi['start_month'] + 1
    df_romi['ROMI %'] = (df_romi['avg_ROMI'] * 100).astype(int)
    df_romi['SOV %'] = (df_romi['SOV_profile'] * 100).astype(int)

    used_budget = df_romi["total_budget"].sum()
    if used_budget > total_budget:
        st.warning("Не существует оптимального плана при заданных параметрах. Увеличьте бюджет или ослабьте параметры")

    # Сохраняем результат в session_state
    st.session_state['df_romi'] = df_romi
    st.session_state['romi'] = df_romi["total_revenue"].sum() / df_romi["total_budget"].sum() - 1
    st.session_state['budget'] = df_romi['total_budget'].sum()/1e6
    st.session_state['revenue'] = df_romi['total_revenue'].sum()/1e6

    # Матрица кампаний для редактирования
    n_month = 12
    campaign_matrix = pd.DataFrame(
        0,
        index=categories,
        columns=range(1, n_month+1)
    )
    for _, row in df_romi.iterrows():
        months = range(int(row['start_month']), int(row['end_month']) + 1)
        campaign_matrix.loc[row['category'], months] = 1
    st.session_state['campaign_matrix'] = campaign_matrix.copy()

# ===== Показываем базовый план, если уже был расчет =====
if 'df_romi' in st.session_state:
    df_romi = st.session_state['df_romi']
    romi = st.session_state['romi']
    budget = st.session_state['budget']
    revenue = st.session_state['revenue']

    y_order = sorted(set(df_romi['category']), reverse=True)
    y_map = {cat: i + 0.5 for i, cat in enumerate(y_order)}
    palette = px.colors.qualitative.Plotly
    color_map = {cat: palette[i % len(palette)] for i, cat in enumerate(y_order)}

    fig = go.Figure()
    for _, row in df_romi.iterrows():
        fig.add_trace(go.Bar(
            x=[row['duration']],
            y=[y_map[row['category']]],
            base=row['start_month'],
            orientation='h',
            marker_color=color_map[row['category']],
            width=1.0,
            hovertemplate=(
                "DTB: %{customdata[0]}k<br>"
                "ROMI: %{customdata[4]}%<br>"
                "TRP: %{customdata[1]}<br>"
                "budget: %{customdata[3]}mln<br>"
                "SOV: %{customdata[2]}%<extra></extra>"
            ),
            customdata=[
                [ int(row['total_DTB']/1e3),
                  row['total_TRP'],
                  row['SOV %'],
                  int(row['total_budget']/1e6),
                  row['ROMI %'] ]
            ],
            showlegend=False
        ))

    fig.update_xaxes(
        title="Месяц",
        range=[0.5, 13],
        tickmode="array",
        tickvals=list(range(1,13)),
        ticktext=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'],
        showgrid=True, gridcolor="lightgrey", gridwidth=1
    )
    fig.update_yaxes(
        title="",
        tickmode="array",
        tickvals=[y_map[c] for c in y_order],
        ticktext=y_order,
        range=[0, len(y_order)],
        dtick=1
    )
    fig.update_layout(
        title="Диаграмма Ганта рекламных кампаний",
        plot_bgcolor="white",
        margin=dict(l=120, r=20, t=60, b=40),
        bargap=1,
        height = 400
    )
    st.plotly_chart(fig)
    st.metric("ROMI", f"{romi:.2%}")
    st.metric("Общий бюджет", f"{budget:.1f} млн")
    st.metric("Выручка", f"{revenue:.1f} млн")

    # ===== Таблица для редактирования =====

    n_month = 12
    zero_matrix = pd.DataFrame(
        0,
        index=categories,
        columns=range(1, n_month+1)
    )
    st.subheader("Ручное редактирование кампаний по месяцам (1 — кампания есть, 0 — нет)")
    editable_matrix = st.data_editor(
        zero_matrix.reset_index().rename(columns={"index": "Категория"}),
        use_container_width=True,
        hide_index=True,
        column_config={
            str(m): st.column_config.NumberColumn(step=1, min_value=0, max_value=1) for m in range(1, 13)
        }
    )
    # Преобразуем обратно в нужный формат
    edited_matrix = editable_matrix.set_index('Категория')
    edited_matrix = edited_matrix[[str(m) if str(m) in edited_matrix.columns else m for m in range(1, 13)]]
    edited_matrix.columns = range(1, 13)
    edited_matrix = edited_matrix.astype(int)
    st.session_state['edited_matrix'] = edited_matrix.copy()

    # ===== Кнопка запуска оптимизации с ручными ограничениями =====
    if st.button("Оптимизировать с учетом ручных ограничений"):
        total_budget = budget_mln * 1e6

        best_plans_fixed = enumerate_plans(
            optimizer, 
            B_tot=total_budget,
            K=100,
            min_campaigns=min_campaigns,
            max_campaigns=max_campaigns,
            b_max=max_cat,
            scenario=3,
            every_month=every_month,
            rk_sale=rk_sale,
            fixed_campaigns=edited_matrix
        )

        index_fixed = mid_CI(best_plans_fixed)
        df_romi_fixed = pd.DataFrame(best_plans_fixed[index_fixed])
        df_romi_fixed['duration'] = df_romi_fixed['end_month'] - df_romi_fixed['start_month'] + 1
        df_romi_fixed['ROMI %'] = (df_romi_fixed['avg_ROMI'] * 100).astype(int)
        df_romi_fixed['SOV %'] = (df_romi_fixed['SOV_profile'] * 100).astype(int)

        used_budget_fixed = df_romi_fixed["total_budget"].sum()
        if used_budget_fixed > total_budget:
            st.warning("Не существует оптимального плана при заданных параметрах (ручные ограничения). Увеличьте бюджет или ослабьте параметры")

        # Сохраняем результат в session_state
        st.session_state['df_romi_fixed'] = df_romi_fixed
        st.session_state['romi_fixed'] = df_romi_fixed["total_revenue"].sum() / df_romi_fixed["total_budget"].sum() - 1
        st.session_state['budget_fixed'] = df_romi_fixed['total_budget'].sum()/1e6
        st.session_state['dtb_fixed'] = df_romi_fixed['total_DTB'].sum()/1e3
        st.session_state['revenue_fixed'] = df_romi_fixed['total_revenue'].sum()/1e6

# ===== Показываем план с ручными ограничениями, если есть =====
if 'df_romi_fixed' in st.session_state:
    df_romi_fixed = st.session_state['df_romi_fixed']
    romi_fixed = st.session_state['romi_fixed']
    budget_fixed = st.session_state['budget_fixed']
    revenue_fixed = st.session_state['revenue_fixed']
    dtb_fixed = st.session_state['dtb_fixed']

    y_order_fixed = sorted(set(df_romi_fixed['category']), reverse=True)
    y_map_fixed = {cat: i + 0.5 for i, cat in enumerate(y_order_fixed)}
    palette = px.colors.qualitative.Plotly
    color_map_fixed = {cat: palette[i % len(palette)] for i, cat in enumerate(y_order_fixed)}

    fig_fixed = go.Figure()
    for _, row in df_romi_fixed.iterrows():
        fig_fixed.add_trace(go.Bar(
            x=[row['duration']],
            y=[y_map_fixed[row['category']]],
            base=row['start_month'],
            orientation='h',
            marker_color=color_map_fixed[row['category']],
            width=1.0,
            hovertemplate=(
                "DTB: %{customdata[0]}k<br>"
                "ROMI: %{customdata[4]}%<br>"
                "TRP: %{customdata[1]}<br>"
                "budget: %{customdata[3]}mln<br>"
                "SOV: %{customdata[2]}%<extra></extra>"
            ),
            customdata=[
                [ int(row['total_DTB']/1e3),
                row['total_TRP'],
                row['SOV %'],
                int(row['total_budget']/1e6),
                row['ROMI %'] ]
            ],
            showlegend=False
        ))

    fig_fixed.update_xaxes(
        title="Месяц",
        range=[0.5, 13],
        tickmode="array",
        tickvals=list(range(1,13)),
        ticktext=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'],
        showgrid=True, gridcolor="lightgrey", gridwidth=1
    )
    fig_fixed.update_yaxes(
        title="",
        tickmode="array",
        tickvals=[y_map_fixed[c] for c in y_order_fixed],
        ticktext=y_order_fixed,
        range=[0, len(y_order_fixed)],
        dtick=1
    )
    fig_fixed.update_layout(
        title="Диаграмма Ганта (ручные ограничения)",
        plot_bgcolor="white",
        margin=dict(l=120, r=20, t=60, b=40),
        bargap=1,
        height = 400
    )
    st.plotly_chart(fig_fixed)

    st.metric("ROMI (ручные ограничения)", f"{romi_fixed:.2%}")
    st.metric("Общий бюджет (ручные ограничения)", f"{budget_fixed:.1f} млн")
    st.metric("Выручка (ручные ограничения)", f"{revenue_fixed:.1f} млн")
    st.metric("Всего DTB", f"{dtb_fixed:.1f} тыс")

    st.subheader("По категориям")

    # 0) исходная агрегация
    df_cat = df_romi_fixed.groupby('category').agg(
        spent  = ('total_budget',  'sum'),
        earned = ('total_revenue', 'sum')
    )

    # 1) объединяем HL-Construction и HL-Furniture в HL
    hl_labels = [lbl for lbl in ("HL-Construction","HL-Furniture") if lbl in df_cat.index]
    if hl_labels:
        # сумма по субкатегориям
        subs = df_cat.loc[hl_labels, ["spent","earned"]].sum()
        df_cat = df_cat.drop(hl_labels)
        # если уже есть HL, прибавим к нему
        if "HL" in df_cat.index:
            subs += df_cat.loc["HL", ["spent","earned"]]
            df_cat = df_cat.drop("HL")
        df_cat.loc["HL", ["spent","earned"]] = subs

    # 2) склеиваем SP и SP-Tires в единую группу "SP"
    sp_labels = [lbl for lbl in ("SP","SP-Tires") if lbl in df_cat.index]
    if sp_labels:
        sp_sum = df_cat.loc[sp_labels, ["spent","earned"]].sum()
        df_cat = df_cat.drop(sp_labels)
        df_cat.loc["SP", ["spent","earned"]] = sp_sum

    # 3) перераспределяем Sale и CC-Test по EL, HL, LS
    promo_labels = [lbl for lbl in ("Sale","CC-Test") if lbl in df_cat.index]
    alloc = {"EL":0.256, "HL":0.31, "LS":0.434}
    for promo in promo_labels:
        promo_spent  = df_cat.at[promo, 'spent']
        promo_earned = df_cat.at[promo, 'earned']
        df_cat = df_cat.drop(promo)
        for cat, frac in alloc.items():
            if cat not in df_cat.index:
                df_cat.loc[cat] = [0.0,0.0]
            df_cat.at[cat,'spent']  += promo_spent  * frac
            df_cat.at[cat,'earned'] += promo_earned * frac

    # 4) переводим в миллионы и форматируем вывод
    df_cat = (
        df_cat
        .assign(
            spent_mln  = df_cat['spent']/1e6,
            earned_mln = df_cat['earned']/1e6
        )
        .loc[:, ['spent_mln','earned_mln']]
        .rename(columns={'spent_mln':'Потрачено, млн','earned_mln':'Заработано, млн'})
    )

    st.table(df_cat.style.format("{:.1f}"))





    # Сравнение с исходным планом
    if 'romi' in st.session_state and 'budget' in st.session_state and 'revenue' in st.session_state:
        st.subheader("Сравнение с исходным оптимальным планом")
        st.metric("Разница ROMI", f"{romi_fixed - st.session_state['romi']:.2%}")
        st.metric("Разница по выручке", f"{revenue_fixed - st.session_state['revenue']:.1f} млн")
