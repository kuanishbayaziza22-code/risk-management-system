import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import norm
import plotly.express as px
import plotly.graph_objects as go
import io

# ------------------------------------------------------------
# НАСТРОЙКА СТРАНИЦЫ
# ------------------------------------------------------------
st.set_page_config(page_title="Полная система риск-менеджмента (SoftRise)", layout="wide")
st.title("🏛️ Полная подсистема управления рисками (как в SoftRise)")
st.write("Все функции: риски, лимиты, виртуальный портфель, стресс-тесты, бэк-тест, эффективность, ГЭП, стоп-лосс, заключение, страновой риск.")

# ------------------------------------------------------------
# ИНИЦИАЛИЗАЦИЯ ДАННЫХ В СЕССИИ (сохраняем портфель и настройки)
# ------------------------------------------------------------
if 'portfolio' not in st.session_state:
    np.random.seed(42)
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'JPM', 'VTI', 'SPY', 'KZTO', 'KAZ.GS']
    currencies = ['USD']*10 + ['KZT', 'KZT']
    weights = np.random.dirichlet(np.ones(len(tickers)))*100
    weights = np.round(weights,2)
    df = pd.DataFrame({
        'ticker': tickers,
        'currency': currencies,
        'weight': weights,
        'rating': np.random.choice(['AAA','AA','A','BBB','BB','B'], len(tickers)),
        'sector': np.random.choice(['Tech','Finance','Energy','Gov','Consumer'], len(tickers)),
        'country': np.random.choice(['US','KZ','EU','CN'], len(tickers)),
        'asset_type': np.random.choice(['Акции','Облигации','Депозит'], len(tickers)),
        'maturity_years': np.random.choice([0.5,2,4,7,10], len(tickers)),
        'price_buy': np.random.uniform(50,200, len(tickers)),  # цена покупки
        'current_price': np.random.uniform(50,200, len(tickers)), # для стоп-лосса
        'issue_volume': np.random.uniform(1e6, 10e6, len(tickers)) # объём эмиссии для лимита %
    })
    st.session_state.portfolio = df
    # Для стоп-лосс и тейк-профит храним отдельно
    st.session_state.stop_loss = {}
    st.session_state.take_profit = {}

portfolio = st.session_state.portfolio

# ------------------------------------------------------------
# ГЕНЕРАЦИЯ ИСТОРИЧЕСКИХ ДАННЫХ (цены, курс)
# ------------------------------------------------------------
np.random.seed(42)
dates = pd.date_range(end=datetime.today(), periods=100, freq='D')
prices = pd.DataFrame(index=dates)
for t in portfolio['ticker']:
    mu = np.random.uniform(0.0005,0.002)
    sigma = np.random.uniform(0.01,0.03)
    ret = np.random.normal(mu, sigma, 100)
    prices[t] = 100 * np.exp(np.cumsum(ret))
# курс USD/KZT
usd_kzt = 450 + np.cumsum(np.random.normal(0,0.5,100))
usd_kzt = pd.Series(usd_kzt, index=dates)

# ------------------------------------------------------------
# ФУНКЦИЯ РАСЧЁТА ОСНОВНЫХ МЕТРИК (используется везде)
# ------------------------------------------------------------
def calc_metrics(portfolio, prices, usd_kzt):
    w = portfolio['weight'].values / 100
    # локальная валюта
    ret = prices.pct_change().dropna()
    port_ret = ret @ w
    cov = ret.cov()
    vol_local = np.sqrt(w @ cov @ w) * np.sqrt(252)
    # тенге
    prices_kzt = prices.copy()
    for t in portfolio['ticker']:
        if portfolio[portfolio['ticker']==t]['currency'].values[0] == 'USD':
            prices_kzt[t] = prices[t] * usd_kzt
    ret_kzt = prices_kzt.pct_change().dropna()
    port_ret_kzt = ret_kzt @ w
    cov_kzt = ret_kzt.cov()
    vol_kzt = np.sqrt(w @ cov_kzt @ w) * np.sqrt(252)
    # VaR
    Z95 = norm.ppf(0.95)
    Z99 = norm.ppf(0.99)
    mean_daily = port_ret_kzt.mean()
    std_daily = port_ret_kzt.std()
    var95 = -(mean_daily - Z95 * std_daily)
    var99 = -(mean_daily - Z99 * std_daily)
    var_hist = -np.percentile(port_ret_kzt, 5)
    value = 1_000_000  # стоимость портфеля
    var95_money = var95 * value
    var99_money = var99 * value
    var_hist_money = var_hist * value
    # Годовая доходность (для Шарпа)
    annual_return = port_ret_kzt.mean() * 252
    return {
        'vol_local': vol_local,
        'vol_kzt': vol_kzt,
        'var95_pct': var95*100,
        'var99_pct': var99*100,
        'var_hist_pct': var_hist*100,
        'var95_money': var95_money,
        'var99_money': var99_money,
        'var_hist_money': var_hist_money,
        'port_ret_kzt': port_ret_kzt,
        'annual_return': annual_return,
        'port_ret_local': port_ret
    }

metrics = calc_metrics(portfolio, prices, usd_kzt)

# ------------------------------------------------------------
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ------------------------------------------------------------
def get_rating_limit(rating):
    # лимитная сетка по рейтингам (в % от портфеля)
    limits = {'AAA': 30, 'AA': 25, 'A': 20, 'BBB': 15, 'BB': 10, 'B': 5}
    return limits.get(rating, 10)

def country_risk_score(country):
    # условный рейтинг стран (чем выше, тем рискованнее)
    scores = {'US': 1, 'EU': 2, 'CN': 3, 'KZ': 4, 'RU': 5}
    return scores.get(country, 3)

# ------------------------------------------------------------
# ВКЛАДКИ (теперь их больше)
# ------------------------------------------------------------
tabs = st.tabs([
    "📊 Основное",
    "📈 Эффективность",
    "💳 Кредит",
    "🌍 Страновой риск",
    "📊 ГЭП (расшир.)",
    "🔁 Бэк-тест",
    "🧪 Вирт.портфель",
    "⚖️ Лимиты (все)",
    "🛑 Stop-loss",
    "📄 Заключение"
])
(t_main, t_perf, t_credit, t_country, t_gap, t_backtest, t_virtual, t_limits, t_stoploss, t_conclusion) = tabs

# ================================================================
# ВКЛАДКА 1: ОСНОВНОЕ (было)
# ================================================================
with t_main:
    st.subheader("📋 Текущий портфель")
    st.dataframe(portfolio.style.format({'weight': '{:.2f}%'}))
    col1,col2 = st.columns(2)
    col1.metric("Волатильность (лок.)", f"{metrics['vol_local']:.2%}")
    col2.metric("Волатильность (тенге)", f"{metrics['vol_kzt']:.2%}")
    st.subheader("📉 VaR (тенге)")
    c1,c2,c3 = st.columns(3)
    c1.metric("Парам. VaR 95%", f"{metrics['var95_pct']:.2f}%", f"{metrics['var95_money']:,.0f} ₸")
    c2.metric("Истор. VaR 95%", f"{metrics['var_hist_pct']:.2f}%", f"{metrics['var_hist_money']:,.0f} ₸")
    c3.metric("Парам. VaR 99%", f"{metrics['var99_pct']:.2f}%", f"{metrics['var99_money']:,.0f} ₸")
    # Стресс-тест
    st.subheader("🌀 Стресс-тест")
    scenario = st.selectbox("Сценарий", ["Без стресса","Падение рынка -10%","Рост доллара +15%","Кризис (-20% рынок, +20% доллар)"], key="stress_main")
    if scenario == "Падение рынка -10%":
        ms, fx = -0.10, 0.0
    elif scenario == "Рост доллара +15%":
        ms, fx = 0.0, 0.15
    elif scenario == "Кризис (-20% рынок, +20% доллар)":
        ms, fx = -0.20, 0.20
    else:
        ms, fx = 0.0, 0.0
    last_prices = prices.iloc[-1]
    last_rate = usd_kzt.iloc[-1]
    new_prices = last_prices.copy()
    for t in portfolio['ticker']:
        if portfolio[portfolio['ticker']==t]['currency'].values[0] == 'USD':
            new_prices[t] = last_prices[t]*(1+ms)*((last_rate*(1+fx))/last_rate)
        else:
            new_prices[t] = last_prices[t]*(1+ms)
    w = portfolio['weight'].values/100
    cur_val = 1_000_000
    new_val = (new_prices / last_prices) @ w * cur_val
    loss = cur_val - new_val
    st.metric("Убыток при стрессе", f"{-loss/cur_val:.2%}", f"{-loss:,.0f} ₸")

# ================================================================
# ВКЛАДКА 2: ЭФФЕКТИВНОСТЬ (коэффициенты Шарпа, Информативности и др.)
# ================================================================
with t_perf:
    st.subheader("📈 Коэффициенты эффективности управляющих")
    st.write("Расчёт на основе исторической доходности портфеля (синтетической).")

    # Для расчёта используем доходности за разные периоды
    port_ret = metrics['port_ret_local']  # локальные доходности (или в тенге)
    # периоды
    periods = {
        'За месяц': 21,
        'За квартал': 63,
        'За год': 252,
        'С начала': len(port_ret)
    }
    # Безрисковая ставка (для примера 5% годовых)
    risk_free = 0.05
    # Эталонная доходность (бенчмарк) – для информативности используем SPY (имитируем)
    # Генерируем случайный бенчмарк с чуть меньшей доходностью
    bench_ret = np.random.normal(0.0004, 0.015, len(port_ret))
    bench_ret = pd.Series(bench_ret, index=port_ret.index)

    results = []
    for period_name, days in periods.items():
        if days > len(port_ret):
            days = len(port_ret)
        ret_subset = port_ret.iloc[-days:]
        bench_subset = bench_ret.iloc[-days:]
        # Годовая доходность
        ann_ret = ret_subset.mean() * 252
        ann_vol = ret_subset.std() * np.sqrt(252)
        # Коэффициент Шарпа (годовой)
        sharpe = (ann_ret - risk_free) / ann_vol if ann_vol > 0 else 0
        # Норма Шарпа (годовая / волатильность)
        normal_sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
        # Информативность (IR) – отношение избыточной доходности к tracking error
        excess = ret_subset - bench_subset
        tracking_error = excess.std() * np.sqrt(252)
        info_ratio = (ret_subset.mean() - bench_subset.mean()) * 252 / tracking_error if tracking_error > 0 else 0
        # Средневзвешенный (просто среднее из трёх)
        weighted = (sharpe + normal_sharpe + info_ratio) / 3
        results.append({
            'Период': period_name,
            'Год. доходность': ann_ret,
            'Волатильность': ann_vol,
            'Коэф. Шарпа': sharpe,
            'Норма Шарпа': normal_sharpe,
            'Информативность': info_ratio,
            'Средневзвеш.': weighted
        })

    df_perf = pd.DataFrame(results)
    st.dataframe(df_perf.style.format({
        'Год. доходность': '{:.2%}',
        'Волатильность': '{:.2%}',
        'Коэф. Шарпа': '{:.3f}',
        'Норма Шарпа': '{:.3f}',
        'Информативность': '{:.3f}',
        'Средневзвеш.': '{:.3f}'
    }))
    st.caption("Безрисковая ставка = 5%, бенчмарк синтетический. Для реальных данных подставьте свои значения.")

# ================================================================
# ВКЛАДКА 3: КРЕДИТНЫЙ РИСК (был)
# ================================================================
with t_credit:
    st.subheader("💳 Кредитный риск – ожидаемые потери")
    rating_map = {'AAA':(0.01,0.30),'AA':(0.05,0.35),'A':(0.10,0.40),'BBB':(0.20,0.45),'BB':(0.50,0.50),'B':(1.00,0.60)}
    df_cr = portfolio.copy()
    df_cr['PD'] = df_cr['rating'].map(lambda r: rating_map.get(r,(0.5,0.5))[0]/100)
    df_cr['LGD'] = df_cr['rating'].map(lambda r: rating_map.get(r,(0.5,0.5))[1])
    df_cr['exposure'] = df_cr['weight']/100 * 1_000_000
    df_cr['expected_loss'] = df_cr['PD'] * df_cr['LGD'] * df_cr['exposure']
    st.dataframe(df_cr[['ticker','rating','weight','PD','LGD','exposure','expected_loss']].style.format({
        'PD':'{:.2%}','LGD':'{:.2%}','exposure':'{:,.0f}','expected_loss':'{:,.0f}'}))
    st.metric("Суммарные ожидаемые потери", f"{df_cr['expected_loss'].sum():,.0f} ₸")

# ================================================================
# ВКЛАДКА 4: СТРАНОВОЙ РИСК
# ================================================================
with t_country:
    st.subheader("🌍 Страновой риск")
    st.write("Оценка риска по странам на основе внутреннего рейтинга (1 – низкий риск, 5 – высокий).")
    # Берём уникальные страны из портфеля
    countries = portfolio['country'].unique()
    country_data = []
    for c in countries:
        weight_sum = portfolio[portfolio['country']==c]['weight'].sum()
        score = country_risk_score(c)
        # Условный риск: вес * рейтинг (чем выше, тем рискованнее)
        risk_contribution = weight_sum * score / 100
        country_data.append({
            'Страна': c,
            'Доля в портфеле (%)': weight_sum,
            'Рейтинг риска (1-5)': score,
            'Вклад в страновой риск': risk_contribution
        })
    df_country = pd.DataFrame(country_data)
    st.dataframe(df_country.style.format({'Доля в портфеле (%)': '{:.2f}%', 'Вклад в страновой риск': '{:.3f}'}))
    # Суммарный страновой риск
    total_country_risk = df_country['Вклад в страновой риск'].sum()
    st.metric("Суммарный страновой риск (портфеля)", f"{total_country_risk:.3f}")

    # График
    fig = px.bar(df_country, x='Страна', y='Вклад в страновой риск', color='Рейтинг риска (1-5)',
                 title='Страновой риск по странам')
    st.plotly_chart(fig, use_container_width=True)

# ================================================================
# ВКЛАДКА 5: РАСШИРЕННЫЙ ГЭП-АНАЛИЗ (с ручным вводом)
# ================================================================
with t_gap:
    st.subheader("📊 Расширенный ГЭП-анализ")
    st.write("Группировка активов/пассивов по срокам. Можно ввести свои значения пассивов и активов вручную.")

    # Сроки (позиции)
    maturity_bins = ['До востребования', 'До 7 дней', '8-30 дней', '1-3 мес', '3-12 мес', '1-5 лет', 'более 5 лет']
    # Сопоставим с maturity_years (для демо)
    # В реальности данные по активам берутся из системы, здесь мы используем портфель
    # Считаем активы по срокам из портфеля (сумма весов)
    # Для упрощения привяжем maturity_years к бинам
    bins = [0, 0.02, 0.08, 0.25, 1, 5, 100]  # годы
    labels = maturity_bins
    portfolio_gap = portfolio.copy()
    portfolio_gap['bin'] = pd.cut(portfolio_gap['maturity_years'], bins=bins, labels=labels, right=False)
    assets_by_bin = portfolio_gap.groupby('bin')['weight'].sum().reindex(labels, fill_value=0).to_dict()

    # Теперь дадим пользователю ввести пассивы вручную (в % от портфеля или в деньгах)
    st.subheader("Ввод пассивов (в % от портфеля)")
    liabilities = {}
    col_liab = st.columns(len(maturity_bins))
    for i, bin_name in enumerate(maturity_bins):
        with col_liab[i]:
            liabilities[bin_name] = st.number_input(f"{bin_name}", min_value=0.0, max_value=100.0, value=5.0, step=0.5, key=f"liab_{i}")

    # Также можно ввести активы вручную, но мы используем данные из портфеля
    # Показываем таблицу
    gap_df = pd.DataFrame({
        'Срок': maturity_bins,
        'Активы (%)': [assets_by_bin.get(b,0) for b in maturity_bins],
        'Пассивы (%)': [liabilities[b] for b in maturity_bins]
    })
    gap_df['ГЭП (%)'] = gap_df['Активы (%)'] - gap_df['Пассивы (%)']

    st.table(gap_df.style.format({'Активы (%)':'{:.2f}%','Пассивы (%)':'{:.2f}%','ГЭП (%)':'{:.2f}%'}))

    # График
    fig = go.Figure()
    fig.add_trace(go.Bar(x=gap_df['Срок'], y=gap_df['Активы (%)'], name='Активы'))
    fig.add_trace(go.Bar(x=gap_df['Срок'], y=gap_df['Пассивы (%)'], name='Пассивы'))
    fig.add_trace(go.Scatter(x=gap_df['Срок'], y=gap_df['ГЭП (%)'], name='ГЭП', mode='lines+markers'))
    fig.update_layout(title='ГЭП по срокам', xaxis_title='Срок', yaxis_title='% портфеля')
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Активы взяты из портфеля (на основе maturity_years). Пассивы вводятся вручную.")

# ================================================================
# ВКЛАДКА 6: БЭК-ТЕСТИРОВАНИЕ (было, оставляем)
# ================================================================
with t_backtest:
    st.subheader("🔁 Бэк-тестирование VaR (95%)")
    port_ret = metrics['port_ret_kzt']
    window = st.slider("Глубина окна (дней)", 20, 100, 60)
    if len(port_ret) >= window:
        var_forecast = []
        actual_loss = []
        dates_test = port_ret.index[window:]
        for i in range(window, len(port_ret)):
            hist = port_ret.iloc[i-window:i]
            mu = hist.mean()
            sigma = hist.std()
            var = -(mu - norm.ppf(0.95)*sigma)
            var_forecast.append(var)
            actual_loss.append(-port_ret.iloc[i])
        exceed = np.sum(np.array(actual_loss) > np.array(var_forecast))
        total = len(actual_loss)
        st.metric("Превышений", f"{exceed}/{total}", f"{exceed/total:.2%} (ожидается ~5%)")
        df_bt = pd.DataFrame({'Дата':dates_test, 'Прогноз VaR':var_forecast, 'Факт.убыток':actual_loss})
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_bt['Дата'], y=df_bt['Прогноз VaR'], mode='lines', name='VaR прогноз'))
        fig.add_trace(go.Scatter(x=df_bt['Дата'], y=df_bt['Факт.убыток'], mode='markers', name='Факт.убыток'))
        fig.update_layout(title='Бэк-тест VaR', xaxis_title='Дата', yaxis_title='Убыток')
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df_bt.style.format({'Прогноз VaR':'{:.4f}','Факт.убыток':'{:.4f}'}))
    else:
        st.warning(f"Недостаточно данных (нужно >= {window} дней)")

# ================================================================
# ВКЛАДКА 7: ВИРТУАЛЬНЫЙ ПОРТФЕЛЬ (был, расширим)
# ================================================================
with t_virtual:
    st.subheader("🧪 Виртуальный портфель – добавление/удаление позиций")
    with st.form("add_virtual"):
        col1,col2,col3 = st.columns(3)
        new_ticker = col1.text_input("Тикер", "NEW")
        new_currency = col2.selectbox("Валюта", ["USD","KZT"])
        new_weight = col3.number_input("Вес (%)", 0.0, 100.0, 1.0)
        if st.form_submit_button("Добавить"):
            if new_ticker not in portfolio['ticker'].values:
                new_row = pd.DataFrame({
                    'ticker':[new_ticker], 'currency':[new_currency], 'weight':[new_weight],
                    'rating':['BBB'], 'sector':['Other'], 'country':['US'],
                    'asset_type':['Акции'], 'maturity_years':[2],
                    'price_buy':[100], 'current_price':[100], 'issue_volume':[1e6]
                })
                portfolio = pd.concat([portfolio,new_row], ignore_index=True)
                portfolio['weight'] = portfolio['weight'] / portfolio['weight'].sum() * 100
                portfolio['weight'] = np.round(portfolio['weight'],2)
                st.session_state.portfolio = portfolio
                st.success("Добавлено")
                st.rerun()
    with st.form("remove_virtual"):
        ticker_to_remove = st.selectbox("Выберите тикер для удаления", portfolio['ticker'].tolist())
        if st.form_submit_button("Удалить"):
            portfolio = portfolio[portfolio['ticker']!=ticker_to_remove].reset_index(drop=True)
            portfolio['weight'] = portfolio['weight'] / portfolio['weight'].sum() * 100
            portfolio['weight'] = np.round(portfolio['weight'],2)
            st.session_state.portfolio = portfolio
            st.success("Удалено")
            st.rerun()
    st.dataframe(portfolio.style.format({'weight':'{:.2f}%'}))
    # Пересчёт метрик для виртуального портфеля
    with st.spinner("Пересчёт..."):
        new_metrics = calc_metrics(portfolio, prices, usd_kzt)
        st.metric("Волатильность (тенге)", f"{new_metrics['vol_kzt']:.2%}")
        st.metric("VaR 95%", f"{new_metrics['var95_pct']:.2f}%", f"{new_metrics['var95_money']:,.0f} ₸")

# ================================================================
# ВКЛАДКА 8: ВСЕ ЛИМИТЫ (расширенные)
# ================================================================
with t_limits:
    st.subheader("⚖️ Все лимиты (с лимитной сеткой по рейтингам)")

    # 1. Лимиты на типы, валюты, сектора, страны (были)
    limits_general = {
        'asset_type': {'Акции':50, 'Облигации':60, 'Депозит':30},
        'currency': {'USD':60, 'KZT':70},
        'sector': {'Tech':30, 'Finance':40, 'Energy':25, 'Gov':20, 'Consumer':30, 'Other':20},
        'country': {'US':50, 'KZ':40, 'EU':30, 'CN':20}
    }
    total = portfolio['weight'].sum()
    cur_asset = portfolio.groupby('asset_type')['weight'].sum() / total * 100
    cur_curr = portfolio.groupby('currency')['weight'].sum() / total * 100
    cur_sector = portfolio.groupby('sector')['weight'].sum() / total * 100
    cur_country = portfolio.groupby('country')['weight'].sum() / total * 100

    def check_limit(curr, lim):
        violations = {}
        for k,v in lim.items():
            if k in curr.index and curr[k] > v:
                violations[k] = (curr[k], v)
        return violations

    v_asset = check_limit(cur_asset, limits_general['asset_type'])
    v_curr = check_limit(cur_curr, limits_general['currency'])
    v_sector = check_limit(cur_sector, limits_general['sector'])
    v_country = check_limit(cur_country, limits_general['country'])

    st.subheader("Общие лимиты")
    if any([v_asset,v_curr,v_sector,v_country]):
        st.error("Нарушения общих лимитов:")
        for d in [v_asset, v_curr, v_sector, v_country]:
            for k, (fact, lim) in d.items():
                st.write(f"- {k}: факт {fact:.2f}% > лимит {lim}% (превышение {fact-lim:.2f}%)")
    else:
        st.success("Все общие лимиты соблюдены.")

    # Свободные лимиты в деньгах (условно 1 млн)
    port_value = 1_000_000
    st.subheader("Свободные лимиты (в денежном выражении)")
    free_asset = {k: max(0, v - cur_asset.get(k,0)) / 100 * port_value for k,v in limits_general['asset_type'].items()}
    free_curr = {k: max(0, v - cur_curr.get(k,0)) / 100 * port_value for k,v in limits_general['currency'].items()}
    free_sector = {k: max(0, v - cur_sector.get(k,0)) / 100 * port_value for k,v in limits_general['sector'].items()}
    free_country = {k: max(0, v - cur_country.get(k,0)) / 100 * port_value for k,v in limits_general['country'].items()}
    st.write("**По типам активов:**", {k: f"{v:,.0f} ₸" for k,v in free_asset.items()})
    st.write("**По валютам:**", {k: f"{v:,.0f} ₸" for k,v in free_curr.items()})
    st.write("**По секторам:**", {k: f"{v:,.0f} ₸" for k,v in free_sector.items()})
    st.write("**По странам:**", {k: f"{v:,.0f} ₸" for k,v in free_country.items()})

    # 2. Лимиты на эмитента (с учётом рейтинга)
    st.subheader("Лимиты на эмитентов (с лимитной сеткой по рейтингу)")
    st.write("Для каждого эмитента лимит устанавливается в % от портфеля в зависимости от рейтинга.")
    # Добавляем колонку с лимитом для каждого эмитента
    port_limits = portfolio.copy()
    port_limits['limit_rating'] = port_limits['rating'].apply(get_rating_limit)
    port_limits['current_weight'] = port_limits['weight']
    port_limits['excess'] = port_limits['current_weight'] - port_limits['limit_rating']
    port_limits['limit_ok'] = port_limits['excess'] <= 0
    st.dataframe(port_limits[['ticker','rating','current_weight','limit_rating','excess','limit_ok']].style.format({
        'current_weight':'{:.2f}%', 'limit_rating':'{:.2f}%', 'excess':'{:.2f}%'
    }).map(lambda x: 'background-color: red' if x == False else '', subset=['limit_ok']))

    # Свободные лимиты по эмитентам (в деньгах)
    st.write("**Свободные лимиты по эмитентам (сколько можно докупить):**")
    for _, row in port_limits.iterrows():
        free_money = max(0, (row['limit_rating'] - row['current_weight']) / 100 * port_value)
        st.write(f"- {row['ticker']}: {free_money:,.0f} ₸")

    # 3. Лимит на количество акций/облигаций (% от эмиссии)
    st.subheader("Лимит на количество акций/облигаций (% от эмиссии)")
    st.write("Учитывается доля в % от общего объёма эмиссии (issue_volume).")
    # Для каждого инструмента считаем долю в % от эмиссии (условно)
    portfolio['position_value'] = portfolio['weight']/100 * port_value  # стоимость позиции
    portfolio['share_of_issue'] = portfolio['position_value'] / portfolio['issue_volume'] * 100
    limit_issue = 5  # например, не более 5% от эмиссии
    portfolio['issue_ok'] = portfolio['share_of_issue'] <= limit_issue
    st.dataframe(portfolio[['ticker','weight','position_value','issue_volume','share_of_issue','issue_ok']].style.format({
        'position_value':'{:,.0f}', 'issue_volume':'{:,.0f}', 'share_of_issue':'{:.2f}%'
    }).map(lambda x: 'background-color: red' if x == False else '', subset=['issue_ok']))
    # Свободный лимит по эмиссии (в деньгах)
    st.write("**Свободный лимит по эмиссии (можно ещё купить):**")
    for _, row in portfolio.iterrows():
        max_invest = row['issue_volume'] * limit_issue / 100
        free_issue = max(0, max_invest - row['position_value'])
        st.write(f"- {row['ticker']}: {free_issue:,.0f} ₸")

    # 4. Лимит на ГЭП-позиции (добавим простой лимит на абсолютный ГЭП)
    st.subheader("Лимит на ГЭП-позиции")
    gap_abs = gap_df['ГЭП (%)'].abs().sum()
    limit_gap = 20  # например, суммарный ГЭП не более 20%
    if gap_abs > limit_gap:
        st.error(f"Нарушение лимита ГЭП: суммарный ГЭП = {gap_abs:.2f}% > {limit_gap}%")
    else:
        st.success(f"Лимит ГЭП соблюдён: {gap_abs:.2f}% <= {limit_gap}%")

    # 5. Лимит НВА (высоколиквидные активы) – упрощённо, считаем, что это активы с рейтингом AAA и сроком <1 года
    hva = portfolio[(portfolio['rating']=='AAA') & (portfolio['maturity_years']<1)]
    hva_share = hva['weight'].sum()
    limit_hva = 30  # не менее 30% (или не более)
    if hva_share < limit_hva:
        st.warning(f"НВА (высоколиквидные активы) = {hva_share:.2f}% < {limit_hva}% (нарушение)")
    else:
        st.success(f"НВА = {hva_share:.2f}% >= {limit_hva}% (соблюдено)")

# ================================================================
# ВКЛАДКА 9: STOP-LOSS / TAKE-PROFIT
# ================================================================
with t_stoploss:
    st.subheader("🛑 Stop-loss / Take-profit")
    st.write("Установите уровни для каждой позиции (в % от цены покупки).")
    # Показываем таблицу с текущими ценами и уровнями
    # Даём возможность задать уровни для каждого тикера
    st.subheader("Текущие позиции и уровни")
    # Инициализируем словари в сессии, если их нет
    if 'stop_loss' not in st.session_state:
        st.session_state.stop_loss = {}
    if 'take_profit' not in st.session_state:
        st.session_state.take_profit = {}

    # Для каждого тикера создаём поля ввода
    cols = st.columns(len(portfolio))
    for i, (idx, row) in enumerate(portfolio.iterrows()):
        ticker = row['ticker']
        with cols[i]:
            st.write(f"**{ticker}**")
            current_price = row['current_price']
            buy_price = row['price_buy']
            # Поля для ввода
            sl = st.number_input(f"SL {ticker} %", min_value=0.0, max_value=50.0, value=5.0, step=0.5, key=f"sl_{ticker}")
            tp = st.number_input(f"TP {ticker} %", min_value=0.0, max_value=50.0, value=10.0, step=0.5, key=f"tp_{ticker}")
            st.session_state.stop_loss[ticker] = sl
            st.session_state.take_profit[ticker] = tp
            # Проверяем нарушение
            price_change = (current_price - buy_price) / buy_price * 100
            if price_change <= -sl:
                st.error(f"❌ Stop-loss сработал! ({price_change:.2f}%)")
            elif price_change >= tp:
                st.success(f"✅ Take-profit сработал! ({price_change:.2f}%)")
            else:
                st.info(f"Изменение: {price_change:.2f}%")

    # Общая таблица
    st.subheader("Сводка по стоп-лосс / тейк-профит")
    sl_df = portfolio.copy()
    sl_df['SL_%'] = sl_df['ticker'].map(lambda x: st.session_state.stop_loss.get(x, 5.0))
    sl_df['TP_%'] = sl_df['ticker'].map(lambda x: st.session_state.take_profit.get(x, 10.0))
    sl_df['изменение_%'] = (sl_df['current_price'] - sl_df['price_buy']) / sl_df['price_buy'] * 100
    sl_df['SL_нарушено'] = sl_df['изменение_%'] <= -sl_df['SL_%']
    sl_df['TP_достигнут'] = sl_df['изменение_%'] >= sl_df['TP_%']
    st.dataframe(sl_df[['ticker','price_buy','current_price','изменение_%','SL_%','TP_%','SL_нарушено','TP_достигнут']].style.format({
        'price_buy':'{:.2f}', 'current_price':'{:.2f}', 'изменение_%':'{:.2f}%', 'SL_%':'{:.2f}%', 'TP_%':'{:.2f}%'
    }).map(lambda x: 'background-color: red' if x == True else '', subset=['SL_нарушено'])\
     .map(lambda x: 'background-color: green' if x == True else '', subset=['TP_достигнут']))

# ================================================================
# ВКЛАДКА 10: ЗАКЛЮЧЕНИЕ РИСК-МЕНЕДЖМЕНТА
# ================================================================
with t_conclusion:
    st.subheader("📄 Заключение риск-менеджмента")
    st.write("Автоматически сгенерированный отчёт о влиянии текущего портфеля на риски и лимиты.")

    # Собираем информацию
    violations_list = []
    # Общие лимиты
    for d in [v_asset, v_curr, v_sector, v_country]:
        for k, (fact, lim) in d.items():
            violations_list.append(f"Нарушение лимита по {k}: факт {fact:.2f}% > лимит {lim}%")
    # Лимиты эмитентов
    for _, row in port_limits.iterrows():
        if row['excess'] > 0:
            violations_list.append(f"Нарушение лимита по эмитенту {row['ticker']}: факт {row['current_weight']:.2f}% > лимит {row['limit_rating']:.2f}%")
    # Лимиты по эмиссии
    for _, row in portfolio.iterrows():
        if not row['issue_ok']:
            violations_list.append(f"Нарушение лимита по эмиссии {row['ticker']}: доля {row['share_of_issue']:.2f}% > 5%")
    # Лимит ГЭП
    if gap_abs > limit_gap:
        violations_list.append(f"Нарушение лимита ГЭП: суммарный ГЭП = {gap_abs:.2f}% > {limit_gap}%")
    # НВА
    if hva_share < limit_hva:
        violations_list.append(f"Нарушение лимита НВА: {hva_share:.2f}% < {limit_hva}%")
    # Стоп-лосс
    sl_violations = sl_df[sl_df['SL_нарушено']]
    if not sl_violations.empty:
        violations_list.append(f"Сработал стоп-лосс по позициям: {', '.join(sl_violations['ticker'])}")

    # Риски
    risk_summary = f"""
    **Риски портфеля:**
    - Годовая волатильность (тенге): {metrics['vol_kzt']:.2%}
    - VaR 95%: {metrics['var95_pct']:.2f}% (≈ {metrics['var95_money']:,.0f} ₸)
    - VaR 99%: {metrics['var99_pct']:.2f}% (≈ {metrics['var99_money']:,.0f} ₸)
    - Исторический VaR 95%: {metrics['var_hist_pct']:.2f}% (≈ {metrics['var_hist_money']:,.0f} ₸)
    - Коэффициент Шарпа (годовой): {df_perf[df_perf['Период']=='За год']['Коэф. Шарпа'].values[0]:.3f}
    - Суммарные ожидаемые потери (кредитный риск): {df_cr['expected_loss'].sum():,.0f} ₸
    - Страновой риск (суммарный): {total_country_risk:.3f}
    """

    if violations_list:
        conclusion = "### ⚠️ Рекомендация: портфель требует корректировки.\n\nНарушены следующие лимиты:\n"
        for v in violations_list:
            conclusion += f"- {v}\n"
        conclusion += "\n### Рекомендации по снижению рисков:\n"
        conclusion += "- Снизить долю в валюте USD до 60%.\n"
        conclusion += "- Уменьшить концентрацию в секторе Energy.\n"
        conclusion += "- Диверсифицировать по странам (снизить долю US и EU).\n"
        conclusion += "- Пересмотреть позиции с превышением лимита по эмитенту.\n"
        conclusion += "- Рассмотреть возможность хеджирования валютного риска.\n"
    else:
        conclusion = "### ✅ Портфель соответствует всем установленным лимитам. Риски находятся в допустимых пределах."

    st.markdown(conclusion)
    st.markdown(risk_summary)

    # Кнопка скачать заключение
    report_text = f"ЗАКЛЮЧЕНИЕ РИСК-МЕНЕДЖМЕНТА\n{datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n{conclusion}\n\n{risk_summary}"
    st.download_button("📥 Скачать заключение (TXT)", report_text, file_name=f"conclusion_{datetime.now().strftime('%Y%m%d')}.txt")

# ------------------------------------------------------------
# БОКОВАЯ ПАНЕЛЬ
# ------------------------------------------------------------
st.sidebar.write("---")
st.sidebar.write("📊 **Полная система риск-менеджмента (MVP)**")
st.sidebar.write("Синтетические данные. Для реальной работы замените генерацию на загрузку из CSV.")
st.sidebar.write(f"Дата расчёта: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
