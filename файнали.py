import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import norm
import plotly.express as px
import plotly.graph_objects as go
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import ssl
import hashlib

# ------------------------------------------------------------
# НАСТРОЙКА СТРАНИЦЫ
# ------------------------------------------------------------
st.set_page_config(page_title="Полная система риск-менеджмента", layout="wide")
st.title("🏛️ Полная подсистема управления рисками")
st.write("Все функции: риски, лимиты, виртуальный портфель, стресс-тесты, бэк-тест, эффективность, ГЭП, стоп-лосс, заключение, страновой риск.")

# ------------------------------------------------------------
# ИНИЦИАЛИЗАЦИЯ ДАННЫХ В СЕССИИ
# ------------------------------------------------------------
if 'portfolio' not in st.session_state:
    np.random.seed(42)
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'JPM', 'VTI', 'SPY', 'KZTO', 'KAZ.GS']
    currencies = ['USD']*10 + ['KZT', 'KZT']
    weights = np.random.dirichlet(np.ones(len(tickers)))*100
    weights = np.round(weights,2)
    rating_map = {'AAA':(0.01,0.30),'AA':(0.05,0.35),'A':(0.10,0.40),'BBB':(0.20,0.45),'BB':(0.50,0.50),'B':(1.00,0.60)}
    df = pd.DataFrame({
        'ticker': tickers,
        'currency': currencies,
        'weight': weights,
        'rating': np.random.choice(['AAA','AA','A','BBB','BB','B'], len(tickers)),
        'sector': np.random.choice(['Tech','Finance','Energy','Gov','Consumer'], len(tickers)),
        'country': np.random.choice(['US','KZ','EU','CN'], len(tickers)),
        'asset_type': np.random.choice(['Акции','Облигации','Депозит'], len(tickers)),
        'maturity_years': np.random.choice([0.5,2,4,7,10], len(tickers)),
        'price_buy': np.random.uniform(50,200, len(tickers)),
        'current_price': np.random.uniform(50,200, len(tickers)),
        'issue_volume': np.random.uniform(1e6, 10e6, len(tickers))
    })
    df['stage'] = 1
    df['macro_k'] = 1.0
    df['sppi'] = True
    df['business_model'] = 'БМ-1'
    df['valuation_category'] = 'Амортизированная стоимость'
    df['EIR'] = 0.05
    df['LGD_case_specific'] = 0.5
    df['ecl'] = 0.0
    df['exposure'] = df['weight'] / 100 * 1_000_000
    df['PD'] = df['rating'].map(lambda r: rating_map.get(r,(0.5,0.5))[0]/100)
    df['LGD'] = df['rating'].map(lambda r: rating_map.get(r,(0.5,0.5))[1])
    st.session_state.portfolio = df
    st.session_state.stop_loss = {}
    st.session_state.take_profit = {}
    st.session_state.alert_sent = {}
    st.session_state.incidents = pd.DataFrame(columns=['Дата','Описание','Категория','Ущерб','Меры'])
    st.session_state.risk_register = pd.DataFrame()
    st.session_state.risk_assessment = pd.DataFrame()
    st.session_state.limits_override = {}
    st.session_state.metrics = {}

if 'email_settings' not in st.session_state:
    st.session_state.email_settings = {
        'smtp_server': 'smtp.gmail.com',
        'smtp_port': 587,
        'sender_email': '',
        'sender_password': '',
        'recipient_email': ''
    }

portfolio = st.session_state.portfolio

# ------------------------------------------------------------
# ГЕНЕРАЦИЯ ИСТОРИЧЕСКИХ ДАННЫХ
# ------------------------------------------------------------
np.random.seed(42)
dates = pd.date_range(end=datetime.today(), periods=100, freq='D')
prices = pd.DataFrame(index=dates)
for t in portfolio['ticker']:
    mu = np.random.uniform(0.0005,0.002)
    sigma = np.random.uniform(0.01,0.03)
    ret = np.random.normal(mu, sigma, 100)
    prices[t] = 100 * np.exp(np.cumsum(ret))
usd_kzt = 450 + np.cumsum(np.random.normal(0,0.5,100))
usd_kzt = pd.Series(usd_kzt, index=dates)

# ------------------------------------------------------------
# ФУНКЦИИ (глобальные)
# ------------------------------------------------------------
def calc_metrics(portfolio, prices, usd_kzt):
    w = portfolio['weight'].values / 100
    ret = prices.pct_change().dropna()
    port_ret = ret @ w
    cov = ret.cov()
    vol_local = np.sqrt(w @ cov @ w) * np.sqrt(252)
    prices_kzt = prices.copy()
    for t in portfolio['ticker']:
        if portfolio[portfolio['ticker']==t]['currency'].values[0] == 'USD':
            prices_kzt[t] = prices[t] * usd_kzt
    ret_kzt = prices_kzt.pct_change().dropna()
    port_ret_kzt = ret_kzt @ w
    cov_kzt = ret_kzt.cov()
    vol_kzt = np.sqrt(w @ cov_kzt @ w) * np.sqrt(252)
    Z95 = norm.ppf(0.95)
    Z99 = norm.ppf(0.99)
    mean_daily = port_ret_kzt.mean()
    std_daily = port_ret_kzt.std()
    var95 = -(mean_daily - Z95 * std_daily)
    var99 = -(mean_daily - Z99 * std_daily)
    var_hist = -np.percentile(port_ret_kzt, 5)
    value = 1_000_000
    var95_money = var95 * value
    var99_money = var99 * value
    var_hist_money = var_hist * value
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
st.session_state.metrics = metrics

# Расчёт кредитных метрик (глобально)
rating_map = {'AAA':(0.01,0.30),'AA':(0.05,0.35),'A':(0.10,0.40),'BBB':(0.20,0.45),'BB':(0.50,0.50),'B':(1.00,0.60)}
df_cr = portfolio.copy()
df_cr['PD'] = df_cr['rating'].map(lambda r: rating_map.get(r,(0.5,0.5))[0]/100)
df_cr['LGD'] = df_cr['rating'].map(lambda r: rating_map.get(r,(0.5,0.5))[1])
df_cr['exposure'] = df_cr['weight']/100 * 1_000_000
df_cr['expected_loss'] = df_cr['PD'] * df_cr['LGD'] * df_cr['exposure']

# Страновой риск (глобально)
def country_risk_score(country):
    scores = {'US': 1, 'EU': 2, 'CN': 3, 'KZ': 4, 'RU': 5}
    return scores.get(country, 3)

countries = portfolio['country'].unique()
country_data = []
for c in countries:
    weight_sum = portfolio[portfolio['country']==c]['weight'].sum()
    score = country_risk_score(c)
    risk_contribution = weight_sum * score / 100
    country_data.append({'Страна': c, 'Доля в портфеле (%)': weight_sum, 'Рейтинг риска (1-5)': score, 'Вклад в страновой риск': risk_contribution})
df_country = pd.DataFrame(country_data)
total_country_risk = df_country['Вклад в страновой риск'].sum()

def get_rating_limit(rating):
    limits = {'AAA': 30, 'AA': 25, 'A': 20, 'BBB': 15, 'BB': 10, 'B': 5}
    return limits.get(rating, 10)

def send_email_alert(subject, message):
    settings = st.session_state.email_settings
    if not all([settings['smtp_server'], settings['sender_email'], 
                settings['sender_password'], settings['recipient_email']]):
        return False, "Почта не настроена: все поля не заполнены"
    try:
        msg = MIMEMultipart()
        msg['From'] = settings['sender_email']
        msg['To'] = settings['recipient_email']
        msg['Subject'] = f"🚨 {subject}"
        body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px; border: 1px solid #ddd; border-radius: 8px;">
            <h2 style="color: #d32f2f; border-bottom: 2px solid #d32f2f; padding-bottom: 10px;">
                🚨 Предупреждение риск-мониторинга!
            </h2>
            <p style="font-size: 16px; line-height: 1.6;">
                {message}
            </p>
            <hr style="border: 1px solid #eee; margin: 20px 0;">
            <small style="color: #888;">
                Это сообщение отправлено автоматически из системы <b>Risk Management MVP</b>.<br>
                Отвечать на него не нужно.
            </small>
        </body>
        </html>
        """
        msg.attach(MIMEText(body, 'html'))
        context = ssl.create_default_context()
        with smtplib.SMTP(settings['smtp_server'], int(settings['smtp_port'])) as server:
            server.starttls(context=context)
            server.login(settings['sender_email'], settings['sender_password'])
            server.send_message(msg)
        return True, "Письмо успешно отправлено"
    except Exception as e:
        return False, f"Ошибка: {str(e)}"

def check_and_alert():
    """Проверяет все лимиты и стоп-лоссы, отправляет уведомление по почте при нарушениях"""
    messages = []
    portfolio = st.session_state.portfolio
    total = portfolio['weight'].sum()
    if total == 0:
        return None

    # 1. Валютный лимит (USD)
    cur_curr = portfolio.groupby('currency')['weight'].sum() / total * 100
    if 'USD' in cur_curr.index and cur_curr['USD'] > 60:
        messages.append(f"💸 Валютный лимит нарушен!\nДоля USD: {cur_curr['USD']:.2f}% (лимит: 60%)")

    # 2. Секторный лимит (Energy)
    cur_sector = portfolio.groupby('sector')['weight'].sum() / total * 100
    if 'Energy' in cur_sector.index and cur_sector['Energy'] > 25:
        messages.append(f"⚡ Секторный лимит нарушен!\nДоля Energy: {cur_sector['Energy']:.2f}% (лимит: 25%)")

    # 3. Стоп-лосс
    sl_df = portfolio.copy()
    sl_df['SL_%'] = sl_df['ticker'].map(lambda x: st.session_state.stop_loss.get(x, 5.0))
    sl_df['изменение_%'] = (sl_df['current_price'] - sl_df['price_buy']) / sl_df['price_buy'] * 100
    sl_df['SL_нарушено'] = sl_df['изменение_%'] <= -sl_df['SL_%']
    sl_violations = sl_df[sl_df['SL_нарушено']]
    if not sl_violations.empty:
        tickers = ', '.join(sl_violations['ticker'])
        messages.append(f"🛑 Стоп-лосс сработал!\nПозиции: {tickers}")

    # 4. VaR лимит
    metrics_local = calc_metrics(portfolio, prices, usd_kzt)
    if metrics_local['var95_pct'] > 5.0:
        messages.append(f"📊 Лимит VaR превышен!\nТекущий VaR (95%): {metrics_local['var95_pct']:.2f}% (лимит: 5%)")

    if messages:
        full_message = "🚨 Предупреждение риск-мониторинга!\n\n" + "\n\n".join(messages)
        msg_hash = hashlib.md5(full_message.encode()).hexdigest()
        today = datetime.now().strftime('%Y-%m-%d')
        if st.session_state.alert_sent.get(msg_hash) != today:
            success, msg = send_email_alert("Предупреждение о рисках!", full_message)
            if success:
                st.sidebar.success(f"📧 Письмо отправлено: {datetime.now().strftime('%H:%M')}")
            else:
                st.sidebar.error(f"❌ Ошибка почты: {msg}")
            st.session_state.alert_sent[msg_hash] = today
        return full_message
    return None

def calculate_ECL(row):
    """Расчёт ECL по трёхстадийной модели (ОКУ)"""
    stage = row['stage']
    EAD = row['exposure']
    LGD = row['LGD']
    EIR = row['EIR']
    macro_k = row['macro_k']
    if stage == 1:
        PD = row['PD']
        n = 360
        ECL = EAD * PD * LGD * macro_k / ((1 + EIR) ** (n/360))
    elif stage == 2:
        PD_lifetime = 1 - (1 - row['PD']) ** row['maturity_years']
        ECL = EAD * PD_lifetime * LGD * macro_k
    else:  # stage == 3
        ECL = EAD * row['LGD_case_specific']
    return ECL

def get_limits():
    """Возвращает лимиты (с возможным переопределением через session_state)"""
    default = {
        'asset_type': {'Акции':50, 'Облигации':60, 'Депозит':30},
        'currency': {'USD':60, 'KZT':70},
        'sector': {'Tech':30, 'Finance':40, 'Energy':25, 'Gov':20, 'Consumer':30, 'Other':20},
        'country': {'US':50, 'KZ':40, 'EU':30, 'CN':20}
    }
    if 'limits_override' in st.session_state:
        # Мержим с переопределёнными значениями (только те, что есть в default)
        merged = default.copy()
        for cat in merged:
            if cat in st.session_state.limits_override:
                for k in merged[cat]:
                    if k in st.session_state.limits_override[cat]:
                        merged[cat][k] = st.session_state.limits_override[cat][k]
        return merged
    return default

def calculate_duration_and_sensitivity(portfolio, prices, usd_kzt, rate_shock=0.01):
    """
    Рассчитывает модифицированную дюрацию для облигаций и оценивает
    изменение стоимости портфеля при изменении процентной ставки.
    """
    if 'maturity_years' not in portfolio.columns or portfolio['maturity_years'].sum() == 0:
        return None, None
    weighted_duration = (portfolio['weight'] * portfolio['maturity_years']).sum() / portfolio['weight'].sum()
    ytm = 0.05
    modified_duration = weighted_duration / (1 + ytm)
    portfolio_value = 1_000_000
    price_change = -modified_duration * rate_shock * portfolio_value
    return modified_duration, price_change

# ------------------------------------------------------------
# ОСНОВНАЯ ЧАСТЬ (рендеринг)
# ------------------------------------------------------------
if st.button("🔄 Проверить лимиты сейчас"):
    alert = check_and_alert()
    if alert:
        st.error(alert)
    else:
        st.success("✅ Все лимиты в порядке")

# Автоматическая проверка при загрузке (без вывода на экран)
check_and_alert()

# ------------------------------------------------------------
# ВКЛАДКИ
# ------------------------------------------------------------
tabs = st.tabs([
    "📊 Основное",
    "📈 Визуализация",
    "📈 Эффективность",
    "💳 Кредит",
    "💱 Валютный риск",
    "📈 Рыночный риск",
    "📊 Процентный риск",
    "🌍 Страновой риск",
    "📊 ГЭП (расшир.)",
    "🔁 Бэк-тест",
    "🧪 Вирт.портфель",
    "⚖️ Лимиты (все)",
    "🛑 Stop-loss",
    "📄 Заключение",
    "📋 Классификация и ОКУ",
    "📊 KRI",
    "📝 Инциденты",
    "📋 Регистр рисков",
    "📊 Оценка рисков"
])
(t_main, t_vis, t_perf, t_credit, t_currency, t_market, t_interest, t_country, t_gap, t_backtest, t_virtual, t_limits, t_stoploss, t_conclusion, t_classification, t_kri, t_incidents, t_risk_register, t_risk_assessment) = tabs

# ================================================================
# ВКЛАДКА 1: ОСНОВНОЕ
# ================================================================
with t_main:
    portfolio = st.session_state.portfolio
    st.subheader("📋 Текущий портфель")
    portfolio_display = portfolio[['ticker','currency','weight','rating','sector','country','asset_type','maturity_years','price_buy','current_price','issue_volume']].copy()
    portfolio_display.columns = ['Тикер','Валюта','Вес, %','Рейтинг','Сектор','Страна','Тип актива','Срок (лет)','Цена покупки','Текущая цена','Объём эмиссии']
    st.dataframe(portfolio_display.style.format({'Вес, %':'{:.2f}%','Срок (лет)':'{:.1f} г.','Цена покупки':'{:.2f}','Текущая цена':'{:.2f}','Объём эмиссии':'{:,.0f}'}))
    metrics = calc_metrics(portfolio, prices, usd_kzt)
    st.session_state.metrics = metrics
    col1,col2 = st.columns(2)
    col1.metric("Волатильность (лок.)", f"{metrics['vol_local']:.2%}")
    col2.metric("Волатильность (тенге)", f"{metrics['vol_kzt']:.2%}")
    st.subheader("📉 VaR (тенге)")
    c1,c2,c3 = st.columns(3)
    c1.metric("Парам. VaR 95%", f"{metrics['var95_pct']:.2f}%", f"{metrics['var95_money']:,.0f} ₸")
    c2.metric("Истор. VaR 95%", f"{metrics['var_hist_pct']:.2f}%", f"{metrics['var_hist_money']:,.0f} ₸")
    c3.metric("Парам. VaR 99%", f"{metrics['var99_pct']:.2f}%", f"{metrics['var99_money']:,.0f} ₸")
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
# ВКЛАДКА 2: ВИЗУАЛИЗАЦИЯ
# ================================================================
with t_vis:
    st.subheader("📈 Расширенная визуализация")
    st.write("Тепловая карта, динамика VaR, сравнение стресс-тестов, 3D ГЭП.")

    # ---------- 1. ТЕПЛОВАЯ КАРТА ----------
    st.subheader("🔥 Тепловая карта активов")
    ret_assets = prices.pct_change().dropna()
    if len(ret_assets.columns) > 1:
        corr = ret_assets.corr()
        fig_corr = px.imshow(corr, text_auto=True, aspect="auto", 
                             title="Корреляция доходностей активов",
                             color_continuous_scale='RdBu_r',
                             zmin=-1, zmax=1)
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("Недостаточно активов для построения тепловой карты корреляции.")

    # Вклад каждого актива в VaR
    w = portfolio['weight'].values / 100
    if not ret_assets.empty and len(ret_assets.columns) > 0:
        cov_matrix = ret_assets.cov()
        port_var = w @ cov_matrix @ w
        if port_var > 0:
            marg_contrib = cov_matrix @ w
            w_series = pd.Series(w, index=portfolio['ticker'])
            var_contrib = (w_series * marg_contrib) / port_var * 100
        else:
            var_contrib = pd.Series([0]*len(portfolio), index=portfolio['ticker'])
    else:
        var_contrib = pd.Series([0]*len(portfolio), index=portfolio['ticker'])
    portfolio['VaR_contrib_%'] = var_contrib.values
    st.write("**Вклад каждого актива в общий VaR (приближённо):**")
    var_contrib_display = portfolio[['ticker','weight','VaR_contrib_%']].copy()
    var_contrib_display.columns = ['Тикер','Вес, %','Вклад в VaR, %']
    st.dataframe(var_contrib_display.style.format({'Вес, %':'{:.2f}%','Вклад в VaR, %':'{:.2f}%'}).bar(subset=['Вклад в VaR, %'], color='#ff6f61'))

    # ---------- 2. ДИНАМИКА VAR ----------
    st.subheader("📉 Динамика VaR (скользящее окно 30 дней)")
    port_ret_kzt = calc_metrics(portfolio, prices, usd_kzt)['port_ret_kzt']
    window_var = 30
    if len(port_ret_kzt) > window_var:
        var_series = []
        for i in range(window_var, len(port_ret_kzt)):
            hist = port_ret_kzt.iloc[i-window_var:i]
            mu = hist.mean()
            sigma = hist.std()
            var = -(mu - norm.ppf(0.95)*sigma)
            var_series.append(var)
        dates_var = port_ret_kzt.index[window_var:]
        var_df = pd.DataFrame({'Дата': dates_var, 'VaR (95%)': var_series})
        fig_var = px.line(var_df, x='Дата', y='VaR (95%)', 
                          title='Ежедневный VaR (скользящее окно 30 дней)',
                          labels={'VaR (95%)': 'VaR, %'})
        st.plotly_chart(fig_var, use_container_width=True)
    else:
        st.warning(f"Недостаточно данных для расчёта динамики VaR (нужно > {window_var} дней)")

    # ---------- 3. СРАВНЕНИЕ СТРЕСС-ТЕСТОВ ----------
    st.subheader("🌀 Сравнение стресс-тестов")
    scenarios = {
        "Базовый": (0.0, 0.0),
        "Оптимистичный": (0.10, -0.05),
        "Пессимистичный": (-0.20, 0.15)
    }
    results = {}
    last_prices = prices.iloc[-1]
    last_rate = usd_kzt.iloc[-1]
    w = portfolio['weight'].values/100
    cur_val = 1_000_000
    for name, (ms, fx) in scenarios.items():
        new_prices = last_prices.copy()
        for t in portfolio['ticker']:
            if portfolio[portfolio['ticker']==t]['currency'].values[0] == 'USD':
                new_prices[t] = last_prices[t]*(1+ms)*((last_rate*(1+fx))/last_rate)
            else:
                new_prices[t] = last_prices[t]*(1+ms)
        new_val = (new_prices / last_prices) @ w * cur_val
        loss = cur_val - new_val
        results[name] = loss / cur_val * 100
    df_stress = pd.DataFrame({'Сценарий': list(results.keys()), 'Убыток, %': list(results.values())})
    fig_stress = px.bar(df_stress, x='Сценарий', y='Убыток, %', 
                        title='Потери портфеля по сценариям',
                        color='Убыток, %', color_continuous_scale='RdYlGn_r')
    st.plotly_chart(fig_stress, use_container_width=True)

    # ---------- 4. 3D ГЭП ----------
    st.subheader("📊 3D ГЭП-анализ")
    maturity_bins = ['До востребования', 'До 7 дней', '8-30 дней', '1-3 мес', '3-12 мес', '1-5 лет', 'более 5 лет']
    bins_gap = [0, 0.02, 0.08, 0.25, 1, 5, 100, 1000]
    portfolio_gap = portfolio.copy()
    portfolio_gap['bin'] = pd.cut(portfolio_gap['maturity_years'], bins=bins_gap, labels=maturity_bins, right=False)
    assets_by_bin = portfolio_gap.groupby('bin')['weight'].sum().reindex(maturity_bins, fill_value=0)
    if 'gap_df' in st.session_state and st.session_state.gap_df is not None:
        gap_data = st.session_state.gap_df
        if 'Срок' in gap_data.columns and 'Пассивы (%)' in gap_data.columns:
            liabilities = gap_data.set_index('Срок')['Пассивы (%)']
        else:
            liabilities = pd.Series([5]*7, index=maturity_bins)
    else:
        liabilities = pd.Series([5]*7, index=maturity_bins)
    data_3d = []
    for bin_name in maturity_bins:
        data_3d.append({'Срок': bin_name, 'Тип': 'Активы', 'Значение': assets_by_bin[bin_name]})
        data_3d.append({'Срок': bin_name, 'Тип': 'Пассивы', 'Значение': liabilities[bin_name]})
        data_3d.append({'Срок': bin_name, 'Тип': 'ГЭП', 'Значение': assets_by_bin[bin_name] - liabilities[bin_name]})
    df_3d = pd.DataFrame(data_3d)
    df_3d['Size'] = df_3d['Значение'].abs() + 1
    fig_3d = px.scatter_3d(df_3d, x='Срок', y='Тип', z='Значение',
                           color='Тип', size='Size', size_max=15,
                           title='3D ГЭП: Активы, Пассивы, ГЭП по срокам')
    st.plotly_chart(fig_3d, use_container_width=True)
    st.caption("Для 3D-графика активы взяты из портфеля, пассивы — из текущего ГЭП-анализа.")

# ================================================================
# ВКЛАДКА 3: ЭФФЕКТИВНОСТЬ
# ================================================================
with t_perf:
    st.subheader("📈 Коэффициенты эффективности")
    port_ret = calc_metrics(portfolio, prices, usd_kzt)['port_ret_local']
    periods = {'За месяц':21,'За квартал':63,'За год':252,'С начала':len(port_ret)}
    risk_free = 0.05
    bench_ret = np.random.normal(0.0004,0.015,len(port_ret))
    bench_ret = pd.Series(bench_ret, index=port_ret.index)
    results=[]
    for period_name,days in periods.items():
        if days>len(port_ret): days=len(port_ret)
        ret_subset=port_ret.iloc[-days:]
        bench_subset=bench_ret.iloc[-days:]
        ann_ret=ret_subset.mean()*252
        ann_vol=ret_subset.std()*np.sqrt(252)
        sharpe=(ann_ret-risk_free)/ann_vol if ann_vol>0 else 0
        normal_sharpe=ann_ret/ann_vol if ann_vol>0 else 0
        excess=ret_subset-bench_subset
        tracking_error=excess.std()*np.sqrt(252)
        info_ratio=(ret_subset.mean()-bench_subset.mean())*252/tracking_error if tracking_error>0 else 0
        weighted=(sharpe+normal_sharpe+info_ratio)/3
        results.append({'Период':period_name,'Год. доходность':ann_ret,'Волатильность':ann_vol,'Коэф. Шарпа':sharpe,'Норма Шарпа':normal_sharpe,'Информативность':info_ratio,'Средневзвеш.':weighted})
    df_perf=pd.DataFrame(results)
    st.dataframe(df_perf.style.format({'Год. доходность':'{:.2%}','Волатильность':'{:.2%}','Коэф. Шарпа':'{:.3f}','Норма Шарпа':'{:.3f}','Информативность':'{:.3f}','Средневзвеш.':'{:.3f}'}))

# ================================================================
# ВКЛАДКА 4: КРЕДИТ
# ================================================================
with t_credit:
    st.subheader("💳 Кредитный риск – ожидаемые потери")
    credit_display = df_cr[['ticker','rating','weight','PD','LGD','exposure','expected_loss']].copy()
    credit_display.columns = ['Тикер','Рейтинг','Вес, %','PD, %','LGD, %','Экспозиция, ₸','Ожидаемые потери, ₸']
    st.dataframe(credit_display.style.format({'Вес, %':'{:.2f}%','PD, %':'{:.2f}%','LGD, %':'{:.2f}%','Экспозиция, ₸':'{:,.0f}','Ожидаемые потери, ₸':'{:,.0f}'}))
    st.metric("Суммарные ожидаемые потери", f"{df_cr['expected_loss'].sum():,.0f} ₸")
    st.subheader("ECL по стадиям")
    if 'ecl' in portfolio.columns:
        ecl_by_stage = portfolio.groupby('stage')['ecl'].sum().reset_index()
        st.bar_chart(ecl_by_stage.set_index('stage'))
    else:
        st.info("ECL ещё не рассчитан. Перейдите на вкладку 'Классификация и ОКУ'.")

# ================================================================
# ВКЛАДКА 5: ВАЛЮТНЫЙ РИСК
# ================================================================
with t_currency:
    st.subheader("💱 Валютный риск")
    st.write("Риск изменения стоимости портфеля из-за колебаний курса USD/KZT.")
    current_rate = usd_kzt.iloc[-1]
    col1, col2, col3 = st.columns(3)
    col1.metric("Текущий курс USD/KZT", f"{current_rate:.2f}", delta=f"{current_rate - usd_kzt.iloc[-2]:.2f}" if len(usd_kzt)>1 else None)
    usd_share = portfolio[portfolio['currency']=='USD']['weight'].sum()
    col2.metric("Доля USD в портфеле", f"{usd_share:.2f}%")
    var_contrib_currency = metrics['var95_pct'] * (usd_share / 100)
    col3.metric("Вклад валютного риска в VaR", f"{var_contrib_currency:.2f}%")
    st.subheader("📈 История курса USD/KZT")
    fig_rate = px.line(x=usd_kzt.index, y=usd_kzt.values, title="USD/KZT курс (последние 100 дней)", labels={'x':'Дата','y':'Курс, ₸'})
    st.plotly_chart(fig_rate, use_container_width=True)
    st.subheader("📋 Активы, номинированные в USD")
    usd_assets = portfolio[portfolio['currency']=='USD']
    if not usd_assets.empty:
        st.dataframe(usd_assets[['ticker','weight','current_price']].style.format({'weight':'{:.2f}%','current_price':'{:.2f}'}))
    else:
        st.info("Нет активов в USD.")
    st.caption("Валютный риск учитывается при пересчёте всех USD-активов в тенге.")

# ================================================================
# ВКЛАДКА 6: РЫНОЧНЫЙ РИСК
# ================================================================
with t_market:
    st.subheader("📈 Рыночный риск (ценовой)")
    st.write("Риск изменения стоимости портфеля из-за колебаний рыночных цен активов.")
    st.subheader("📉 Value at Risk (VaR)")
    col1, col2, col3 = st.columns(3)
    col1.metric("Параметрический VaR (95%)", f"{metrics['var95_pct']:.2f}%", delta=f"{metrics['var95_money']:,.0f} ₸")
    col2.metric("Исторический VaR (95%)", f"{metrics['var_hist_pct']:.2f}%", delta=f"{metrics['var_hist_money']:,.0f} ₸")
    col3.metric("Параметрический VaR (99%)", f"{metrics['var99_pct']:.2f}%", delta=f"{metrics['var99_money']:,.0f} ₸")
    st.subheader("🌀 Стресс-тестирование")
    scenario = st.selectbox("Выберите сценарий", ["Без стресса","Падение рынка -10%","Падение рынка -20%","Падение рынка -30%","Кризис 2008 (-50%)"], key="stress_market")
    shock_map = {"Без стресса":0.0,"Падение рынка -10%":-0.10,"Падение рынка -20%":-0.20,"Падение рынка -30%":-0.30,"Кризис 2008 (-50%)":-0.50}
    shock = shock_map[scenario]
    last_prices = prices.iloc[-1]
    new_prices = last_prices * (1 + shock)
    w = portfolio['weight'].values/100
    cur_val = 1_000_000
    new_val = (new_prices / last_prices) @ w * cur_val
    loss = cur_val - new_val
    col1, col2 = st.columns(2)
    col1.metric("Потери при стрессе", f"{-loss/cur_val:.2%}", delta=f"{-loss:,.0f} ₸")
    col2.metric("Остаточная стоимость", f"{new_val:,.0f} ₸")
    st.subheader("🔁 Краткий бэк-тест")
    port_ret = metrics['port_ret_kzt']
    window = 30
    if len(port_ret) > window:
        var_forecast=[]; actual_loss=[]; 
        for i in range(window, len(port_ret)):
            hist = port_ret.iloc[i-window:i]
            mu=hist.mean(); sigma=hist.std()
            var = -(mu - norm.ppf(0.95)*sigma)
            var_forecast.append(var); actual_loss.append(-port_ret.iloc[i])
        exceed=np.sum(np.array(actual_loss)>np.array(var_forecast))
        total=len(actual_loss)
        st.metric("Превышений VaR (30 дней)", f"{exceed}/{total}", delta=f"{exceed/total:.2%} (ожидается ~5%)")
    else:
        st.info("Недостаточно данных для бэк-теста.")
    st.caption("Рыночный риск измеряется через VaR, стресс-тесты и бэк-тестирование.")

# ================================================================
# ВКЛАДКА 7: ПРОЦЕНТНЫЙ РИСК
# ================================================================
with t_interest:
    st.subheader("📊 Процентный риск")
    st.write("Риск изменения стоимости портфеля из-за колебаний процентных ставок.")
    st.subheader("📋 ГЭП-анализ (разрыв по срокам)")
    if 'gap_df' in st.session_state and st.session_state.gap_df is not None:
        gap_data = st.session_state.gap_df
        st.dataframe(gap_data.style.format({'Активы (%)':'{:.2f}%','Пассивы (%)':'{:.2f}%','ГЭП (%)':'{:.2f}%'}))
        total_gap = gap_data['ГЭП (%)'].abs().sum()
        st.metric("Суммарный абсолютный ГЭП", f"{total_gap:.2f}%")
        col1,col2 = st.columns(2)
        rate_shock = 0.01
        interest_risk = total_gap * rate_shock
        col1.metric("Оценка убытка при росте ставки на 1%", f"{interest_risk:.2f}%")
        rate_shock_2 = 0.02
        interest_risk_2 = total_gap * rate_shock_2
        col2.metric("Оценка убытка при росте ставки на 2%", f"{interest_risk_2:.2f}%")
        st.subheader("📈 График ГЭП по срокам")
        fig_gap = go.Figure()
        fig_gap.add_trace(go.Bar(x=gap_data['Срок'], y=gap_data['Активы (%)'], name='Активы'))
        fig_gap.add_trace(go.Bar(x=gap_data['Срок'], y=gap_data['Пассивы (%)'], name='Пассивы'))
        fig_gap.add_trace(go.Scatter(x=gap_data['Срок'], y=gap_data['ГЭП (%)'], name='ГЭП', mode='lines+markers'))
        fig_gap.update_layout(title='ГЭП по срокам', xaxis_title='Срок', yaxis_title='% портфеля')
        st.plotly_chart(fig_gap, use_container_width=True)
    else:
        st.info("Сначала заполните пассивы во вкладке 'ГЭП (расшир.)'")
        example_gap = pd.DataFrame({
            'Срок': ['До востребования','До 7 дней','8-30 дней','1-3 мес','3-12 мес','1-5 лет','более 5 лет'],
            'Активы (%)': [0,0,0,5,10,60,25],
            'Пассивы (%)': [5,5,5,5,5,5,5]
        })
        example_gap['ГЭП (%)'] = example_gap['Активы (%)'] - example_gap['Пассивы (%)']
        st.dataframe(example_gap.style.format({'Активы (%)':'{:.2f}%','Пассивы (%)':'{:.2f}%','ГЭП (%)':'{:.2f}%'}))
    st.subheader("📊 Чувствительность портфеля к ставке")
    # Добавим расчёт дюрации
    mod_duration, price_change = calculate_duration_and_sensitivity(portfolio, prices, usd_kzt)
    if mod_duration is not None:
        st.metric("Модифицированная дюрация", f"{mod_duration:.2f} лет")
        st.metric("Изменение стоимости при росте ставки на 1%", f"{price_change:,.0f} ₸")
    else:
        st.info("Нет данных для расчёта дюрации (нужны облигации с maturity_years).")
    st.caption("Процентный риск оценивается через ГЭП-анализ, дюрацию и анализ чувствительности.")

# ================================================================
# ВКЛАДКА 8: СТРАНОВОЙ РИСК
# ================================================================
with t_country:
    st.subheader("🌍 Страновой риск")
    st.dataframe(df_country.style.format({'Доля в портфеле (%)':'{:.2f}%','Вклад в страновой риск':'{:.3f}'}))
    st.metric("Суммарный страновой риск", f"{total_country_risk:.3f}")
    fig = px.bar(df_country, x='Страна', y='Вклад в страновой риск', color='Рейтинг риска (1-5)', title='Страновой риск по странам')
    st.plotly_chart(fig, use_container_width=True)

# ================================================================
# ВКЛАДКА 9: ГЭП (расшир.)
# ================================================================
with t_gap:
    st.subheader("📊 Расширенный ГЭП-анализ")
    maturity_bins = ['До востребования','До 7 дней','8-30 дней','1-3 мес','3-12 мес','1-5 лет','более 5 лет']
    labels=maturity_bins
    bins_gap=[0,0.02,0.08,0.25,1,5,100,1000]
    portfolio_gap=portfolio.copy()
    portfolio_gap['bin']=pd.cut(portfolio_gap['maturity_years'], bins=bins_gap, labels=labels, right=False)
    assets_by_bin=portfolio_gap.groupby('bin')['weight'].sum().reindex(labels, fill_value=0).to_dict()
    liabilities={}
    col_liab=st.columns(len(maturity_bins))
    for i,bin_name in enumerate(maturity_bins):
        with col_liab[i]:
            liabilities[bin_name]=st.number_input(f"{bin_name}", min_value=0.0, max_value=100.0, value=5.0, step=0.5, key=f"liab_{i}")
    gap_df=pd.DataFrame({'Срок':maturity_bins,'Активы (%)':[assets_by_bin.get(b,0) for b in maturity_bins],'Пассивы (%)':[liabilities[b] for b in maturity_bins]})
    gap_df['ГЭП (%)']=gap_df['Активы (%)']-gap_df['Пассивы (%)']
    st.session_state.gap_df = gap_df
    st.table(gap_df.style.format({'Активы (%)':'{:.2f}%','Пассивы (%)':'{:.2f}%','ГЭП (%)':'{:.2f}%'}))
    fig=go.Figure()
    fig.add_trace(go.Bar(x=gap_df['Срок'], y=gap_df['Активы (%)'], name='Активы'))
    fig.add_trace(go.Bar(x=gap_df['Срок'], y=gap_df['Пассивы (%)'], name='Пассивы'))
    fig.add_trace(go.Scatter(x=gap_df['Срок'], y=gap_df['ГЭП (%)'], name='ГЭП', mode='lines+markers'))
    fig.update_layout(title='ГЭП по срокам', xaxis_title='Срок', yaxis_title='% портфеля')
    st.plotly_chart(fig, use_container_width=True)

# ================================================================
# ВКЛАДКА 10: БЭК-ТЕСТ
# ================================================================
with t_backtest:
    st.subheader("🔁 Бэк-тестирование VaR (95%)")
    port_ret = calc_metrics(portfolio, prices, usd_kzt)['port_ret_kzt']
    max_window = len(port_ret)-1
    if max_window < 20:
        st.warning("Недостаточно данных для бэк-теста (нужно минимум 21 день)")
    else:
        default_window = min(60, max_window)
        window = st.slider("Глубина окна (дней)", 20, max_window, default_window, key="backtest_window")
        var_forecast=[]; actual_loss=[]; dates_test=port_ret.index[window:]
        for i in range(window, len(port_ret)):
            hist=port_ret.iloc[i-window:i]
            mu=hist.mean(); sigma=hist.std()
            var=-(mu-norm.ppf(0.95)*sigma)
            var_forecast.append(var); actual_loss.append(-port_ret.iloc[i])
        exceed=np.sum(np.array(actual_loss)>np.array(var_forecast))
        total=len(actual_loss)
        st.metric("Превышений", f"{exceed}/{total}", f"{exceed/total:.2%} (ожидается ~5%)")
        df_bt=pd.DataFrame({'Дата':dates_test,'Прогноз VaR':var_forecast,'Факт.убыток':actual_loss})
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=df_bt['Дата'], y=df_bt['Прогноз VaR'], mode='lines', name='VaR прогноз'))
        fig.add_trace(go.Scatter(x=df_bt['Дата'], y=df_bt['Факт.убыток'], mode='markers', name='Факт.убыток'))
        fig.update_layout(title='Бэк-тест VaR', xaxis_title='Дата', yaxis_title='Убыток')
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df_bt.style.format({'Прогноз VaR':'{:.4f}','Факт.убыток':'{:.4f}'}))

# ================================================================
# ВКЛАДКА 11: ВИРТУАЛЬНЫЙ ПОРТФЕЛЬ
# ================================================================
with t_virtual:
    st.subheader("🧪 Виртуальный портфель – добавление/удаление позиций")
    with st.form("add_virtual"):
        col1,col2,col3=st.columns(3)
        new_ticker=col1.text_input("Тикер","NEW")
        new_currency=col2.selectbox("Валюта",["USD","KZT"])
        new_weight=col3.number_input("Вес (%)",0.0,100.0,1.0)
        if st.form_submit_button("Добавить"):
            if new_ticker not in portfolio['ticker'].values:
                new_row=pd.DataFrame({'ticker':[new_ticker],'currency':[new_currency],'weight':[new_weight],'rating':['BBB'],'sector':['Other'],'country':['US'],'asset_type':['Акции'],'maturity_years':[2],'price_buy':[100],'current_price':[100],'issue_volume':[1e6]})
                portfolio=pd.concat([portfolio,new_row], ignore_index=True)
                portfolio['weight']=portfolio['weight']/portfolio['weight'].sum()*100
                portfolio['weight']=np.round(portfolio['weight'],2)
                st.session_state.portfolio=portfolio
                st.success("Добавлено")
                st.rerun()
    with st.form("remove_virtual"):
        ticker_to_remove=st.selectbox("Выберите тикер для удаления", portfolio['ticker'].tolist())
        if st.form_submit_button("Удалить"):
            portfolio=portfolio[portfolio['ticker']!=ticker_to_remove].reset_index(drop=True)
            portfolio['weight']=portfolio['weight']/portfolio['weight'].sum()*100
            portfolio['weight']=np.round(portfolio['weight'],2)
            st.session_state.portfolio=portfolio
            st.success("Удалено")
            st.rerun()
    virtual_display=portfolio[['ticker','currency','weight','rating','sector']].copy()
    virtual_display.columns=['Тикер','Валюта','Вес, %','Рейтинг','Сектор']
    st.dataframe(virtual_display.style.format({'Вес, %':'{:.2f}%'}))
    with st.spinner("Пересчёт..."):
        new_metrics=calc_metrics(portfolio, prices, usd_kzt)
        st.metric("Волатильность (тенге)", f"{new_metrics['vol_kzt']:.2%}")
        st.metric("VaR 95%", f"{new_metrics['var95_pct']:.2f}%", f"{new_metrics['var95_money']:,.0f} ₸")

# ================================================================
# ВКЛАДКА 12: ЛИМИТЫ (ВСЕ)
# ================================================================
with t_limits:
    st.subheader("⚖️ Все лимиты (с лимитной сеткой по рейтингам)")
    limits = get_limits()
    total = portfolio['weight'].sum()
    cur_asset = portfolio.groupby('asset_type')['weight'].sum() / total * 100
    cur_curr = portfolio.groupby('currency')['weight'].sum() / total * 100
    cur_sector = portfolio.groupby('sector')['weight'].sum() / total * 100
    cur_country = portfolio.groupby('country')['weight'].sum() / total * 100

    def check_limit(curr, lim):
        violations={}
        for k,v in lim.items():
            if k in curr.index and curr[k] > v:
                violations[k]=(curr[k], v)
        return violations

    v_asset=check_limit(cur_asset, limits['asset_type'])
    v_curr=check_limit(cur_curr, limits['currency'])
    v_sector=check_limit(cur_sector, limits['sector'])
    v_country=check_limit(cur_country, limits['country'])
    st.subheader("Общие лимиты")
    if any([v_asset,v_curr,v_sector,v_country]):
        st.error("Нарушения общих лимитов:")
        for d in [v_asset,v_curr,v_sector,v_country]:
            for k,(fact,lim) in d.items():
                st.write(f"- {k}: факт {fact:.2f}% > лимит {lim}% (превышение {fact-lim:.2f}%)")
    else:
        st.success("Все общие лимиты соблюдены.")

    port_value = 1_000_000
    st.subheader("Свободные лимиты (в денежном выражении)")
    free_asset = {k: max(0, v - cur_asset.get(k,0)) / 100 * port_value for k,v in limits['asset_type'].items()}
    free_curr = {k: max(0, v - cur_curr.get(k,0)) / 100 * port_value for k,v in limits['currency'].items()}
    free_sector = {k: max(0, v - cur_sector.get(k,0)) / 100 * port_value for k,v in limits['sector'].items()}
    free_country = {k: max(0, v - cur_country.get(k,0)) / 100 * port_value for k,v in limits['country'].items()}
    st.write("**По типам активов:**", {k: f"{v:,.0f} ₸" for k,v in free_asset.items()})
    st.write("**По валютам:**", {k: f"{v:,.0f} ₸" for k,v in free_curr.items()})
    st.write("**По секторам:**", {k: f"{v:,.0f} ₸" for k,v in free_sector.items()})
    st.write("**По странам:**", {k: f"{v:,.0f} ₸" for k,v in free_country.items()})

    st.subheader("Лимиты на эмитентов (с лимитной сеткой по рейтингу)")
    port_limits = portfolio.copy()
    port_limits['limit_rating'] = port_limits['rating'].apply(get_rating_limit)
    port_limits['excess'] = port_limits['weight'] - port_limits['limit_rating']
    port_limits['limit_ok'] = port_limits['excess'] <= 0
    limits_display = port_limits[['ticker','rating','weight','limit_rating','excess','limit_ok']].copy()
    limits_display.columns = ['Тикер','Рейтинг','Вес, %','Лимит, %','Превышение, %','ОК']
    st.dataframe(limits_display.style.format({'Вес, %':'{:.2f}%','Лимит, %':'{:.2f}%','Превышение, %':'{:.2f}%'}).map(lambda x: 'background-color: red' if x==False else '', subset=['ОК']))
    st.write("**Свободные лимиты по эмитентам:**")
    for _, row in port_limits.iterrows():
        free_money = max(0, (row['limit_rating'] - row['weight']) / 100 * port_value)
        st.write(f"- {row['ticker']}: {free_money:,.0f} ₸")

    st.subheader("Лимит на % от эмиссии")
    portfolio['position_value'] = portfolio['weight']/100 * port_value
    portfolio['share_of_issue'] = portfolio['position_value'] / portfolio['issue_volume'] * 100
    limit_issue = 5
    portfolio['issue_ok'] = portfolio['share_of_issue'] <= limit_issue
    issue_display = portfolio[['ticker','weight','position_value','issue_volume','share_of_issue','issue_ok']].copy()
    issue_display.columns = ['Тикер','Вес, %','Стоимость позиции, ₸','Объём эмиссии, ₸','Доля от эмиссии, %','ОК']
    st.dataframe(issue_display.style.format({'Вес, %':'{:.2f}%','Стоимость позиции, ₸':'{:,.0f}','Объём эмиссии, ₸':'{:,.0f}','Доля от эмиссии, %':'{:.2f}%'}).map(lambda x: 'background-color: red' if x==False else '', subset=['ОК']))
    st.write("**Свободный лимит по эмиссии:**")
    for _, row in portfolio.iterrows():
        max_invest = row['issue_volume'] * limit_issue / 100
        free_issue = max(0, max_invest - row['position_value'])
        st.write(f"- {row['ticker']}: {free_issue:,.0f} ₸")

    st.subheader("Лимит на ГЭП-позиции")
    if 'gap_df' in st.session_state and st.session_state.gap_df is not None:
        gap_abs = st.session_state.gap_df['ГЭП (%)'].abs().sum()
    else:
        gap_abs = 0
    limit_gap = 20
    if gap_abs > limit_gap:
        st.error(f"Нарушение лимита ГЭП: суммарный ГЭП = {gap_abs:.2f}% > {limit_gap}%")
    else:
        st.success(f"Лимит ГЭП соблюдён: {gap_abs:.2f}% <= {limit_gap}%")

    st.subheader("Лимит НВА (высоколиквидные активы)")
    hva = portfolio[(portfolio['rating']=='AAA') & (portfolio['maturity_years']<1)]
    hva_share = hva['weight'].sum()
    limit_hva = 30
    if hva_share < limit_hva:
        st.warning(f"НВА = {hva_share:.2f}% < {limit_hva}% (нарушение)")
    else:
        st.success(f"НВА = {hva_share:.2f}% >= {limit_hva}% (соблюдено)")

    # ---------- РАСШИРЕННЫЕ НАСТРОЙКИ ЛИМИТОВ ----------
    with st.expander("⚙️ Настройка лимитов"):
        st.write("Измените значения лимитов для валют, секторов и стран.")
        col1, col2 = st.columns(2)
        with col1:
            new_usd_limit = st.number_input("Лимит USD (%)", min_value=0, max_value=100, value=60, key="usd_limit")
            new_energy_limit = st.number_input("Лимит Energy (%)", min_value=0, max_value=100, value=25, key="energy_limit")
        with col2:
            new_tech_limit = st.number_input("Лимит Tech (%)", min_value=0, max_value=100, value=30, key="tech_limit")
            new_us_country_limit = st.number_input("Лимит US (%)", min_value=0, max_value=100, value=50, key="us_country_limit")
        if st.button("Применить новые лимиты"):
            st.session_state.limits_override = {
                'currency': {'USD': new_usd_limit, 'KZT': 70},
                'sector': {'Tech': new_tech_limit, 'Energy': new_energy_limit, 'Finance': 40, 'Gov': 20, 'Consumer': 30, 'Other': 20},
                'country': {'US': new_us_country_limit, 'KZ': 40, 'EU': 30, 'CN': 20}
            }
            st.success("✅ Лимиты обновлены")
            st.rerun()

# ================================================================
# ВКЛАДКА 13: STOP-LOSS
# ================================================================
with t_stoploss:
    st.subheader("🛑 Stop-loss / Take-profit")
    if 'stop_loss' not in st.session_state: st.session_state.stop_loss={}
    if 'take_profit' not in st.session_state: st.session_state.take_profit={}
    cols=st.columns(min(len(portfolio),4))
    for i,(idx,row) in enumerate(portfolio.iterrows()):
        ticker=row['ticker']
        with cols[i%len(cols)]:
            st.write(f"**{ticker}**")
            current_price=row['current_price']; buy_price=row['price_buy']
            sl=st.number_input(f"SL {ticker} %",0.0,50.0,5.0,0.5,key=f"sl_{ticker}_{i}")
            tp=st.number_input(f"TP {ticker} %",0.0,50.0,10.0,0.5,key=f"tp_{ticker}_{i}")
            st.session_state.stop_loss[ticker]=sl; st.session_state.take_profit[ticker]=tp
            price_change=(current_price-buy_price)/buy_price*100
            if price_change <= -sl: st.error(f"❌ Stop-loss сработал! ({price_change:.2f}%)")
            elif price_change >= tp: st.success(f"✅ Take-profit сработал! ({price_change:.2f}%)")
            else: st.info(f"Изменение: {price_change:.2f}%")
    sl_df=portfolio.copy()
    sl_df['SL_%']=sl_df['ticker'].map(lambda x: st.session_state.stop_loss.get(x,5.0))
    sl_df['TP_%']=sl_df['ticker'].map(lambda x: st.session_state.take_profit.get(x,10.0))
    sl_df['изменение_%']=(sl_df['current_price']-sl_df['price_buy'])/sl_df['price_buy']*100
    sl_df['SL_нарушено']=sl_df['изменение_%'] <= -sl_df['SL_%']
    sl_df['TP_достигнут']=sl_df['изменение_%'] >= sl_df['TP_%']
    st.subheader("Сводка")
    sl_display=sl_df[['ticker','price_buy','current_price','изменение_%','SL_%','TP_%','SL_нарушено','TP_достигнут']].copy()
    sl_display.columns=['Тикер','Цена покупки','Текущая цена','Изменение, %','SL, %','TP, %','SL сработал','TP достигнут']
    st.dataframe(sl_display.style.format({'Цена покупки':'{:.2f}','Текущая цена':'{:.2f}','Изменение, %':'{:.2f}%','SL, %':'{:.2f}%','TP, %':'{:.2f}%'}).map(lambda x: 'background-color: red' if x==True else '', subset=['SL сработал']).map(lambda x: 'background-color: green' if x==True else '', subset=['TP достигнут']))

# ================================================================
# ВКЛАДКА 14: ЗАКЛЮЧЕНИЕ (ДАШБОРД)
# ================================================================
with t_conclusion:
    st.subheader("📊 Дашборд риск-менеджмента")
    st.caption(f"Актуальное состояние портфеля на {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    col1,col2,col3,col4=st.columns(4)
    col1.metric("📉 VaR 95%", f"{metrics['var95_pct']:.2f}%")
    total_ecl=df_cr['expected_loss'].sum() if 'df_cr' in locals() else 0
    col2.metric("💰 Ожидаемые потери (ECL)", f"{total_ecl:,.0f} ₸")
    violations_list=[]
    for d in [v_asset, v_curr, v_sector, v_country]:
        for k,(fact,lim) in d.items():
            violations_list.append(f"{k}: {fact:.2f}% > {lim}%")
    col3.metric("🚨 Нарушений", len(violations_list))
    col4.metric("🌀 Стресс-убыток", f"{-loss/cur_val:.2%}" if 'loss' in locals() else "0%")
    st.divider()
    col_left,col_right=st.columns(2)
    with col_left:
        st.subheader("📈 Динамика VaR (30 дней)")
        st.info("История VaR будет доступна после накопления данных")
    with col_right:
        st.subheader("📊 ECL по стадиям")
        if 'ecl' in portfolio.columns:
            ecl_by_stage=portfolio.groupby('stage')['ecl'].sum()
            st.bar_chart(ecl_by_stage)
        else:
            st.info("ECL не рассчитан")
    st.divider()
    st.subheader("⚠️ Нарушения лимитов")
    if violations_list:
        for v in violations_list:
            st.error(f"• {v}")
    else:
        st.success("✅ Все лимиты соблюдены")
    with st.expander("📋 Детали портфеля"):
        st.dataframe(portfolio[['ticker','weight','rating','stage']])
    if st.button("📥 Скачать отчёт (TXT)"):
        report_text = f"ОТЧЕТ РИСК-МЕНЕДЖМЕНТА\n{datetime.now().strftime('%Y-%m-%d %H:%M')}\nVaR 95%: {metrics['var95_pct']:.2f}%\nECL: {total_ecl:,.0f} ₸\nНарушений: {len(violations_list)}"
        st.download_button("Скачать", report_text, file_name=f"report_{datetime.now().strftime('%Y%m%d')}.txt")

    # ---------- РИСК-АППЕТИТ (RAS) ----------
    st.subheader("🎯 Риск-аппетит (целевые уровни)")
    ras_targets = {
        "VaR 95% (дневной)": {"target": 2.0, "warning": 1.5},
        "Концентрация на эмитента": {"target": 5.0, "warning": 4.0},
        "Доля активов с рейтингом ниже BBB-": {"target": 15.0, "warning": 12.0},
        "Доля высоколиквидных активов (HQLA)": {"target": 20.0, "warning": 25.0},
    }
    current_var = metrics['var95_pct']
    max_concentration = portfolio['weight'].max()
    low_rating_share = portfolio[portfolio['rating'].isin(['BB', 'B'])]['weight'].sum()
    hva_share = portfolio[(portfolio['rating']=='AAA') & (portfolio['maturity_years']<1)]['weight'].sum()
    ras_data = {
        "Показатель": list(ras_targets.keys()),
        "Текущее значение": [
            f"{current_var:.2f}%",
            f"{max_concentration:.2f}%",
            f"{low_rating_share:.2f}%",
            f"{hva_share:.2f}%"
        ],
        "Целевое": [f"{v['target']}%" for v in ras_targets.values()],
        "Предупреждение": [f"{v['warning']}%" for v in ras_targets.values()],
        "Статус": []
    }
    for i, key in enumerate(ras_targets.keys()):
        if key == "VaR 95% (дневной)": val = current_var
        elif key == "Концентрация на эмитента": val = max_concentration
        elif key == "Доля активов с рейтингом ниже BBB-": val = low_rating_share
        else: val = hva_share
        target = ras_targets[key]['target']
        warning = ras_targets[key]['warning']
        if key == "Доля высоколиквидных активов (HQLA)":
            if val >= target: ras_data["Статус"].append("🟢")
            elif val >= warning: ras_data["Статус"].append("🟡")
            else: ras_data["Статус"].append("🔴")
        else:
            if val <= target: ras_data["Статус"].append("🟢")
            elif val <= warning: ras_data["Статус"].append("🟡")
            else: ras_data["Статус"].append("🔴")
    st.table(pd.DataFrame(ras_data))

# ================================================================
# ВКЛАДКА 15: КЛАССИФИКАЦИЯ И ОКУ
# ================================================================
with t_classification:
    st.subheader("📋 Классификация активов и расчёт ECL")
    st.write("Установите стадию, макро-коэффициент, SPPI и бизнес-модель для каждого актива.")
    edited_df = st.data_editor(
        portfolio[['ticker','rating','stage','macro_k','sppi','business_model','EIR']],
        num_rows="dynamic",
        use_container_width=True
    )
    if st.button("🔄 Пересчитать ECL"):
        for col in ['stage','macro_k','sppi','business_model','EIR']:
            portfolio[col] = edited_df[col]
        portfolio['ecl'] = portfolio.apply(calculate_ECL, axis=1)
        st.session_state.portfolio = portfolio
        st.success("✅ ECL пересчитан!")
        st.dataframe(portfolio[['ticker','stage','ecl']])
    st.subheader("Распределение активов по стадиям")
    stage_counts = portfolio['stage'].value_counts().sort_index()
    st.bar_chart(stage_counts)

# ================================================================
# ВКЛАДКА 16: KRI
# ================================================================
with t_kri:
    st.subheader("📊 Ключевые индикаторы риска (KRI)")
    kri_data={
        "Индикатор": ["VaR 95%", "Концентрация", "Доля Стадии 2+3", "Инциденты/квартал"],
        "Текущее": [f"{metrics['var95_pct']:.2f}%", f"{portfolio['weight'].max():.2f}%", f"{(portfolio['stage']>1).mean()*100:.2f}%", str(len(st.session_state.get('incidents',pd.DataFrame())))],
        "Зелёная": ["< 3%", "< 30%", "< 10%", "0-1"],
        "Жёлтая": ["3-5%", "30-40%", "10-20%", "2-3"],
        "Красная": ["> 5%", "> 40%", "> 20%", "> 3"]
    }
    st.table(pd.DataFrame(kri_data))

# ================================================================
# ВКЛАДКА 17: ИНЦИДЕНТЫ
# ================================================================
with t_incidents:
    st.subheader("📝 Регистрация операционных инцидентов")
    with st.form("add_incident"):
        desc=st.text_input("Описание инцидента")
        cat=st.selectbox("Категория",["Ошибка персонала","Сбой ИТ","Внешнее событие","Мошенничество","Другое"])
        loss=st.number_input("Ущерб (тенге)",0.0,step=1000.0)
        measures=st.text_area("Принятые меры")
        if st.form_submit_button("➕ Добавить инцидент"):
            new_row=pd.DataFrame({'Дата':[datetime.now().strftime('%Y-%m-%d %H:%M')],'Описание':[desc],'Категория':[cat],'Ущерб':[loss],'Меры':[measures]})
            st.session_state.incidents=pd.concat([st.session_state.incidents,new_row], ignore_index=True)
            st.success("✅ Инцидент зарегистрирован")
            st.rerun()
    if not st.session_state.incidents.empty:
        st.dataframe(st.session_state.incidents)
        csv=st.session_state.incidents.to_csv(index=False).encode('utf-8-sig')
        st.download_button("📥 Скачать инциденты (CSV)", csv, "incidents.csv", "text/csv")

# ================================================================
# ВКЛАДКА 18: РЕГИСТР РИСКОВ
# ================================================================
with t_risk_register:
    st.subheader("📋 Регистр рисков (интегрированный)")
    st.write("Автоматически обновляемый регистр с привязкой к текущим количественным оценкам.")
    if st.session_state.risk_register.empty:
        st.session_state.risk_register = pd.DataFrame({
            "ID": [1, 2, 3, 4, 5],
            "Наименование риска": [
                "Рыночный риск (ценовой)",
                "Валютный риск",
                "Процентный риск",
                "Кредитный риск",
                "Операционный риск"
            ],
            "Категория": ["Рыночный", "Рыночный", "Рыночный", "Кредитный", "Операционный"],
            "Владелец": ["Департамент управления активами"] * 5,
            "Текущий уровень (количественный)": [
                f"{metrics['var95_pct']:.2f}%",
                f"{metrics['var95_pct'] * (portfolio['currency']=='USD').mean():.2f}%",
                "Не рассчитан",
                f"{df_cr['expected_loss'].sum():,.0f} ₸",
                f"{len(st.session_state.get('incidents', pd.DataFrame()))} инцидентов"
            ],
            "Оценка (приоритет)": ["Высокий", "Высокий", "Средний", "Высокий", "Средний"],
            "Статус": ["Активен"] * 5
        })
    st.dataframe(st.session_state.risk_register, use_container_width=True)
    if st.button("🔄 Обновить регистр из текущих данных"):
        df_reg = st.session_state.risk_register.copy()
        df_reg.loc[df_reg['Наименование риска'] == "Рыночный риск (ценовой)", "Текущий уровень (количественный)"] = f"{metrics['var95_pct']:.2f}%"
        df_reg.loc[df_reg['Наименование риска'] == "Валютный риск", "Текущий уровень (количественный)"] = f"{metrics['var95_pct'] * (portfolio['currency']=='USD').mean():.2f}%"
        df_reg.loc[df_reg['Наименование риска'] == "Кредитный риск", "Текущий уровень (количественный)"] = f"{df_cr['expected_loss'].sum():,.0f} ₸"
        df_reg.loc[df_reg['Наименование риска'] == "Операционный риск", "Текущий уровень (количественный)"] = f"{len(st.session_state.get('incidents', pd.DataFrame()))} инцидентов"
        st.session_state.risk_register = df_reg
        st.success("✅ Регистр обновлен")
        st.rerun()

# ================================================================
# ВКЛАДКА 19: КАЧЕСТВЕННАЯ ОЦЕНКА РИСКОВ
# ================================================================
with t_risk_assessment:
    st.subheader("📋 Качественная оценка рисков")
    st.write("Оцените риски по шкале вероятности (1-5) и влияния (1-5).")
    if st.session_state.risk_assessment.empty:
        risks = [
            "Рыночный риск (ценовой)",
            "Валютный риск",
            "Процентный риск",
            "Кредитный риск",
            "Операционный риск",
            "Риск ликвидности"
        ]
        st.session_state.risk_assessment = pd.DataFrame({
            "Риск": risks,
            "Вероятность (1-5)": [3] * len(risks),
            "Влияние (1-5)": [3] * len(risks)
        })
    edited_risks = st.data_editor(
        st.session_state.risk_assessment,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "Вероятность (1-5)": st.column_config.NumberColumn("Вероятность (1-5)", min_value=1, max_value=5, step=1),
            "Влияние (1-5)": st.column_config.NumberColumn("Влияние (1-5)", min_value=1, max_value=5, step=1)
        }
    )
    if st.button("Сохранить оценку"):
        st.session_state.risk_assessment = edited_risks
        st.success("✅ Оценка сохранена")
    st.subheader("📊 Матрица рисков")
    if not st.session_state.risk_assessment.empty:
        st.session_state.risk_assessment["Уровень"] = (
            st.session_state.risk_assessment["Вероятность (1-5)"] * 
            st.session_state.risk_assessment["Влияние (1-5)"]
        )
        def color_risk(val):
            if val >= 20:
                return "background-color: #ffcccc"
            elif val >= 12:
                return "background-color: #ffffcc"
            else:
                return "background-color: #ccffcc"
        st.dataframe(
            st.session_state.risk_assessment.style.map(color_risk, subset=["Уровень"])
        )

# ================================================================
# SIDEBAR (ПОЧТА, МАКРО, ЗАГРУЗКА, ОБНОВЛЕНИЕ)
# ================================================================
st.sidebar.markdown("---")
st.sidebar.subheader("📧 Почтовое уведомление")
with st.sidebar.expander("⚙️ Настройки почты"):
    st.session_state.email_settings['smtp_server'] = st.text_input("SMTP сервер", value=st.session_state.email_settings['smtp_server'])
    st.session_state.email_settings['smtp_port'] = st.text_input("Порт", value=str(st.session_state.email_settings['smtp_port']))
    st.session_state.email_settings['sender_email'] = st.text_input("Почта отправителя", value=st.session_state.email_settings['sender_email'])
    st.session_state.email_settings['sender_password'] = st.text_input("Пароль", type="password", value=st.session_state.email_settings['sender_password'])
    st.session_state.email_settings['recipient_email'] = st.text_input("Почта получателя", value=st.session_state.email_settings['recipient_email'])
    if st.button("📤 Отправить тестовое письмо"):
        success, msg = send_email_alert("Тестовое сообщение", "✅ Это тестовое письмо. Ваша система риск-менеджмента работает корректно.")
        if success:
            st.sidebar.success("✅ " + msg)
        else:
            st.sidebar.error("❌ " + msg)

st.sidebar.markdown("---")
st.sidebar.subheader("🌍 Макро-сценарий")
scenario = st.sidebar.selectbox("Выберите сценарий", ["Базовый","Оптимистичный","Пессимистичный"], key="macro_scenario")
macro_map = {"Базовый":1.0,"Оптимистичный":0.8,"Пессимистичный":1.5}
st.session_state.macro_k = macro_map[scenario]
portfolio['macro_k'] = st.session_state.macro_k

st.sidebar.markdown("---")
st.sidebar.subheader("📂 Загрузка данных (CSV / Excel)")
with st.sidebar.expander("⚙️ Загрузить файлы", expanded=False):
    uploaded_portfolio = st.file_uploader("Портфель (CSV / Excel)", type=["csv","xlsx"], key="upload_portfolio_ext")
    uploaded_prices = st.file_uploader("История цен (CSV / Excel)", type=["csv","xlsx"], key="upload_prices_ext")
    uploaded_rates = st.file_uploader("Курс USD/KZT (CSV / Excel)", type=["csv","xlsx"], key="upload_rates_ext")
    uploaded_macro = st.file_uploader("Макро-коэффициенты (CSV / Excel)", type=["csv","xlsx"], key="upload_macro_ext")
    if st.button("📥 Применить загруженные данные"):
        try:
            if uploaded_portfolio is not None:
                if uploaded_portfolio.name.endswith('.csv'):
                    df_port = pd.read_csv(uploaded_portfolio, encoding='utf-8-sig', header=0, sep=',', engine='python', skipinitialspace=True)
                else:
                    df_port = pd.read_excel(uploaded_portfolio, sheet_name=0, engine='openpyxl')
                df_port.columns = df_port.columns.str.strip()
                required_cols = ['ticker','currency','weight','rating','sector','country','asset_type','maturity_years','price_buy','current_price','issue_volume']
                if all(col in df_port.columns for col in required_cols):
                    for col in ['stage','macro_k','sppi','business_model','EIR','LGD_case_specific','ecl']:
                        if col not in df_port.columns:
                            if col=='stage': df_port[col]=1
                            elif col=='macro_k': df_port[col]=1.0
                            elif col=='sppi': df_port[col]=True
                            elif col=='business_model': df_port[col]='БМ-1'
                            elif col=='EIR': df_port[col]=0.05
                            elif col=='LGD_case_specific': df_port[col]=0.5
                            elif col=='ecl': df_port[col]=0.0
                    rating_map={'AAA':(0.01,0.30),'AA':(0.05,0.35),'A':(0.10,0.40),'BBB':(0.20,0.45),'BB':(0.50,0.50),'B':(1.00,0.60)}
                    df_port['PD']=df_port['rating'].map(lambda r: rating_map.get(r,(0.5,0.5))[0]/100)
                    df_port['LGD']=df_port['rating'].map(lambda r: rating_map.get(r,(0.5,0.5))[1])
                    df_port['exposure']=df_port['weight']/100*1_000_000
                    st.session_state.portfolio = df_port
                    st.sidebar.success("✅ Портфель загружен")
                    st.rerun()
                else:
                    st.sidebar.error(f"❌ Отсутствуют колонки: {set(required_cols)-set(df_port.columns)}")
            if uploaded_prices is not None:
                if uploaded_prices.name.endswith('.csv'):
                    df_prices=pd.read_csv(uploaded_prices, index_col=0, parse_dates=True)
                else:
                    df_prices=pd.read_excel(uploaded_prices, index_col=0, parse_dates=True, engine='openpyxl')
                st.session_state.prices = df_prices
                st.session_state.uploaded_prices = True
                st.sidebar.success("✅ Цены загружены")
            if uploaded_rates is not None:
                if uploaded_rates.name.endswith('.csv'):
                    df_rates=pd.read_csv(uploaded_rates, index_col=0, parse_dates=True)
                else:
                    df_rates=pd.read_excel(uploaded_rates, index_col=0, parse_dates=True, engine='openpyxl')
                if 'USD_KZT' in df_rates.columns:
                    st.session_state.usd_kzt = df_rates['USD_KZT']
                    st.session_state.uploaded_rates = True
                    st.sidebar.success("✅ Курс загружен")
                else:
                    st.sidebar.error("❌ В файле курса нет колонки 'USD_KZT'")
            if uploaded_macro is not None:
                if uploaded_macro.name.endswith('.csv'):
                    df_macro=pd.read_csv(uploaded_macro)
                else:
                    df_macro=pd.read_excel(uploaded_macro, engine='openpyxl')
                if 'k_macro' in df_macro.columns:
                    macro_k=df_macro['k_macro'].iloc[-1]
                    st.session_state.macro_k_global=macro_k
                    if 'portfolio' in st.session_state:
                        st.session_state.portfolio['macro_k']=macro_k
                    st.sidebar.success(f"✅ Макро-коэф. {macro_k:.2f}")
                else:
                    st.sidebar.error("❌ Нет колонки 'k_macro'")
        except Exception as e:
            st.sidebar.error(f"❌ Ошибка: {e}")

st.sidebar.markdown("---")
st.sidebar.subheader("🔄 Обновление данных")
if st.sidebar.button("📥 Обновить данные за сегодня"):
    st.session_state.portfolio = portfolio
    st.session_state.metrics = calc_metrics(portfolio, prices, usd_kzt)
    st.sidebar.success("✅ Данные обновлены")
    st.rerun()

st.sidebar.write("---")
st.sidebar.write("📊 **Полная система риск-менеджмента (MVP)**")
st.sidebar.write("Синтетические данные. Для реальной работы замените генерацию на загрузку из CSV.")
st.sidebar.write(f"Дата расчёта: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
