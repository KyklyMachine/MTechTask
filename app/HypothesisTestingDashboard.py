import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from statsmodels.stats.proportion import proportions_ztest


# Константы
PAGE_CONFIG = {
    "layout": "wide"
}
MAIN_TITLE = "Тестирование Гипотез"
FILE_LOAD_ERROR = ('Загрузите файл формата .csv в кодировке "Кириллица (Windows)" (cp-1251), '
                   'который имеет 3 столбца с названиями "Количество больничных дней", "Возраст", "Пол"')
GENERAL_DESCRIPTION = (r"**Гипотезы $H_0$ и $H'$**" + "  \n"
                       r"$H_0: p_1 = p_2$" + "  \n"
                       r"$H': p_1 > p_2$" + "  \n" + "  \n"
                       r"**Статистика критерия**" + "  \n"
                       r"$Z=\frac{H_1-H_2}{\sqrt{H(1-H)}\sqrt{\frac{1}{n_1}+\frac{1}{n_2}}}$" + "  \n" + "  \n"
                       r"**Закон распределения $f(z|H_0)$**" + "  \n"
                       r"$f(z|H_0)\sim N(0,1)$" + "  \n" + "  \n"
                       "**Критическая область**" + "  \n"
                       "критическая область выбирается правосторонней" + "  \n" + "  \n")


DESCRIPTION_TEST_1 = (r"$p_1$ - вероятность того, что мужчина пропустит более {work_days} рабочих дней" + "  \n"
                      r"$p_2$ - вероятность того, что женщина пропустит более {work_days} рабочих дней" + "  \n  \n")

DESCRIPTION_TEST_2 = (r"$p_1$ - вероятность пропуска более {work_days} рабочих дней "
                      r"работником старше {age} лет" + "  \n"
                      r"$p_2$ - вероятность пропуска более {work_days} рабочих дней "
                      r"работником младше {age} лет" + "  \n  \n")


# Конфиг страницы
st.set_page_config(**PAGE_CONFIG)

# Убираем пустое пространство вверху
st.markdown("""
        <style>
               .block-container {
                    padding-top: 1rem;
                    padding-bottom: 0rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
        </style>
        """, unsafe_allow_html=True)


def preprocess_data(dataframe: pd.DataFrame):
    """
    Приводит dataframe к требуемому для дальнейшей формату
    """
    dataframe.columns = dataframe.iloc[0]
    dataframe.drop(index=0, inplace=True)
    dataframe["Количество больничных дней"] = dataframe["Количество больничных дней"].astype(np.int_)
    dataframe["Возраст"] = dataframe["Возраст"].astype(np.int_)
    return dataframe


@st.cache_resource(show_spinner="Fetching data...")
def load_data(uploaded_file):
    """
    Загрузчик данных с кэшем
    """
    try:
        raw_df = (pd.read_csv(uploaded_file,
                              encoding='cp1251',
                              sep="\,",
                              engine="python",
                              header=None)
                  .apply(lambda x: x.str.replace(r"\"", "")))
        return preprocess_data(raw_df)
    except Exception as e:
        st.warning(FILE_LOAD_ERROR)
        st.warning(e)
        return None


def plot_histogram(age_array, work_days_array):
    """
    Строит гистограммы возраста и количества пропущенных дней
    """
    fig = make_subplots(rows=1, cols=2)
    trace0 = go.Histogram(x=age_array, name="Возраст")
    trace1 = go.Histogram(x=work_days_array, name="Количество больничных дней")
    fig.add_trace(trace0, 1, 1)
    fig.add_trace(trace1, 1, 2)
    fig.update_layout(title_text="Гистограммы абсолютных частот")
    st.plotly_chart(fig, use_container_width=True)


def plot_density_function(stat, alpha):
    """
    Строит функцию плотности N(0,1) с обозначенной критической областью (правосторонней) и
    закрашенной областью = p-value
    """
    r = stats.norm.ppf([1 - alpha], 0, 1).item()
    x1 = np.arange(-3, 3, 0.01)
    y1 = stats.norm.pdf(x1)
    x2 = np.arange((r // 0.01) * 0.01, 3, 0.01)
    y2 = stats.norm.pdf(x2)

    if not np.isnan(stat):
        x3 = np.arange(stat, 3, 0.01)
        y3 = stats.norm.pdf(x3)
    else:
        x3 = y3 = np.array([])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x1, y=y1, fill='tozeroy', mode='none', fillcolor="rgba(165, 165, 165, 0.5)",
                             name=r"f(z|H_0)"
                             ))
    fig.add_trace(go.Scatter(x=x2, y=y2, fill='tozeroy', mode='none', fillcolor="rgba(255, 94, 94, 0.5)",
                             name="critical area"
                             ))
    if not np.isnan(stat):
        fig.add_trace(go.Scatter(x=x3, y=y3, fill='tozeroy', mode='none', fillcolor="rgba(109, 141, 255, 0.5)",
                                 name="p-value"
                                 ))
    fig.update_layout(height=400, title="Функция плотности f(z|H_0)")
    return fig


def get_left_right_points(m, std, step):
    """
    Находит левую и правую "крайние" части распределения N(m, std)
    """
    if np.isnan(m):
        left = None
        right = None
    elif not std:
        left = m - step * 10
        right = m - step * 10
    else:
        left = stats.norm.ppf(step * 10, m, std)
        right = stats.norm.ppf(1 - step * 10, m, std)
    return left, right


def plot_pdfs(count, nobs, step=0.001):
    """
    Строит график функций распределения для p_1 и p_2
    """
    p1, p2 = count / nobs
    std1 = np.sqrt(p1 * (1 - p1) / nobs[0])
    std2 = np.sqrt(p2 * (1 - p2) / nobs[1])

    left1, right1 = get_left_right_points(p1, std1, step)
    left2, right2 = get_left_right_points(p2, std2, step)
    left = np.min([left1 if left1 else np.infty, left2 if left2 else np.infty])
    right = np.max([right1 if right1 else -np.infty, right2 if right2 else -np.infty])

    x = np.arange(left, right, step)
    y1 = stats.norm.pdf(x, p1, std1)
    y2 = stats.norm.pdf(x, p2, std2)
    y_min = list(map(min, zip(y1, y2)))

    fig = go.Figure()
    if std1 == 0 or np.isnan(p1):
        pass
    else:
        fig.add_trace(go.Scatter(x=x, y=y1,
                                 mode='lines',
                                 name='Распределение p1',
                                 fill='tozeroy',
                                 line_color="rgba(39, 108, 245, 0.6)"))
        fig.add_trace(go.Scatter(x=[p1, p1], y=[0, stats.norm.pdf(p1, p1, std1)],
                                 mode='lines',
                                 name='p1-среднее',
                                 fill='tozeroy',
                                 line_color="rgba(39, 108, 245, 0.9)"))
    if std2 == 0 or np.isnan(p2):
        pass
    else:
        fig.add_trace(go.Scatter(x=x, y=y2,
                                 mode='lines',
                                 name='Распределение p2',
                                 fill='tozeroy',
                                 line_color="rgba(245, 40, 145, 0.6)"))
        fig.add_trace(go.Scatter(x=[p2, p2], y=[0, stats.norm.pdf(p2, p2, std2)],
                                 mode='lines',
                                 name='p2-среднее',
                                 fill='tozeroy',
                                 line_color="rgba(245, 40, 145, 0.9)"))

    if std1 and std2 and not np.isnan(p1) and not np.isnan(p2):
        fig.add_trace(go.Scatter(x=x, y=y_min,
                                 mode='lines',
                                 name='Пересечение',
                                 fill='tozeroy',
                                 line_color="rgba(187, 0, 231, 0.8)"))

    fig.update_layout(height=400, title="Распределение вероятностей")
    return fig


def file_loader():
    """
    Вкладка загрузки данных
    """
    uploaded_file = st.file_uploader("Choose a CSV file")
    if uploaded_file is not None:
        raw_df = load_data(uploaded_file)
        if raw_df is not None:
            with st.container(border=False):
                st.dataframe(raw_df, use_container_width=True)
                plot_histogram(raw_df["Возраст"], raw_df["Количество больничных дней"])
            return raw_df
    return None


def show_results(count, nobs, stat, pval, alpha, description=""):
    """
    Выводит на страницу 2 столбца (описание | графики) с результатами проверки гипотезы
    """
    col11, col12 = st.columns(2)
    with col11:
        st.write(description)

        p1 = str(count[0]) + "/" + str(nobs[0])
        p2 = str(count[1]) + "/" + str(nobs[1])

        res_str = "**Итоги проверки гипотезы**:  \n"

        if np.isnan(stat):
            res_str += (r"Невозможно проверить гипотезу, так как: $\tilde "
                        fr"{r'p_1=' + p1 if p1 == '0/0' else 'p_2=' + p2}$" + "  \n")
        else:
            res_str += (fr"Выборочное значение статистики критерия $z_в={round(stat, 5)}$" + "  \n"
                        fr"$P–value = {round(pval, 5)}$" + "  \n"
                        fr"Уровень значимости $\alpha={alpha}$" + "  \n"
                        fr"$P–value {'>' if pval > alpha else '<'} \alpha \Rightarrow$ "
                        fr"**Гипотеза $H_0$ {'принимается' if pval > alpha else r'отклоняется'}!**"
                        fr"")
        st.write(res_str)
    with col12:
        st.plotly_chart(plot_pdfs(count, nobs), use_container_width=True)
        st.plotly_chart(plot_density_function(stat, alpha), use_container_width=True)


def old_young_hypothesis(dataframe: pd.DataFrame):
    """
    Вкладка с тестированием гипотезы №2
    """
    if dataframe is not None:

        # 1. Название теста
        header = (r"Тест 2: Работники старше $\textbf{age}$ лет пропускают в течение года более $\textbf{work\_days}$ "
                  r"рабочих дней по болезни значимо чаще своих более молодых коллег.")

        st.subheader(header)

        # 2. Выбор переменных пользователем
        col1, col2, col3 = st.columns(3)
        with col1:
            age_list = np.sort(dataframe["Возраст"].unique()).tolist()[::-1]
            age = st.selectbox(r'Возраст ($\textbf{age}$)', age_list, key="age_test2")
        with col2:
            work_days_list = np.sort(dataframe["Количество больничных дней"].unique()).tolist()[:-1]
            work_days = st.selectbox(r'Количество больничных дней ($\textbf{work\_days}$)',
                                     work_days_list, key="work_days_test2")
        with col3:
            alpha = st.slider(r"Уровень значимости $\alpha$", 0.01, 0.1, 0.05, key="alpha_test2")

        # 3. Получение необходимых данных
        n_1 = len(dataframe[dataframe["Возраст"] > age]["Возраст"])
        n_2 = len(dataframe[dataframe["Возраст"] <= age]["Возраст"])

        # используя условие на "Количество больничных дней" находим всех людей, которые пропустили по болезни
        # больше WORK_DAYS. Далее используем условие на количество лет AGE и находим искомые n_pass_1 и n_pass_2
        work_days_greater = dataframe[(dataframe["Количество больничных дней"] > work_days)]["Возраст"]
        n_pass_1 = len(work_days_greater[work_days_greater > age])
        n_pass_2 = len(work_days_greater[work_days_greater <= age])

        # 4. Проведение теста
        count = np.array([n_pass_1, n_pass_2])
        nobs = np.array([n_1, n_2])
        stat, pval = proportions_ztest(count, nobs, alternative="larger")

        # 5. Вывод результатов и графиков
        all_description = DESCRIPTION_TEST_2.format(work_days=work_days, age=age) + GENERAL_DESCRIPTION
        show_results(count, nobs, stat, pval, alpha, all_description)
    else:
        st.info("Сначала загрузите файл")


def man_woman_hypothesis(dataframe: pd.DataFrame):
    """
    Вкладка с тестированием гипотезы №1
    """
    if dataframe is not None:

        # 1. Название теста
        header = (r"Тест 1: Мужчины пропускают в течение года более $\textbf{work\_days}$ рабочих дней по болезни"
                  r" значимо чаще женщин.")
        st.subheader(header)

        # 2. Выбор переменных пользователем
        col1, col2 = st.columns(2)
        with col1:
            work_days_list = np.sort(dataframe["Количество больничных дней"].unique()).tolist()[:-1]
            work_days = st.selectbox(r'Количество больничных дней ($\textbf{work\_days}$)',
                                     work_days_list, key="work_days_test1")
        with col2:
            alpha = st.slider(r"Уровень значимости $\alpha$", 0.01, 0.1, 0.05, key="age_test1")

        # 3. Получение необходимых данных
        work_days_total = dataframe.groupby("Пол").size()
        n_1, n_2 = work_days_total[["М", "Ж"]]

        # используя условие на "Количество больничных дней", группируя по полу и считая количество по каждому разбиению
        # получаем необходимые n_pass_1 и n_pass_2 (или нули, если таких представителей после группировки не осталось)
        work_days_greater = dataframe[(dataframe["Количество больничных дней"] > work_days)].groupby("Пол").size()
        n_pass_1 = work_days_greater["М"] if "М" in work_days_greater.index else 0
        n_pass_2 = work_days_greater["Ж"] if "Ж" in work_days_greater.index else 0

        # 4. Проведение теста
        count = np.array([n_pass_1, n_pass_2])
        nobs = np.array([n_1, n_2])
        stat, pval = proportions_ztest(count, nobs, alternative="larger")

        # 5. Вывод результатов и графиков
        all_description = DESCRIPTION_TEST_1.format(work_days=work_days) + GENERAL_DESCRIPTION
        show_results(count, nobs, stat, pval, alpha, all_description)
    else:
        st.info("Сначала загрузите файл")


def main():
    """
    Основная функция дашборда
    """
    st.title(MAIN_TITLE)

    tab1, tab2, tab3 = st.tabs(["Данные", "Тест 1", "Тест 2"])

    with tab1:
        dataframe = file_loader()

    with tab2:
        man_woman_hypothesis(dataframe)

    with tab3:
        old_young_hypothesis(dataframe)


if __name__ == "__main__":
    main()
