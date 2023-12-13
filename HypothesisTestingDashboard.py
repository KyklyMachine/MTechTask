import streamlit as st
import time
import pandas as pd
import numpy as np
import altair as alt
import scipy.stats as stats
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from statsmodels.stats.proportion import proportions_ztest, binom_test


# Константы
PAGE_CONFIG = {
    "layout": "wide"
}
MAIN_TITLE = "Тестирование Гипотез"
FILE_LOAD_ERROR = ('Загрузите файл формата .csv в кодировке "Кириллица (Windows)" (cp-1251), '
                   'который имеет 3 столбца с названиями "Количество больничных дней", "Возраст", "Пол"')


# Конфиг страницы
st.set_page_config(**PAGE_CONFIG)


def preprocess_data(dataframe: pd.DataFrame):
    dataframe.columns = dataframe.iloc[0]
    dataframe.drop(index=0, inplace=True)
    dataframe["Количество больничных дней"] = dataframe["Количество больничных дней"].astype(np.int_)
    dataframe["Возраст"] = dataframe["Возраст"].astype(np.int_)
    return dataframe


@st.cache_resource(show_spinner="Fetching data...")
def load_data(uploaded_file):
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
        return None


def plot_histogram(df: pd.Series):
    fig = make_subplots(rows=1, cols=2)
    trace0 = go.Histogram(x=df["Возраст"], name="Возраст")
    trace1 = go.Histogram(x=df["Количество больничных дней"], name="Количество больничных дней")
    fig.add_trace(trace0, 1, 1)
    fig.add_trace(trace1, 1, 2)
    fig.update_layout(title_text="Гистограммы частот")
    st.plotly_chart(fig, use_container_width=True)


def file_loader():
    uploaded_file = st.file_uploader("Choose a CSV file")
    if uploaded_file is not None:
        raw_df = load_data(uploaded_file)

        if raw_df is not None:
            st.dataframe(raw_df, use_container_width=True)
            plot_histogram(raw_df)
            return raw_df

    return None


def hypothesis_test(dataframe: pd.DataFrame):
    if dataframe is not None:
        pass
    else:
        st.info("Сначала загрузите файл")


def test_hypothesis1(dataframe: pd.DataFrame):
    pass


def test_hypothesis2(dataframe: pd.DataFrame):
    pass


def main():
    st.title(MAIN_TITLE)

    tab1, tab2, tab3 = st.tabs(["Данные", "Тест 1", "Тест 2"])

    with tab1:
        dataframe = file_loader()

    with tab2:
        hypothesis_test(dataframe)

    with tab3:
        hypothesis_test(dataframe)


if __name__ == "__main__":
    main()
