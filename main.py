import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.cluster import KMeans
import geopandas as gpd
from shapely.geometry import Point
import folium
from streamlit_folium import folium_static
from folium.plugins import MarkerCluster, HeatMap

st.set_page_config(layout="wide", page_title="Real Estate Dashboard - Almaty")

# Добавляем выпадающий список для выбора файла
file_options = ['krisha_data.xlsx', 'krisha_arenda_data.xlsx']
selected_file = st.selectbox("Выберите файл для загрузки", file_options, index=0)  # По умолчанию выбран первый файл

# Загрузка данных на основе выбора
data = pd.read_excel(selected_file, sheet_name='Sheet')

# Для отображения загруженных данных (например, первых 5 строк)
st.write("Загруженные данные:")
st.write(data.head())

# Переименование столбцов для соответствия требованиям карты
data.rename(columns={'Latitude': 'latitude', 'Longitude': 'longitude'}, inplace=True)

# Выбор числовых данных для корреляционной матрицы
numeric_data = data.select_dtypes(include=['number'])

# Конвертация данных в GeoDataFrame
geometry = [Point(xy) for xy in zip(data['longitude'], data['latitude'])]
geodata = gpd.GeoDataFrame(data, geometry=geometry)


# Streamlit приложение
def main():
    st.title("Real Estate Dashboard - Almaty")

    col1, col2 = st.columns(2)

    # Распределение цен
    with col1:
        st.markdown("### Распределение цен")
        fig_price_dist = px.histogram(data, x="Price", nbins=20, title="Распределение цен на недвижимость",
                                      template="plotly_white", color_discrete_sequence=['#1f77b4'])
        fig_price_dist.update_layout(plot_bgcolor='white', margin=dict(l=20, r=20, t=50, b=20),
                                     height=400)
        fig_price_dist.update_traces(texttemplate='%{y}', textposition='outside')
        st.plotly_chart(fig_price_dist, use_container_width=False)

        # Средняя цена по районам
        with col2:
            st.markdown("### Средняя цена по районам")

            # Рассчет средней цены по районам
            avg_price_district = data.groupby("District")["Price"].mean().reset_index()

            # Создание графика с улучшенными элементами визуализации
            fig_avg_price_district = px.bar(
                avg_price_district,
                x="District",
                y="Price",
                title="Средняя цена по районам",
                template="plotly_white",
                color_discrete_sequence=['#2ca02c']
            )

            # Улучшение оформления графика
            fig_avg_price_district.update_layout(
                plot_bgcolor='white',
                margin=dict(l=40, r=40, t=50, b=100),  # Добавляем больше пространства для подписей
                height=500,
                title_font=dict(size=20, family='Arial, sans-serif'),
                xaxis_tickangle=-45,  # Наклон подписей на оси X для удобочитаемости
                xaxis_title="Район",
                yaxis_title="Средняя цена (KZT)",
                xaxis=dict(tickmode='linear', tickfont=dict(size=12)),
                yaxis=dict(tickfont=dict(size=12)),
            )

            # Улучшение меток на столбцах для более профессионального вида
            fig_avg_price_district.update_traces(
                texttemplate='%{y:,.0f} KZT',
                textposition='outside',
                marker=dict(line=dict(color='#333', width=1.5))  # Добавляем обводку для столбцов
            )

            # Вывод графика с использованием ширины контейнера
            st.plotly_chart(fig_avg_price_district, use_container_width=True)

    col3, col4 = st.columns(2)

    # Количество объектов по году постройки
    with col3:
        st.markdown("### Количество объектов по году постройки")
        year_counts = data["Year of Construction"].value_counts().sort_index().reset_index()
        year_counts.columns = ["Year of Construction", "Count"]
        fig_year_constructed = px.line(year_counts, x="Year of Construction", y="Count",
                                       labels={"Year of Construction": "Год постройки",
                                               "Count": "Количество объектов"},
                                       title="Количество объектов по году постройки",
                                       template="plotly_white", color_discrete_sequence=['#d62728'])
        fig_year_constructed.update_layout(plot_bgcolor='white', margin=dict(l=20, r=20, t=50, b=20),
                                           height=400)
        fig_year_constructed.update_traces(texttemplate='%{y}', textposition='top center')
        st.plotly_chart(fig_year_constructed, use_container_width=False)

    # Средняя цена по количеству комнат
    with col4:
        st.markdown("### Средняя цена по количеству комнат")

        # Расчет средней цены по количеству комнат
        avg_price_rooms = data.groupby("Rooms")["Price"].mean().reset_index()

        # Создание графика с улучшенной визуализацией
        fig_avg_price_rooms = px.bar(
            avg_price_rooms,
            x="Rooms",
            y="Price",
            title="Средняя цена по количеству комнат",
            template="plotly_white",
            color_discrete_sequence=['#9467bd']
        )

        # Настройка оформления графика
        fig_avg_price_rooms.update_layout(
            plot_bgcolor='white',
            margin=dict(l=40, r=40, t=50, b=80),  # Увеличение пространства для подписей
            height=450,
            title_font=dict(size=20, family='Arial, sans-serif'),
            xaxis_title="Количество комнат",  # Четкий заголовок оси X
            yaxis_title="Средняя цена (KZT)",  # Четкий заголовок оси Y
            xaxis=dict(
                tickmode='linear',
                tickfont=dict(size=12)
            ),
            yaxis=dict(
                tickfont=dict(size=12),
                tickformat=',',  # Формат для разделения тысяч на оси Y
            ),
        )

        # Улучшение меток на столбцах для удобства восприятия
        fig_avg_price_rooms.update_traces(
            texttemplate='%{y:,.0f} KZT',  # Форматирование чисел с разделением тысяч
            textposition='outside',
            marker=dict(line=dict(color='#333', width=1.5))  # Добавление обводки для четкости
        )

        # Отображение графика с полной шириной контейнера
        st.plotly_chart(fig_avg_price_rooms, use_container_width=True)

    col5, col6 = st.columns(2)

    # Карта объектов в Алматы с цветовой индикацией цен
    with col5:
        st.markdown("### Карта объектов недвижимости в Алматы")
        fig_map = px.scatter_mapbox(data, lat="latitude", lon="longitude", size="Price",
                                    color="Price", color_continuous_scale="thermal", hover_name="District", zoom=10,
                                    title="Недвижимость в Алматы с цветовой индикацией цен",
                                    mapbox_style="carto-positron")
        fig_map.update_layout(margin=dict(l=20, r=20, t=50, b=20), height=500)
        st.plotly_chart(fig_map, use_container_width=False)

    with col6:
        st.markdown("### Кластерная карта объектов недвижимости в Алматы")

        # Вычисляем квартильные значения для цены
        q50 = data['Price'].quantile(0.50)
        q75 = data['Price'].quantile(0.75)

        # Создаем карту
        m = folium.Map(location=[data['latitude'].mean(), data['longitude'].mean()], zoom_start=10)

        # Добавляем кластер маркеров
        marker_cluster = MarkerCluster().add_to(m)

        # Итерация по строкам данных для добавления маркеров
        for idx, row in data.iterrows():
            # Цвет маркера в зависимости от ценовой категории
            price_category = "red" if row['Price'] > q75 else (
                "orange" if row['Price'] > q50 else "green")

            # Улучшенное содержимое всплывающего окна
            popup_content = f"""
                <b>Название:</b> {row['Title']}<br>
                <b>Цена:</b> {row['Price']:,} KZT<br>
                <b>Площадь:</b> {row['Square']:,} м²<br>
                <b>Количество комнат:</b> {row['Rooms']}<br>
                <b>Год постройки:</b> {row['Year of Construction']}<br>
                <b>Район:</b> {row['District']}<br>
                <b>Состояние:</b> {row['Condition']}<br>
                <b>Ссылка:</b> <a href="https://krisha.kz/a/show/{row['ID']}" target="_blank">Открыть объявление</a> <br>
            """

            # Добавляем маркер на карту с всплывающим окном
            folium.Marker(
                location=[row['latitude'], row['longitude']],
                popup=folium.Popup(popup_content, max_width=300),
                icon=folium.Icon(color=price_category)
            ).add_to(marker_cluster)

        # Добавляем тепловую карту на основе цен
        HeatMap(data[['latitude', 'longitude', 'Price']].values.tolist(), radius=15).add_to(m)

        # Вывод карты с помощью folium_static
        folium_static(m)

        # Легенда для цветов маркеров (красный, оранжевый, зеленый), с указанием порогов
        st.markdown(f"""
        **Легенда ценовых категорий:**
        - <span style="color:red">Высокая цена (больше {q75:,.0f} KZT)</span><br>
        - <span style="color:orange">Средняя цена (от {q50:,.0f} до {q75:,.0f} KZT)</span><br>
        - <span style="color:green">Низкая цена (меньше {q50:,.0f} KZT)</span>
        """, unsafe_allow_html=True)

    col9, col10 = st.columns(2)

    # Корреляционная матрица
    with col9:
        st.markdown("### Корреляционная матрица данных")
        corr_matrix = numeric_data.corr()
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='Viridis',
            text=corr_matrix.values,
            texttemplate='%{text:.2f}',
            hoverongaps=False))
        fig_corr.update_layout(title="Корреляционная матрица данных", margin=dict(l=20, r=20, t=50, b=20),
                               height=600, width=800)
        st.plotly_chart(fig_corr, use_container_width=False)

        # Цена недвижимости vs. Площадь
        with col10:
            st.markdown("### Зависимость цены недвижимости от площади")

            # Создание улучшенного scatter-графика с трендовой линией
            fig_price_sqft = px.scatter(
                data,
                x="Square",
                y="Price",
                trendline="ols",
                title="Цена недвижимости в зависимости от площади",
                template="plotly_white",
                color_discrete_sequence=['#ff7f0e'],
                labels={"Square": "Площадь (м²)", "Price": "Цена (KZT)"}  # Уточнение подписей осей
            )

            # Настройка оформления графика
            fig_price_sqft.update_layout(
                plot_bgcolor='white',
                margin=dict(l=40, r=40, t=50, b=60),  # Добавление пространства для подписей
                height=450,
                title_font=dict(size=20, family='Arial, sans-serif'),
                xaxis_title="Площадь (м²)",  # Заголовок оси X
                yaxis_title="Цена (KZT)",  # Заголовок оси Y
                xaxis=dict(
                    tickmode='array',
                    tickvals=[round(x, -1) for x in range(int(data['Square'].min()), int(data['Square'].max()), 10)],
                    # Шаг 10 м²
                    tickfont=dict(size=12),
                    tickformat=',',  # Форматирование оси X для больших значений площади
                    title_standoff=10  # Дополнительное пространство между осью и заголовком
                ),
                yaxis=dict(
                    tickfont=dict(size=12),
                    tickformat=',',  # Разделение тысяч на оси Y
                    title_standoff=10  # Дополнительное пространство между осью и заголовком
                ),
            )

            # Настройка меток точек и трендовой линии
            fig_price_sqft.update_traces(
                marker=dict(size=8, line=dict(width=1, color='#333')),  # Увеличение размера точек и обводка
                hovertemplate='Площадь: %{x} м²<br>Цена: %{y:,.0f} KZT'
            )

            # Настройка трендовой линии
            fig_price_sqft.data[1].update(line=dict(color='blue', width=2))  # Синяя линия для тренда

            # Визуализация графика с полной шириной контейнера
            st.plotly_chart(fig_price_sqft, use_container_width=True)

    col11, col12 = st.columns(2)

    # Цена за квадратный метр по районам
    with col11:
        st.markdown("### Цена за квадратный метр по районам")

        # Расчет средней цены за квадратный метр по районам
        avg_price_per_sqm = data.groupby("District")["Price per m²"].mean().reset_index()

        # Создание графика с улучшенной визуализацией
        fig_avg_price_per_sqm = px.bar(
            avg_price_per_sqm,
            x="District",
            y="Price per m²",
            title="Средняя цена за квадратный метр по районам",
            template="plotly_white",
            color_discrete_sequence=['#17becf']
        )

        # Настройка оформления графика
        fig_avg_price_per_sqm.update_layout(
            plot_bgcolor='white',
            margin=dict(l=40, r=40, t=50, b=120),  # Увеличение пространства для подписей оси X
            height=450,
            title_font=dict(size=20, family='Arial, sans-serif'),
            xaxis_title="Район",  # Четкий заголовок оси X
            yaxis_title="Средняя цена за м² (KZT)",  # Четкий заголовок оси Y
            xaxis=dict(
                tickmode='linear',
                tickfont=dict(size=12),
                tickangle=-45  # Наклон подписей оси X для удобочитаемости
            ),
            yaxis=dict(
                tickfont=dict(size=12),
                tickformat=',',  # Разделение тысяч для оси Y
            )
        )

        # Улучшение меток на столбцах для удобства восприятия
        fig_avg_price_per_sqm.update_traces(
            texttemplate='%{y:,.0f} KZT',  # Форматирование чисел с разделением тысяч и указанием валюты
            textposition='outside',
            marker=dict(line=dict(color='#333', width=1.5))  # Добавление обводки для четкости
        )

        # Визуализация графика с полной шириной контейнера
        st.plotly_chart(fig_avg_price_per_sqm, use_container_width=True)

    with col12:
        st.markdown("### Количество объектов по состоянию")
        condition_counts = data['Condition'].value_counts().reset_index()
        condition_counts.columns = ['Condition', 'Count']
        fig_condition = px.bar(condition_counts, x='Condition', y='Count',
                               title='Количество объектов по состоянию',
                               template='plotly_white', color_discrete_sequence=['#bcbd22'])
        fig_condition.update_layout(plot_bgcolor='white', margin=dict(l=20, r=20, t=50, b=20))
        fig_condition.update_traces(texttemplate='%{y}', textposition='outside')
        st.plotly_chart(fig_condition, use_container_width=False)

    # Карта кластеров недвижимости
    st.markdown("### Карта кластеров недвижимости по выбранным параметрам")

    # Выбор дополнительных параметров для кластеризации
    available_features = ['Price', 'Square', 'Rooms', 'Year of Construction', 'Floor']
    selected_features = st.multiselect(
        "Выберите дополнительные параметры для кластеризации (широта и долгота включены по умолчанию):",
        options=available_features,
        default=[]
    )

    # Параметры для кластеризации (широта и долгота всегда включены)
    clustering_features = ['latitude', 'longitude'] + selected_features

    # Выбор количества кластеров
    n_clusters = st.slider("Выберите количество кластеров", min_value=2, max_value=10, value=5)

    # Выполнение кластеризации KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    data['Cluster'] = kmeans.fit_predict(data[clustering_features])

    # Получение центроидов кластеров
    centroids = kmeans.cluster_centers_

    # Сортировка кластеров по медиане выбранного параметра (если что-то выбрано)
    if selected_features:
        # Берем первый выбранный параметр для сортировки
        sort_feature = selected_features[0]
        # Вычисляем медиану для каждого кластера
        cluster_medians = data.groupby('Cluster')[sort_feature].median().reset_index()
        # Сортируем кластеры по медиане
        sorted_clusters = cluster_medians.sort_values(by=sort_feature)['Cluster'].tolist()
    else:
        sorted_clusters = list(range(n_clusters))  # Если параметры не выбраны, сортировка по порядку

    # Создание колонок для визуализации и описания
    col_visualization, col_description = st.columns([2, 1])

    with col_visualization:
        # Определение пользовательской цветовой палитры
        custom_colors = ['#e6194b',  # красный
                         '#3cb44b',  # зеленый
                         '#ffe119',  # желтый
                         '#4363d8',  # синий
                         '#f58231',  # оранжевый
                         '#911eb4',  # фиолетовый
                         '#46f0f0',  # циан
                         '#f032e6',  # маджента
                         '#bcf60c',  # лайм
                         '#fabebe']  # розовый

        # Проверяем, достаточно ли цветов для выбранного количества кластеров
        if n_clusters > len(custom_colors):
            st.error(
                "Количество кластеров превышает количество доступных цветов. Пожалуйста, уменьшите количество кластеров.")
        else:
            # Добавление информации о кластерах для каждого объекта
            for cluster in sorted_clusters:
                # Получение объектов, принадлежащих текущему кластеру
                cluster_data = data[data['Cluster'] == cluster]

                # Определение диапазонов характеристик только для выбранных параметров
                cluster_info = []
                if 'Price' in selected_features:
                    price_range = f"{cluster_data['Price'].min():,.0f} - {cluster_data['Price'].max():,.0f} KZT"
                    cluster_info.append(f"Цена: {price_range}")

                if 'Square' in selected_features:
                    square_range = f"{cluster_data['Square'].min():,.0f} - {cluster_data['Square'].max():,.0f} м²"
                    cluster_info.append(f"Площадь: {square_range}")

                if 'Rooms' in selected_features:
                    rooms_range = f"{cluster_data['Rooms'].min()} - {cluster_data['Rooms'].max()} комнат"
                    cluster_info.append(f"Количество комнат: {rooms_range}")

                if 'Year of Construction' in selected_features:
                    year_range = f"{cluster_data['Year of Construction'].min()} - {cluster_data['Year of Construction'].max()}"
                    cluster_info.append(f"Год постройки: {year_range}")

                if 'Floor' in selected_features:
                    floor_range = f"{cluster_data['Floor'].min()} - {cluster_data['Floor'].max()}"
                    cluster_info.append(f"Этаж: {floor_range}")

                # Обновляем информацию в hover_data, чтобы включить диапазоны характеристик кластера
                data.loc[data['Cluster'] == cluster, 'Cluster Info'] = "\n".join(cluster_info)

            # Создание карты с кластерной информацией
            fig_cluster = px.scatter_mapbox(
                data,
                lat="latitude",
                lon="longitude",
                color="Cluster",
                color_continuous_scale="thermal",
                size="Price",
                hover_name="Title",
                hover_data={
                    "Price": True,
                    "Square": True,
                    "Rooms": True,
                    "Year of Construction": True,
                    "District": True,
                    "Condition": True,
                    "Cluster Info": True
                },
                zoom=10,
                title="Карта кластеров недвижимости по выбранным параметрам",
                mapbox_style="carto-positron"
            )

            # Добавление центроидов кластеров на карту
            centroid_df = pd.DataFrame(centroids, columns=clustering_features)
            fig_cluster.add_trace(
                go.Scattermapbox(
                    lat=centroid_df['latitude'],
                    lon=centroid_df['longitude'],
                    mode='markers',
                    marker=go.scattermapbox.Marker(
                        size=15,
                        color='black',
                        symbol='star'
                    ),
                    showlegend=False,
                    hoverinfo='none'
                )
            )

            fig_cluster.update_layout(
                margin=dict(l=20, r=20, t=50, b=20),
                height=500
            )
            st.plotly_chart(fig_cluster, use_container_width=True)

    with col_description:
        # Описание кластеров с характеристиками
        st.markdown(f"**Описание кластеров недвижимости:**")

        for cluster in sorted_clusters:
            cluster_data = data[data['Cluster'] == cluster]

            # Динамическое создание описания только для выбранных параметров
            description = []
            if 'Price' in selected_features:
                price_range = f"{cluster_data['Price'].min():,.0f} - {cluster_data['Price'].max():,.0f} KZT"
                description.append(f"Цена: {price_range}")

            if 'Square' in selected_features:
                square_range = f"{cluster_data['Square'].min():,.0f} - {cluster_data['Square'].max():,.0f} м²"
                description.append(f"Площадь: {square_range}")

            if 'Rooms' in selected_features:
                rooms_range = f"{cluster_data['Rooms'].min()} - {cluster_data['Rooms'].max()} комнат"
                description.append(f"Количество комнат: {rooms_range}")

            if 'Year of Construction' in selected_features:
                year_range = f"{cluster_data['Year of Construction'].min()} - {cluster_data['Year of Construction'].max()}"
                description.append(f"Год постройки: {year_range}")

            if 'Floor' in selected_features:
                floor_range = f"{cluster_data['Floor'].min()} - {cluster_data['Floor'].max()}"
                description.append(f"Этаж: {floor_range}")

            # Вывод описания кластера с новой строки для каждого параметра
            st.markdown(f"""
            - **Кластер {sorted_clusters.index(cluster) + 1}:**<br>
              {"<br>".join(description)}
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
