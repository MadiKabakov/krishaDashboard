import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
import geopandas as gpd
from shapely.geometry import Point
import folium
from streamlit_folium import folium_static
from folium.plugins import MarkerCluster, HeatMap
from sklearn.neighbors import KNeighborsClassifier
import re


# ========================================
# FUNCTIONS FOR DATA PROCESSING
# ========================================

def fill_missing_square(df):
    try:
        if 'Square' in df.columns:
            if (df['Square'].isna().all()) or (df['Square'].eq(0).all()):
                st.warning("All 'Square' values are either 0 or missing. Attempting to extract area from 'Title'.")

                def extract_square_from_title(title):
                    match = re.search(r'(\d+)\s?м²', str(title))
                    return int(match.group(1)) if match else 0

                df['Square'] = df['Title'].apply(extract_square_from_title)
                st.write("Area values have been extracted from 'Title'.")
            square_median = df['Square'].median()
            if pd.isna(square_median):
                square_median = 0
            df['Square'] = df['Square'].fillna(square_median)
            st.write(f"Missing values in 'Square' have been replaced with: {square_median}")
        else:
            st.error("'Square' column is missing in the dataset.")
    except Exception as e:
        st.error(f"Error filling missing values in 'Square': {e}")


def fill_missing_floors(df):
    try:
        if 'Current Floor' in df.columns and 'Max Floor' in df.columns:
            current_floor_median = df['Current Floor'].median()
            max_floor_median = df['Max Floor'].median()
            df['Current Floor'] = df['Current Floor'].fillna(current_floor_median)
            df['Max Floor'] = df['Max Floor'].fillna(max_floor_median)
            st.write(f"Missing values in 'Current Floor' replaced with median: {current_floor_median}")
            st.write(f"Missing values in 'Max Floor' replaced with median: {max_floor_median}")
        else:
            st.error("'Current Floor' or 'Max Floor' columns are missing in the dataset.")
    except Exception as e:
        st.error(f"Error filling missing floor values: {e}")


def fill_missing_year_of_construction(df, column_name):
    try:
        median_value = df[column_name].median()
        if column_name not in df.columns:
            st.error(f"Column '{column_name}' is missing in the dataset.")
            return df
        elif df[column_name].isna().sum() == 0:
            st.write(f"No missing values in '{column_name}'.")
            return df
        df[column_name] = df[column_name].fillna(median_value)
        st.write(f"Missing values in '{column_name}' have been replaced with median: {median_value}")
        return df
    except Exception as e:
        st.error(f"Error filling missing values in '{column_name}': {e}")
        return df


def fill_missing_districts(df, latitude_col, longitude_col, district_col):
    try:
        known_districts = df[df[district_col].notna()]
        missing_districts = df[df[district_col].isna()]

        if missing_districts.empty:
            st.write("No missing values in 'District'.")
            return df

        X_known = known_districts[[latitude_col, longitude_col]]
        y_known = known_districts[district_col]
        X_missing = missing_districts[[latitude_col, longitude_col]]

        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(X_known, y_known)
        predicted_districts = knn.predict(X_missing)
        df.loc[missing_districts.index, district_col] = predicted_districts
        st.write(f"Missing districts have been filled: {len(predicted_districts)}")
        return df
    except Exception as e:
        st.error(f"Error filling missing districts: {e}")
        return df


def remove_price_outliers(df, price_column):
    try:
        Q1 = df[price_column].quantile(0.25)
        Q3 = df[price_column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        filtered_df = df[(df[price_column] >= lower_bound) & (df[price_column] <= upper_bound)]
        st.write(f"Outliers removed: {len(df) - len(filtered_df)}")
        return filtered_df
    except Exception as e:
        st.error(f"Error removing price outliers: {e}")
        return df


# ========================================
# PAGE SETTINGS
# ========================================
st.set_page_config(layout="wide", page_title="Real Estate Dashboard - Almaty")

# ========================================
# SIDEBAR SETTINGS
# ========================================
DASHBOARD_DATA_TYPES = {
    "Sale of apartments": "data/Sale_of_apartments.xlsx",
    "Sale of houses": "data/Sale_of_houses.xlsx",
    "Sale of land": "data/Sale_of_land.xlsx",
    "Sale of commercial real estate": "data/Sale_of_commercial_real_estate.xlsx",
    "Sale of business": "data/Sale_of_business.xlsx",
    "Rent of apartments": "data/Rent_of_apartments.xlsx",
    "Rent of houses": "data/Rent_of_houses.xlsx",
    "Commercial real estate rental": "data/Commercial_real_estate_rental.xlsx"
}

st.sidebar.title("Dashboard Settings")
selected_file = st.sidebar.selectbox("Select a file to load", DASHBOARD_DATA_TYPES.keys(), index=0)


@st.cache_data
def load_data(file_path):
    return pd.read_excel(file_path, sheet_name='Sheet')


try:
    data = load_data(DASHBOARD_DATA_TYPES[selected_file])
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

with st.expander("Data Processing Steps"):
    # Extract floor data
    try:
        data['Floor'] = data['Floor'].astype(str)
        data['Current Floor'] = None
        data['Max Floor'] = None
        mask = ~data['Floor'].str.lower().isin(['nan', 'none'])
        extracted_floors = data.loc[mask, 'Floor'].str.extract(r'(\d+)\sиз\s(\d+)')
        data.loc[mask, 'Current Floor'] = pd.to_numeric(extracted_floors[0], errors='coerce')
        data.loc[mask, 'Max Floor'] = pd.to_numeric(extracted_floors[1], errors='coerce')
        st.write("Extracted 'Current Floor' and 'Max Floor' from the 'Floor' column.")
    except Exception as e:
        st.error(f"Error extracting floor data: {e}")

    # Remove outliers in price
    data = remove_price_outliers(data, 'Price')

    # Replace empty values in Condition
    try:
        data['Condition'] = data['Condition'].replace(['None', None, 'nan'], 'No Data')
        data['Condition'] = data['Condition'].fillna('No Data')
        st.write("Empty values in 'Condition' have been replaced with 'No Data'.")
    except Exception as e:
        st.error(f"Error processing 'Condition' column: {e}")

    # Fill missing districts
    data = fill_missing_districts(data, 'Latitude', 'Longitude', 'District')

    # Fill missing year of construction
    data = fill_missing_year_of_construction(data, 'Year of Construction')

    # Fill missing floors
    fill_missing_floors(data)

    # Fill missing square
    fill_missing_square(data)

# Rename columns for map usage
data.rename(columns={'Latitude': 'latitude', 'Longitude': 'longitude'}, inplace=True)

geometry = [Point(xy) for xy in zip(data['longitude'], data['latitude'])]
geodata = gpd.GeoDataFrame(data, geometry=geometry)

numeric_data = data.select_dtypes(include=['number'])

# ========================================
# TOP SUMMARY METRICS (MOVED TO SIDEBAR)
# ========================================
num_rows = data.shape[0]
num_columns = data.shape[1]
memory_usage = data.memory_usage(deep=True).sum() / 1024 ** 2

st.sidebar.markdown("### Dataset Overview")
st.sidebar.metric("Number of Rows", num_rows)
st.sidebar.metric("Number of Columns", num_columns)
st.sidebar.metric("Memory Usage", f"{memory_usage:.2f} MB")

# ========================================
# PLOT SETTINGS
# ========================================
px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = px.colors.qualitative.Bold
px.defaults.width = 850
px.defaults.height = 400
common_layout = dict(
    font_family="Arial",
    title_font_size=20,
    margin=dict(l=40, r=40, t=50, b=60),
    plot_bgcolor='white'
)

# ========================================
# TITLE AND DESCRIPTION
# ========================================
st.markdown(
    f"""
    <h1 style="text-align:center; color:#1f4e79; font-family:Arial, sans-serif; font-size:48px;">
        Real Estate Dashboard - Almaty
    </h1>
    <p style="text-align:center; font-size:20px; font-family:Arial, sans-serif; color:#555;">
        Analytics, visualization, and clustering of the Almaty real estate market.
        Use the sidebar to select data type and clustering parameters.
    </p>
    """,
    unsafe_allow_html=True
)

# ========================================
# TABS
# ========================================
tab_overview, tab_analytics, tab_corr, tab_map, tab_cluster = st.tabs(
    ["Data Overview", "Price & Features Analytics", "Correlations & Dependencies", "Listings Map", "Clustering"])

# ----------------------------------------
# TAB "DATA OVERVIEW"
# ----------------------------------------
with tab_overview:
    st.markdown("### Data Preview")
    st.write("Below are the first 100 rows of the processed dataset:")
    st.write(data.head(100))

    st.markdown("### General Dataset Information")
    missing_values = data.isna().sum().sum()
    st.write(f"- Total Missing Values: {missing_values}")
    st.write("Basic statistical overview:")
    st.dataframe(data.describe(include='all'))

# ----------------------------------------
# TAB "PRICE & FEATURES ANALYTICS"
# ----------------------------------------
with tab_analytics:
    st.markdown("## Price and Key Features Analytics")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Price Distribution")
        fig_price_dist = px.histogram(data, x="Price", nbins=20, title="Distribution of Real Estate Prices")
        fig_price_dist.update_layout(**common_layout)
        fig_price_dist.update_traces(texttemplate='%{y}', textposition='outside')
        st.plotly_chart(fig_price_dist, use_container_width=True)

    with col2:
        st.markdown("### Average Price by District")
        avg_price_district = data.groupby("District")["Price"].mean().reset_index()
        fig_avg_price_district = px.bar(avg_price_district, x="District", y="Price", title="Average Price by District")
        fig_avg_price_district.update_layout(**common_layout, xaxis_tickangle=-45, xaxis_title="District",
                                             yaxis_title="Average Price (KZT)", height=500)
        fig_avg_price_district.update_traces(texttemplate='%{y:,.0f} KZT', textposition='outside',
                                             marker=dict(line=dict(color='#333', width=1.5)))
        st.plotly_chart(fig_avg_price_district, use_container_width=True)

    st.markdown("---")

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("### Number of Listings by Year of Construction")
        year_counts = data["Year of Construction"].value_counts().sort_index().reset_index()
        year_counts.columns = ["Year of Construction", "Count"]
        fig_year_constructed = px.line(year_counts, x="Year of Construction", y="Count",
                                       title="Number of Listings by Construction Year")
        fig_year_constructed.update_layout(**common_layout)
        fig_year_constructed.update_traces(texttemplate='%{y}', textposition='top center')
        st.plotly_chart(fig_year_constructed, use_container_width=True)

    with col4:
        st.markdown("### Average Price by Number of Rooms")
        avg_price_rooms = data.groupby("Rooms")["Price"].mean().reset_index()
        fig_avg_price_rooms = px.bar(avg_price_rooms, x="Rooms", y="Price", title="Average Price by Number of Rooms")
        fig_avg_price_rooms.update_layout(**common_layout, xaxis_title="Number of Rooms",
                                          yaxis_title="Average Price (KZT)", height=450)
        fig_avg_price_rooms.update_traces(texttemplate='%{y:,.0f} KZT', textposition='outside',
                                          marker=dict(line=dict(color='#333', width=1.5)))
        st.plotly_chart(fig_avg_price_rooms, use_container_width=True)

    st.markdown("---")

    col5, col6 = st.columns(2)
    with col5:
        st.markdown("### Listings by Floor Analysis")
        floor_data = data.dropna(subset=['Current Floor', 'Max Floor'])
        if not floor_data.empty:
            floor_counts = floor_data['Current Floor'].value_counts().reset_index()
            floor_counts.columns = ['Current Floor', 'Count']
            fig_bar = px.bar(floor_counts, x='Current Floor', y='Count',
                             title="Distribution of Listings by Current Floor")
            fig_bar.update_layout(**common_layout, xaxis_title='Current Floor', yaxis_title='Number of Listings',
                                  height=500)
            fig_bar.update_traces(textposition='outside')
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.error("Not enough data for floor analysis.")

    with col6:
        st.markdown("### Number of Listings by Condition")
        condition_counts = data['Condition'].value_counts().reset_index()
        condition_counts.columns = ['Condition', 'Count']
        fig_condition = px.bar(condition_counts, x='Condition', y='Count', title='Number of Listings by Condition')
        fig_condition.update_layout(**common_layout)
        fig_condition.update_traces(texttemplate='%{y}', textposition='outside')
        st.plotly_chart(fig_condition, use_container_width=True)

    st.markdown("---")

    st.markdown("### Price per Square Meter by District")
    avg_price_per_sqm = data.groupby("District")["Price per m²"].mean().reset_index()
    fig_avg_price_per_sqm = px.bar(avg_price_per_sqm, x="District", y="Price per m²",
                                   title="Average Price per m² by District")
    fig_avg_price_per_sqm.update_layout(**common_layout, xaxis_title="District",
                                        yaxis_title="Average Price per m² (KZT)", xaxis_tickangle=-45, height=450)
    fig_avg_price_per_sqm.update_traces(texttemplate='%{y:,.0f} KZT', textposition='outside',
                                        marker=dict(line=dict(color='#333', width=1.5)))
    st.plotly_chart(fig_avg_price_per_sqm, use_container_width=True)

# ----------------------------------------
# TAB "CORRELATIONS & DEPENDENCIES"
# ----------------------------------------
with tab_corr:
    st.markdown("## Correlations & Dependencies")

    st.markdown("### Correlation Matrix")
    corr_matrix = numeric_data.corr()
    fig_corr = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='Viridis',
        text=corr_matrix.values,
        texttemplate='%{text:.2f}',
        hoverongaps=False
    ))
    fig_corr.update_layout(
        title="Correlation Matrix", margin=dict(l=20, r=20, t=50, b=20),
        height=600, width=800, font_family="Arial", title_font_size=20
    )
    st.plotly_chart(fig_corr, use_container_width=False)

    st.markdown("---")

    st.markdown("### Price vs. Square Footage")
    fig_price_sqft = px.scatter(
        data, x="Square", y="Price", trendline="ols",
        title="Price vs. Square Footage"
    )
    fig_price_sqft.update_layout(**common_layout, xaxis_title="Square (m²)", yaxis_title="Price (KZT)", height=450)
    fig_price_sqft.update_traces(marker=dict(size=8, line=dict(width=1, color='#333')),
                                 hovertemplate='Square: %{x} m²<br>Price: %{y:,.0f} KZT')
    if len(fig_price_sqft.data) > 1:
        fig_price_sqft.data[1].update(line=dict(color='blue', width=2))
    st.plotly_chart(fig_price_sqft, use_container_width=True)

# ----------------------------------------
# TAB "LISTINGS MAP"
# ----------------------------------------
with tab_map:
    st.markdown("## Clustered Map of Listings in Almaty")
    q50 = data['Price'].quantile(0.50)
    q75 = data['Price'].quantile(0.75)

    m = folium.Map(location=[data['latitude'].mean(), data['longitude'].mean()], zoom_start=10)
    marker_cluster = MarkerCluster().add_to(m)
    for idx, row in data.iterrows():
        price_category = "red" if row['Price'] > q75 else ("orange" if row['Price'] > q50 else "green")
        popup_content = f"""
            <b>Title:</b> {row['Title']}<br>
            <b>Price:</b> {row['Price']:,} KZT<br>
            <b>Area:</b> {row['Square']:,} m²<br>
            <b>Rooms:</b> {row['Rooms']}<br>
            <b>Year of Construction:</b> {row['Year of Construction']}<br>
            <b>District:</b> {row['District']}<br>
            <b>Condition:</b> {row['Condition']}<br>
            <b>Link:</b> <a href="https://krisha.kz/a/show/{row['ID']}" target="_blank">Open Listing</a> <br>
        """
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=folium.Popup(popup_content, max_width=300),
            icon=folium.Icon(color=price_category)
        ).add_to(marker_cluster)

    HeatMap(data[['latitude', 'longitude', 'Price']].values.tolist(), radius=15).add_to(m)
    folium_static(m)

    st.markdown(f"""
    **Price Categories Legend:**
    - <span style="color:red">High price (above {q75:,.0f} KZT)</span><br>
    - <span style="color:orange">Medium price (from {q50:,.0f} to {q75:,.0f} KZT)</span><br>
    - <span style="color:green">Low price (below {q50:,.0f} KZT)</span>
    """, unsafe_allow_html=True)

# ----------------------------------------
# TAB "CLUSTERING"
# ----------------------------------------
with tab_cluster:
    st.markdown("## Clustering of Real Estate Listings")

    available_features = ['Price', 'Square', 'Rooms', 'Year of Construction', 'Current Floor', 'Max Floor']
    selected_features = st.multiselect(
        "Select additional features for clustering (latitude and longitude included by default):",
        options=available_features,
        default=[]
    )

    clustering_features = ['latitude', 'longitude'] + selected_features
    valid_features = []
    for feature in clustering_features:
        if feature not in data.columns:
            st.warning(f"Parameter '{feature}' is not available and will be excluded.")
        elif data[feature].isna().all():
            st.warning(f"Parameter '{feature}' contains only missing values and will be excluded.")
        else:
            valid_features.append(feature)

    if len(valid_features) < 3:
        st.error("Not enough parameters for clustering. Please select at least one available parameter.")
    else:
        n_clusters = st.sidebar.slider("Select number of clusters", min_value=2, max_value=10, value=5)

        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            data['Cluster'] = kmeans.fit_predict(data[valid_features])
            centroids = kmeans.cluster_centers_

            if len(valid_features) > 2:
                sort_feature = valid_features[2]
                cluster_medians = data.groupby('Cluster')[sort_feature].median().reset_index()
                sorted_clusters = cluster_medians.sort_values(by=sort_feature)['Cluster'].tolist()
            else:
                sorted_clusters = list(range(n_clusters))

            col_visualization, col_description = st.columns([2, 1])
            with col_visualization:
                fig_cluster = px.scatter_mapbox(
                    data,
                    lat="latitude",
                    lon="longitude",
                    color="Cluster",
                    hover_name="Title",
                    hover_data={
                        "Price": True,
                        "Square": True,
                        "Rooms": True,
                        "Year of Construction": True,
                        "District": True,
                        "Condition": True,
                    },
                    zoom=10,
                    title="Cluster Map by Selected Parameters",
                    mapbox_style="carto-positron"
                )

                centroid_df = pd.DataFrame(centroids, columns=valid_features)
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
                fig_cluster.update_layout(margin=dict(l=20, r=20, t=50, b=20), height=500, font_family="Arial")
                st.plotly_chart(fig_cluster, use_container_width=True)

            with col_description:
                st.markdown("**Cluster Descriptions:**")
                for cluster in sorted_clusters:
                    cluster_data = data[data['Cluster'] == cluster]
                    description = []
                    for feature in valid_features[2:]:
                        feature_range = f"{cluster_data[feature].min()} - {cluster_data[feature].max()}"
                        description.append(f"{feature}: {feature_range}")
                    st.markdown(f"""
                    - **Cluster {sorted_clusters.index(cluster) + 1}:**<br>
                      {"<br>".join(description)}
                    """, unsafe_allow_html=True)

            st.markdown("### Cluster Analysis: Bar Chart of Record Counts")
            if 'Cluster' in data.columns:
                cluster_counts = data['Cluster'].value_counts().reset_index()
                cluster_counts.columns = ['Cluster', 'Count']
                cluster_counts.sort_values(by='Cluster', inplace=True)
                fig_bar_chart = px.bar(
                    cluster_counts,
                    x='Cluster',
                    y='Count',
                    text='Count',
                    title='Number of Records in Each Cluster'
                )
                fig_bar_chart.update_layout(**common_layout, xaxis_title="Clusters", yaxis_title="Number of Records",
                                            height=500)
                fig_bar_chart.update_traces(texttemplate='%{text}', textposition='outside')
                st.plotly_chart(fig_bar_chart, use_container_width=True)
            else:
                st.error("Clusters have not been computed. Please perform clustering to visualize the chart.")
        except Exception as e:
            st.error(f"Error during clustering: {e}")

# ========================================
# FOOTER
# ========================================
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center; color:#777; font-size:14px;'>© 2024 Real Estate Dashboard - Almaty</p>",
    unsafe_allow_html=True
)
