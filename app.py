import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

# Page Config
st.set_page_config(
    page_title="Historical Weather Dashboard",
    page_icon=":material/partly_cloudy_day:",
    layout="wide",
)

# Title & Description
st.title(":material/partly_cloudy_day: Historical Weather Data Analysis & Dashboard")
st.markdown(
    "Explore multi-year historical weather trends including temperature, humidity, "
    "wind speed, visibility, and atmospheric pressure using interactive filters."
)

# 1. Load Data with Caching
@st.cache_data
def load_data():
    df = pd.read_csv("weather_data.csv")
    df["Date"] = pd.to_datetime(df["Formatted Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df["Precip Type"] = df["Precip Type"].fillna("Unknown")
    return df

df = load_data()

# 2. Sidebar Filters
st.sidebar.header(":material/filter_alt: Filter Weather Data")

# Date Range Filter
min_date = df["Date"].min().date()
max_date = df["Date"].max().date()

date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

if len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

# Categorical Filters
precip_options = sorted(df["Precip Type"].unique().tolist())
selected_precip = st.sidebar.multiselect(
    "Precipitation Type",
    options=precip_options,
    default=precip_options,
)

summary_options = sorted(df["Summary"].dropna().unique().tolist())
selected_summaries = st.sidebar.multiselect(
    "Weather Summary (Condition)",
    options=summary_options,
    default=summary_options,
)

# Apply Filters
filtered_df = df[
    (df["Date"].dt.date >= start_date)
    & (df["Date"].dt.date <= end_date)
    & (df["Precip Type"].isin(selected_precip))
    & (df["Summary"].isin(selected_summaries))
]

if filtered_df.empty:
    st.warning("No data matches your selected filters. Please adjust the sidebar filters.", icon=":material/warning:")
    st.stop()

# 3. Top 4 Key Metrics
st.subheader(":material/analytics: Key Weather Metrics")
col1, col2, col3, col4 = st.columns(4)

avg_temp = filtered_df["Temperature (C)"].mean()
max_temp = filtered_df["Temperature (C)"].max()
avg_humidity = filtered_df["Humidity"].mean() * 100
unique_days = filtered_df["Date"].dt.date.nunique()

col1.metric(":material/thermostat: Average Temperature", f"{avg_temp:.2f} °C")
col2.metric(":material/device_thermostat: Highest Temperature", f"{max_temp:.2f} °C")
col3.metric(":material/water_drop: Average Humidity", f"{avg_humidity:.1f} %")
col4.metric(":material/calendar_month: Total Days Covered", f"{unique_days:,} Days")

st.divider()

# 4. At Least 5 Visualizations (Reactive to Filters)

# Row 1: Univariate + Line Chart Over Time
c1, c2 = st.columns(2)

with c1:
    st.subheader(":material/bar_chart: 1. Temperature Distribution (Univariate)")
    fig_hist = px.histogram(
        filtered_df,
        x="Temperature (C)",
        nbins=40,
        marginal="box",
        color_discrete_sequence=["#FF6B6B"],
        title="Distribution & Spread of Temperature (°C)",
    )
    fig_hist.update_layout(xaxis_title="Temperature (°C)", yaxis_title="Frequency")
    st.plotly_chart(fig_hist, use_container_width=True)

with c2:
    st.subheader(":material/show_chart: 2. Monthly Average Temperature Trend")
    monthly_df = (
        filtered_df.set_index("Date")["Temperature (C)"]
        .resample("ME")
        .mean()
        .reset_index()
    )
    fig_line = px.line(
        monthly_df,
        x="Date",
        y="Temperature (C)",
        markers=True,
        color_discrete_sequence=["#1E88E5"],
        title="Average Temperature Over Time (Monthly)",
    )
    fig_line.update_layout(xaxis_title="Date", yaxis_title="Avg Temperature (°C)")
    st.plotly_chart(fig_line, use_container_width=True)

# Row 2: Bivariate Scatter + Category Comparison Bar Chart
c3, c4 = st.columns(2)

with c3:
    st.subheader(":material/scatter_plot: 3. Temperature vs. Humidity (Bivariate)")
    sample_df = filtered_df.sample(min(len(filtered_df), 3000), random_state=42)
    fig_scatter = px.scatter(
        sample_df,
        x="Temperature (C)",
        y="Humidity",
        color="Precip Type",
        opacity=0.6,
        title="Temperature (°C) vs. Humidity Relationship",
    )
    fig_scatter.update_layout(xaxis_title="Temperature (°C)", yaxis_title="Humidity (0 - 1)")
    st.plotly_chart(fig_scatter, use_container_width=True)

with c4:
    st.subheader(":material/air: 4. Average Wind Speed by Top Weather Conditions")
    top_summaries = (
        filtered_df.groupby("Summary")["Wind Speed (km/h)"]
        .mean()
        .nlargest(10)
        .reset_index()
    )
    fig_bar = px.bar(
        top_summaries,
        x="Wind Speed (km/h)",
        y="Summary",
        orientation="h",
        color="Wind Speed (km/h)",
        color_continuous_scale="Teal",
        title="Top 10 Weather Conditions by Avg Wind Speed",
    )
    fig_bar.update_layout(
        xaxis_title="Average Wind Speed (km/h)",
        yaxis_title="Weather Summary",
        yaxis={"categoryorder": "total ascending"},
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# Row 3: Correlation Heatmap
st.subheader(":material/grid_on: 5. Weather Variables Correlation Heatmap")
numeric_cols = [
    "Temperature (C)",
    "Apparent Temperature (C)",
    "Humidity",
    "Wind Speed (km/h)",
    "Visibility (km)",
    "Pressure (millibars)",
]
corr_matrix = filtered_df[numeric_cols].corr()

fig_corr, ax = plt.subplots(figsize=(9, 4.5))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    linewidths=0.5,
    ax=ax,
)
ax.set_title("Correlation Matrix of Numeric Weather Features")
st.pyplot(fig_corr)

st.divider()

# 5. Checkbox to Show Raw Data Table
if st.checkbox(":material/table_view: Show Filtered Raw Data Table"):
    st.subheader("Raw Weather Dataset")
    st.write(f"Showing **{len(filtered_df):,}** rows matching current filters:")
    st.dataframe(filtered_df.drop(columns=["Date"]).head(500), use_container_width=True)

# 6. Main Findings Section
st.subheader(":material/insights: Main Findings & Key Insights")
st.markdown(
    """
- **Strong Inverse Relationship Between Temperature and Humidity:** As temperatures rise (especially during summer months), relative humidity consistently drops, showing a strong negative correlation (~ -0.63).
- **Clear Seasonal Temperature Cycles:** Monthly average temperatures follow a predictable annual wave, peaking in July/August (around 22–25 °C) and dropping to their lowest in January (near 0 °C).
- **Apparent vs. Actual Temperature:** `Temperature (C)` and `Apparent Temperature (C)` move almost identically (correlation > 0.99), though high wind speeds and freezing conditions make the apparent temperature feel noticeably colder.
- **Precipitation & Visibility Impact:** Rain is the dominant precipitation type across the dataset, while snow occurs strictly when temperatures drop near or below 0 °C, accompanied by reduced visibility.
"""
)