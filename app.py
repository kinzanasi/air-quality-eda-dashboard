import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

st.set_page_config(
	page_title="Air Quality Analysis & Monitoring",
	page_icon="🌬️",
	layout="wide"
)

# ---------------------------------------------------------
# 1. DATA LOADING & PREPROCESSING
# ---------------------------------------------------------
@st.cache_data
def load_data():
	df = pd.read_csv('air_quality_dataset.csv')

	# Clean duplicates safely for display/analysis
	df_clean = df.drop_duplicates().copy()

	def classify_pm25(val):
		if pd.isna(val):
			return 'Unknown'
		elif val <= 35.4:
			return 'Good'
		elif val <= 55.4:
			return 'Moderate'
		else:
			return 'Poor'

	df_clean['Air_Quality_Class'] = df_clean['PM2.5'].apply(classify_pm25)
	return df_clean

df = load_data()

# ---------------------------------------------------------
# 2. DASHBOARD HEADER
# ---------------------------------------------------------
st.title("🌬️ Air Quality Analysis Dashboard")
st.markdown("Interactive presentation dashboard based on **Colab Data Cleaning, EDA & Key Insights**.")
st.divider()

# ---------------------------------------------------------
# 3. SIDEBAR FILTERS (Interactive Demo)
# ---------------------------------------------------------
st.sidebar.header("🔍 Filter Options")

available_cities = sorted(df['City'].dropna().unique().tolist())
selected_cities = st.sidebar.multiselect(
	"Select City:",
	options=available_cities,
	default=available_cities[:3] if len(available_cities) >= 3 else available_cities
)

available_classes = [c for c in df['Air_Quality_Class'].dropna().unique() if c != 'Unknown']
selected_classes = st.sidebar.multiselect(
	"Select Air Quality Category:",
	options=available_classes,
	default=available_classes
)

filtered_df = df[
	(df['City'].isin(selected_cities)) &
	(df['Air_Quality_Class'].isin(selected_classes))
]

if filtered_df.empty:
	st.warning("Selected filters ke hisab se koi data nahi mila. Please sidebar mein kam se kam ek City aur ek Category select karein.")
	st.stop()

# ---------------------------------------------------------
# 4. KEY METRICS / EXECUTIVE SUMMARY
# ---------------------------------------------------------
st.subheader("📌 Key Summary Metrics")
col1, col2, col3, col4 = st.columns(4)

with col1:
	st.metric("Total Samples", len(filtered_df))
with col2:
	st.metric("Avg PM2.5", f"{filtered_df['PM2.5'].mean():.2f} µg/m³")
with col3:
	st.metric("Avg Temperature", f"{filtered_df['Temperature'].mean():.2f} °C")
with col4:
	st.metric("Avg AQI", f"{filtered_df['AQI'].mean():.2f}")

st.divider()

# ---------------------------------------------------------
# 5. PRESENTATION FLOW - VISUALIZATIONS & ANALYSIS
# ---------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
	"📈 Q1 & Q2: Distributions",
	"🔄 Q4 & Q6: Correlations",
	"📊 Q7: Group Comparison",
	"💡 Presentation Insights"
])

# --- TAB 1: Univariate Analysis ---
with tab1:
	st.header("Univariate Distribution Analysis")
	col_a, col_b = st.columns(2)

	with col_a:
		st.subheader("Q1: PM2.5 Distribution")
		fig1, ax1 = plt.subplots(figsize=(6, 4))
		sns.histplot(filtered_df['PM2.5'].dropna(), kde=True, color='skyblue', ax=ax1)
		ax1.set_title("Distribution of PM2.5 Concentrations")
		ax1.set_xlabel("PM2.5 (µg/m³)")
		st.pyplot(fig1)
		plt.close(fig1)

	with col_b:
		st.subheader("Q2: Air Quality Class Distribution")
		fig2, ax2 = plt.subplots(figsize=(6, 4))
		sns.countplot(
			x='Air_Quality_Class',
			hue='Air_Quality_Class',
			data=filtered_df,
			palette='Set2',
			legend=False,
			ax=ax2
		)
		ax2.set_title("Air Quality Class Distribution")
		ax2.set_xlabel("Air Quality Class")
		st.pyplot(fig2)
		plt.close(fig2)

# --- TAB 2: Bivariate & Correlation Analysis ---
with tab2:
	st.header("Bivariate & Linear Relationships")
	col_c, col_d = st.columns(2)

	with col_c:
		st.subheader("Q4: Temperature vs PM2.5")
		fig3, ax3 = plt.subplots(figsize=(6, 4))
		sns.scatterplot(
			x='Temperature',
			y='PM2.5',
			hue='Air_Quality_Class',
			data=filtered_df,
			ax=ax3
		)
		ax3.set_title("Temperature vs PM2.5 Levels")
		st.pyplot(fig3)
		plt.close(fig3)

	with col_d:
		st.subheader("Q6: Pollutants Correlation Heatmap")
		numeric_df = filtered_df.select_dtypes(include=[np.number])
		fig4, ax4 = plt.subplots(figsize=(7, 5))
		sns.heatmap(numeric_df.corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=ax4)
		ax4.set_title("Pollutants Correlation Heatmap")
		st.pyplot(fig4)
		plt.close(fig4)

# --- TAB 3: Group Comparison ---
with tab3:
	st.header("Group Comparison Across Categories")
	st.subheader("Q7: Temperature Variation Across Air Quality Classes")

	fig5, ax5 = plt.subplots(figsize=(8, 4))
	sns.boxplot(
		x='Air_Quality_Class',
		y='Temperature',
		hue='Air_Quality_Class',
		data=filtered_df,
		palette='Pastel1',
		legend=False,
		ax=ax5
	)
	ax5.set_title("Temperature Variation Across Air Quality Classes")
	st.pyplot(fig5)
	plt.close(fig5)

# --- TAB 4: Main Takeaways ---
with tab4:
	st.header("💡 Final Presentation Takeaways & Conclusions")

	st.success("1. **Pollutant Correlation:** Strong relationships appear among particulate and gaseous air quality variables.")
	st.warning("2. **Air Quality Skew:** Data distribution shows clear concentration differences across Good, Moderate, and Poor categories.")
	st.info("3. **Temperature Impact:** Temperature patterns vary across air quality classes and help contextualize pollution trends.")
