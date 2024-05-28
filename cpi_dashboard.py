import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from yellowbrick.cluster import KElbowVisualizer
import streamlit as st
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import zscore 
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
warnings.filterwarnings('ignore')

#Sklearn libraries
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

df = pd.read_csv('all_india_cpi.csv')
df.set_index('month_year',inplace=True)
data_urban = df[df['Sector']=='Urban']
data_rural = df[df['Sector']=='Rural']
data_both = df[df['Sector']=='Rural+Urban']

data_both.index = pd.to_datetime(data_both.index)
data_rural.index = pd.to_datetime(data_rural.index)
data_urban.index = pd.to_datetime(data_urban.index)

data_both.drop(columns='Sector',inplace=True)
data_rural.drop(columns='Sector',inplace=True)
data_urban.drop(columns='Sector',inplace=True)

commodities = ['General index','Cereals and products', 'Meat and fish', 'Egg',
       'Milk and products', 'Oils and fats', 'Fruits', 'Vegetables',
       'Pulses and products', 'Sugar and Confectionery', 'Spices',
       'Non-alcoholic beverages', 'Prepared meals, snacks, sweets etc.',
       'Food and beverages', 'Pan, tobacco and intoxicants', 'Clothing',
       'Footwear', 'Clothing and footwear', 'Housing', 'Fuel and light',
       'Household goods and services', 'Health', 'Transport and communication',
       'Recreation and amusement', 'Education', 'Personal care and effects',
       'Miscellaneous']

for comm in commodities:
    data_both.fillna(data_both[comm].mean(),inplace=True)
    data_urban.fillna(data_urban[comm].mean(),inplace=True)
    data_rural.fillna(data_rural[comm].mean(),inplace=True)
    
def Trend_visualization():
    st.sidebar.header('Trend Visualization')
    select_sector = st.sidebar.selectbox('Select Sector',['Rural','Urban','Rural+Urban'])
    selected_category = st.sidebar.multiselect('Select Commodities',commodities,default='General index')
    
    if select_sector == 'Rural':
        commodity_data = data_rural
    elif select_sector == 'Urban':
        commodity_data = data_urban
    else:
        commodity_data = data_both
        
    plt.figure(figsize=(10,6))
    plt.plot(commodity_data.index, commodity_data[selected_category],label=selected_category)
    plt.title(f'{selected_category} over time')
    plt.xlabel('Month Year')
    plt.ylabel('Index')
    plt.grid(True)
    plt.axhline(y=commodity_data['General index'].mean(), color='r', linestyle='--', label='Mean CPI')
    plt.legend()
    st.pyplot(plt)
    
def correlation_plot():
    select_sector = st.sidebar.selectbox('Select Sector',['Rural','Urban','Rural+Urban'])

    
    if select_sector == 'Rural':
        commodity_data = data_rural
    elif select_sector == 'Urban':
        commodity_data = data_urban
    else:
        commodity_data = data_both
        
    plt.figure(figsize=(20,12))
    sns.heatmap(commodity_data.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix Heat map')
    st.pyplot(plt)
    
def decompose_time_series(data, commodity):
    result = seasonal_decompose(data[commodity], model='additive', period=12)
    return result
    
def timeseries_analysis():
    st.sidebar.header('Time-series Analysis')
    select_sector = st.sidebar.selectbox('Select Sector',['Rural','Urban','Rural+Urban'])
    selected_category = st.sidebar.selectbox('Select Commodities',commodities)
    
    if select_sector == 'Rural':
        commodity_data = data_rural
    elif select_sector == 'Urban':
        commodity_data = data_urban
    else:
        commodity_data = data_both
        
    decomposed = decompose_time_series(commodity_data, selected_category)

    # Plotting the decomposed time series
    st.title("Trend component")
    trend_component = decomposed.trend
    plt.figure(figsize=(12,6))
    plt.plot(commodity_data.index, trend_component)
    plt.xlabel('Time')
    plt.ylabel('Trend Component')
    plt.title(f'Trend Analysis of commodity {selected_category} CPI')
    st.pyplot(plt)
    
    st.title("Residual component")
    res_component = decomposed.resid
    plt.figure(figsize=(12,6))
    plt.plot(commodity_data.index, res_component)
    plt.xlabel('Time')
    plt.ylabel('Residual Component')
    plt.title(f'Residual Analysis of commodity {selected_category} CPI')
    st.pyplot(plt)
    
    st.title("Seasional component")
    season_component = decomposed.seasonal
    plt.figure(figsize=(12,6))
    plt.plot(commodity_data.index, season_component)
    plt.xlabel('Time')
    plt.ylabel('Seasion Component')
    plt.title(f'Seasionality Analysis of commodity {selected_category} CPI')
    st.pyplot(plt)
    

    seasonality_test_result = 'Stationary' if decomposed.seasonal.dropna().std() < 0.1 else 'Seasonal'
    st.header(f"The '{selected_category}' commodity is likely {seasonality_test_result}.")
    
    
def timeseries_forecasting():
    st.title('Time Series Forecasting')
    select_sector = st.sidebar.selectbox('Select Sector',['Rural','Urban','Rural+Urban'])
    selected_comm = st.sidebar.selectbox('Select Commodities',commodities)
    
    if select_sector == 'Rural':
        commodity_data = data_rural
    elif select_sector == 'Urban':
        commodity_data = data_urban
    else:
        commodity_data = data_both

    forecasting_data = commodity_data[selected_comm].values
        
    forecasting_data = forecasting_data.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(forecasting_data)

    # Create X, y datasets
    X, y = [], []
    for i in range(10, len(data_scaled)):
        X.append(data_scaled[i-10:i, 0])
        y.append(data_scaled[i, 0])

    X, y = np.array(X), np.array(y)

    # Reshape X to be [samples, time steps, features]
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    model = Sequential()
    model.add(LSTM(50, input_shape=(X.shape[1], 1)))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=20, batch_size=1, verbose=2)
    
    predictions = model.predict(X)
    predictions = scaler.inverse_transform(predictions)  # Reverting scaling
    plt.figure(figsize=(10, 6))
    plt.plot(forecasting_data, label='True')
    plt.plot(np.arange(10, len(forecasting_data)), predictions, label='Predicted')
    plt.legend()
    st.pyplot(plt)
    
    future_steps = 8  # to predict the next three months
    input_sequence = X[-1]  # the last sequence in the dataset

    predictions_future = []

    for _ in range(future_steps):
        # Use last sequence to predict the next value
        predicted = model.predict(input_sequence.reshape(1, -1, 1))

        # Append predicted value to the input_sequence
        input_sequence = np.append(input_sequence, predicted)

        # Remove the first value of the sequence to maintain the sequence length
        input_sequence = input_sequence[1:]

        # Append the predicted value to the list of future predictions
        predictions_future.append(scaler.inverse_transform(predicted)[0, 0])
        
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(forecasting_data)), forecasting_data, label='True')
    plt.plot(range(len(forecasting_data), len(forecasting_data) + future_steps), predictions_future, label='Forecasted', linestyle='dashed')
    plt.legend()
    st.pyplot(plt)
    
    future_direction = "increase" if predictions_future[-1] > predictions_future[0] else "decrease" if predictions_future[-1] < predictions_future[0] else "remain the same"    
    st.write(f"The forecasted values for the next coming months are likely to {future_direction}.")

def inflation_analysis():
    st.title('Inflation analysis')
    select_sector = st.sidebar.selectbox('Select Sector',['Rural','Urban','Rural+Urban'])
    selected_comm = st.sidebar.selectbox('Select Commodities',commodities)
    
    if select_sector == 'Rural':
        commodity_data = data_rural
    elif select_sector == 'Urban':
        commodity_data = data_urban
    else:
        commodity_data = data_both
        
    monthly_inflation_rate = commodity_data[selected_comm].pct_change() * 100
    yearly_inflation_rate = commodity_data[selected_comm].pct_change(12) * 100
    
    st.title(f'Yearly Inflation Rate {selected_comm}')
    plt.figure(figsize=(10, 6))
    plt.plot(yearly_inflation_rate.index, yearly_inflation_rate.values, label=f'Yearly Inflation Rate {selected_comm}', color='red')
    plt.title('Yearly Inflation Rate Over Time')
    plt.xlabel('Time')
    plt.ylabel('Inflation Rate (%)')
    plt.axhline(0, color='black', linewidth=0.5)  # Add horizontal line at y=0
    plt.legend(loc='best')
    st.pyplot(plt)
        
def volatility_analysis():
    st.title('Volatility Analysis')
    st.write("Volatility analysis helps us understand how the prices of commodities or categories fluctuate over time.")
    st.write("In this analysis, we calculate the rolling standard deviation of prices over a 12-month window to measure volatility.")
    st.write("Higher volatility indicates greater price fluctuations, while lower volatility suggests price stability.")

    select_sector = st.sidebar.selectbox('Select Sector',['Rural','Urban','Rural+Urban'])
    selected_comm = st.sidebar.selectbox('Select Commodities',commodities)
    
    if select_sector == 'Rural':
        commodity_data = data_rural
    elif select_sector == 'Urban':
        commodity_data = data_urban
    else:
        commodity_data = data_both
        
    volatility_df =commodity_data.std()
    volatility_df_cat = commodity_data[selected_comm].rolling(window=12).std()
    # print(volatility_df)
    st.write('Top 3 Highest volatility: ', volatility_df.nlargest(3))
    st.write('Top 3 Lowest volatility (Stable): ', volatility_df.nsmallest(3))

    st.header(f"Volatility visualization of {selected_comm}")
    # Create a figure and set its size
    plt.figure(figsize=(12, 6))

    # Plot the volatility graph
    plt.plot(volatility_df_cat.index, volatility_df_cat.values)
    plt.xlabel('Time')
    plt.ylabel('Volatility')
    plt.title(f'Volatility in {selected_comm} CPI')


    st.pyplot(plt)
    
def home_page():
    # Streamlit UI
    st.title('Consumer Price Index (CPI)')

    st.write("The Consumer Price Index (CPI) is a measure that examines the average change in prices paid by consumers "
             "for a basket of goods and services over time. It is a key economic indicator for assessing inflation or "
             "deflation trends and is used in various economic and policy decisions.")

    st.write("Two important points about CPI:")
    st.markdown("1. **Inflation Measurement**: CPI tracks inflation rates, helping policymakers, economists, and businesses "
                "assess the rate of inflation or deflation in an economy.")
    st.markdown("2. **Cost-of-Living Adjustments**: Many government programs, labor contracts, and pension plans are tied "
                "to the CPI. It is used to automatically adjust payments or wages to account for the rising cost of living.")

    st.write("Formula for CPI (Laspeyres Price Index):")
    st.latex(r"CPI = \left(\frac{\sum_{i=1}^{n} \text{Current Prices}_i}{\sum_{i=1}^{n} \text{Previous Prices}_i}\right) \times 100")
    
    st.header("Correlation Matrix Heatmap")
    correlation_plot()
    
def main():
    st.sidebar.title('Navigation')
    pages = {
        "Home Page":home_page,
        "Trend Visualization": Trend_visualization,
        "Calculate Volatility": volatility_analysis,
        "Inflation Analysis": inflation_analysis,
        "Timeseries Analysis": timeseries_analysis,
        "Timeseries Forecasting":timeseries_forecasting
    }
    
    selected_page = st.sidebar.radio("Go to", list(pages.keys()))
    pages[selected_page]()

if __name__ == "__main__":
    main()
