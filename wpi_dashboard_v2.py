#Libraries
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

#Data preparation
data = pd.read_csv('WPI_dataset.csv', index_col='COMM_IDX_MONTH', parse_dates=True)
#Creating dictionary of 22categories dataframe.
col_names = ['(A).  FOOD ARTICLES','(B).  NON-FOOD ARTICLES','(C).  MINERALS','(D). CRUDE PETROLEUM & NATURAL GAS','II FUEL & POWER',
            '(B). MINERAL OILS','(C). ELECTRICITY','III   MANUFACTURED PRODUCTS','(B). MANUFACTURE OF BEVERAGES','(C). MANUFACTURE OF TOBACCO PRODUCTS',
            '(D). MANUFACTURE OF TEXTILES','(E). MANUFACTURE OF WEARING APPAREL','(F). MANUFACTURE OF LEATHER AND RELATED PRODUCTS',
            '(G). MANUFACTURE OF WOOD AND OF PRODUCTS OF WOOD AND CORK  ','(H). MANUFACTURE OF PAPER AND PAPER PRODUCTS',
            '(I). PRINTING AND REPRODUCTION OF RECORDED MEDIA ','(J). MANUFACTURE OF CHEMICALS AND CHEMICAL PRODUCTS',
            '(K). MANUFACTURE OF PHARMACEUTICALS, MEDICINAL CHEMICAL AND BOTANICAL PRODUCTS','(L). MANUFACTURE OF RUBBER AND PLASTICS PRODUCTS',
            '(M). MANUFACTURE OF OTHER NON-METALLIC MINERAL PRODUCTS','(N). MANUFACTURE OF BASIC METALS','(O). MANUFACTURE OF FABRICATED METAL PRODUCTS, EXCEPT MACHINERY AND EQUIPMENT']
categories_dict  = {}

for i in range(len(col_names)-1):
    cl = col_names[i]
    start_col = col_names[i]
    end_col = col_names[i + 1]
    dataframe = data.loc[:, start_col:end_col]
    dataframe = dataframe.iloc[:, :-1]
    categories_dict [cl] = dataframe
last_df = data.loc[:,'(O). MANUFACTURE OF FABRICATED METAL PRODUCTS, EXCEPT MACHINERY AND EQUIPMENT':]
categories_dict ['(O). MANUFACTURE OF FABRICATED METAL PRODUCTS, EXCEPT MACHINERY AND EQUIPMENT'] = last_df

def trend_visualization_page():
    #1. Trend Visualization
    st.title('Interactive WPI Trend Visualization')

    # Task 1: Trend Visualization
    st.sidebar.header('Trend Visualization')
    selected_category = st.sidebar.selectbox('Select Category', ['Entire Dataset'] + list(categories_dict.keys()))
    if selected_category == 'Entire Dataset':
        selected_commodities = st.sidebar.multiselect('Select Commodities', data.columns.tolist(), default='ALL COMMODITIES')
    else:
        selected_commodities = st.sidebar.multiselect('Select Commodities', ['All in Category'] + categories_dict[selected_category].columns.tolist())
    if selected_commodities:
        plt.figure(figsize=(12, 6))

    if 'All in Category' in selected_commodities and selected_category != 'Entire Dataset':
        for commodity in categories_dict[selected_category].columns.tolist():
            plt.plot(data.index, categories_dict[selected_category][commodity], label=commodity)
    else:
        for commodity in selected_commodities:
            plt.plot(data.index, data[commodity], label=commodity)

    plt.axhline(y=data['ALL COMMODITIES'].mean(), color='r', linestyle='--', label='Mean WPI')
    plt.xlabel('Time')
    plt.ylabel('WPI')
    plt.title('WPI Trends Over Time')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    st.pyplot(plt)
    
def calculate_volatility_page():
    
    st.title('Volatility Analysis')
    st.write("Volatility analysis helps us understand how the prices of commodities or categories fluctuate over time.")
    st.write("In this analysis, we calculate the rolling standard deviation of prices over a 12-month window to measure volatility.")
    st.write("Higher volatility indicates greater price fluctuations, while lower volatility suggests price stability.")

    selected_category_volatility = st.sidebar.selectbox('Select Category for Volatility Calculation', list(categories_dict.keys()))
    selected_commodity_volatility = st.sidebar.selectbox('Select Commodity for Volatility Calculation', ['Entire Category'] + list(categories_dict[selected_category_volatility].columns))
    if selected_commodity_volatility == 'Entire Category':
        volatility_df = categories_dict[selected_category_volatility].std()
        volatility_df_cat = data[selected_category_volatility].rolling(window=12).std()
        st.write('Top 3 Highest volatility: ', volatility_df.nlargest(3))
        st.write('Top 3 Lowest volatility (Stable): ', volatility_df.nsmallest(3))
    else:
        volatility_df = data[selected_commodity_volatility].rolling(window=12).std()
        
        highest_volatility_date = volatility_df.nlargest(1).index[0].strftime('%B %d, %Y')
        highest_volatility_std = round(volatility_df.nlargest(1).values[0], 2)
        st.write(f"{selected_commodity_volatility} exhibited the highest volatility on {highest_volatility_date}, with a standard deviation of {highest_volatility_std}. This indicates significant price fluctuations for {selected_commodity_volatility} commodities during that period.")

    # Create a figure and set its size
    plt.figure(figsize=(12, 6))

    # Plot the volatility graph
    plt.plot(volatility_df.index, volatility_df.values)
    plt.xlabel('Time')
    plt.ylabel('Volatility')
    plt.title(f'Volatility in {selected_commodity_volatility} WPI')


    st.pyplot(plt)

def monthy_yearly_rate_change_analysis():
    st.title("Inflation trend Analysis")
    inflation_plot()
    
    selected_category_column = st.sidebar.selectbox('Select Category', list(categories_dict.keys()))
    selected_commodity_column = st.sidebar.selectbox('Select Commodity', ['Entire Category'] + list(categories_dict[selected_category_column].columns))
    
    
    if selected_commodity_column == 'Entire Category':
        st.header(f'Monthly and Yearly Inflation analysis of {selected_category_column}')
        rate_change_monthly = data[selected_category_column].pct_change() * 100  # Monthly rate of change
        rate_change_yearly = data[selected_category_column].pct_change(periods=12) * 100  # Yearly rate of change
    else:
        st.header(f'Monthly and Yearly Inflation analysis of {selected_commodity_column}')
        rate_change_monthly = data[selected_commodity_column].pct_change() * 100  
        rate_change_yearly = data[selected_commodity_column].pct_change(periods=12)
        
    # Monthly rate of change
    plt.figure(figsize=(12, 6))
    plt.title(f"Monthly Rate Change for {selected_category_column}")
    plt.xlabel("Date")
    plt.ylabel("Monthly Rate Change (%)")
    plt.plot(rate_change_monthly.index, rate_change_monthly.values, label="Monthly Rate Change", marker='o')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)
    
    # Yearly rate of change
    plt.figure(figsize=(12, 6))
    plt.title(f"Yearly Rate Change for {selected_category_column}")
    plt.xlabel("Date")
    plt.ylabel("Yearly Rate Change (%)")
    plt.plot(rate_change_yearly.index, rate_change_yearly.values, label="Yearly Rate Change", marker='o')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)
    
    # Find and print the maximum inflation year and month
    max_monthly_inflation = rate_change_monthly.idxmax()
    max_yearly_inflation = rate_change_yearly.idxmax()
    
    # st.write(f"Maximum Inflation occurred in {max_monthly_inflation.strftime('%B %Y')}")
    st.write(f"Maximum Inflation occurred in {max_yearly_inflation.strftime('%Y')}")

    
def correlation_analysis_commodity():
    #Correlation between Category index and commodities in that category
    st.title('Correlation Analysis')

    # Add information about correlation analysis
    st.write("Correlation analysis helps us understand the relationship between different commodities within a selected category.")
    st.write("The heatmap below shows the correlation between commodities in the selected category.")

    selected_category_correlation = st.sidebar.selectbox('Select Category for Correlation Calculation', list(categories_dict.keys()))
    correlation_df = categories_dict[selected_category_correlation]
    correlation_matrix = correlation_df.corr()
    
    plt.figure(figsize=(12, 10))
    mask = correlation_matrix == 1.0
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Correlation between Category-wise Overall Value and commodities in that Category')
    st.pyplot(plt)
    
    category_correlation_matrix = correlation_matrix[selected_category_correlation].drop(selected_category_correlation)
    top_corr_df = category_correlation_matrix.abs().sort_values(ascending=False)
    top_corr_df = top_corr_df[top_corr_df < 1.0]  
    top_corr = top_corr_df.head(3)
    
    st.write(f"The commodity with the maximum correlation to '{selected_category_correlation}' is:")
    st.write(f"{top_corr.index[0]} with correlation value: {top_corr.values[0]:.2f}")
    st.write(f"After that {top_corr.index[1]} with correlation value: {top_corr.values[1]:.2f}")
    st.write(f"After that {top_corr.index[2]} with correlation value: {top_corr.values[2]:.2f}")
    
def clustering_page():
    st.header("Cluster Analysis")
    st.write("Using clustering algorithms to group similar categories together.")
    excluded_columns = col_names + ['ALL COMMODITIES']
    commodities_data = data.drop(columns=excluded_columns).T
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(commodities_data)
    
    model = KMeans()
    visualizer = KElbowVisualizer(model, k=(1,12))
    visualizer.fit(scaled_data)
    # visualizer.show()

    # Get the optimal number of clusters
    n_clusters = visualizer.elbow_value_

    # Apply K-Means clustering with the optimal number of clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(scaled_data)
    
    commodities_data['Cluster'] = clusters
    
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(scaled_data)

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=clusters, cmap='viridis', alpha=0.6)
    plt.title('2D PCA of Commodities Clusters')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar(scatter)
    st.pyplot(plt)
    
def heatmap_page1(): #Correlation between categories and General index
    st.header("Heatmap")
    st.write("Heatmap showing the correlation between categories in the WPI dataset.")

    selected_columns = col_names + ['ALL COMMODITIES']
    correlation_df = data[selected_columns]
    correlation_matrix = correlation_df.corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Correlation between Category-wise Overall Value and General WPI Index')
    st.pyplot(plt)
    
    all_comm_correlation_matrix = correlation_matrix['ALL COMMODITIES'].drop('ALL COMMODITIES')
    top_corr_df = all_comm_correlation_matrix.abs().sort_values(ascending=False)
    top_corr = top_corr_df.head(2)
    st.write(f"{top_corr.index[1]} has Maximum Correlation with General Trend.")
    
def time_series_analysis():
    st.title('Time Series Analysis Page')

    def decompose_time_series(data, commodity):
        result = seasonal_decompose(data[commodity], model='additive', period=12)
        return result
    
    selected_category_decompose = st.sidebar.selectbox('Select Category for Time Series Decomposition', list(categories_dict.keys()))
    selected_commodity_decompose = st.sidebar.selectbox('Select Commodity for Time Series Decomposition', ['Entire Category'] + list(categories_dict[selected_category_decompose].columns))

    if selected_commodity_decompose == 'Entire Category':
        decomposed = decompose_time_series(categories_dict[selected_category_decompose], selected_category_decompose)
    else:
        decomposed = decompose_time_series(categories_dict[selected_category_decompose], selected_commodity_decompose)

    # Plotting the decomposed time series
    st.header("Tend: ")
    st.write("Trend Analysis helps us identify the long-term movement or direction in the time series data.")
    st.write("It provides insights into whether the data is increasing, decreasing, or stable over time.")
    st.header("Residual: ")
    st.write("Residual Analysis examines the differences between observed values and the values predicted by the trend and seasonal components.")
    st.write("It helps us understand the randomness or unexplained variability in the data.")
    
    st.header("Seasionality: ")
    st.write("Seasonality Analysis identifies repetitive patterns or cycles in the data that occur at regular intervals.")
    st.write("It helps us understand whether the data exhibits seasonality (repeating patterns) or not.")
    
    trend_component = decomposed.trend
    plt.figure(figsize=(12,6))
    plt.plot(categories_dict[selected_category_decompose].index, trend_component)
    plt.xlabel('Time')
    plt.ylabel('Trend Component')
    plt.title(f'Trend Analysis of commodity {selected_commodity_decompose} WPI')
    st.pyplot(plt)
    
    
    res_component = decomposed.resid
    plt.figure(figsize=(12,6))
    plt.plot(categories_dict[selected_category_decompose].index, res_component)
    plt.xlabel('Time')
    plt.ylabel('Residual Component')
    plt.title(f'Residual Analysis of commodity {selected_commodity_decompose} WPI')
    st.pyplot(plt)
    
    season_component = decomposed.seasonal
    plt.figure(figsize=(12,6))
    plt.plot(categories_dict[selected_category_decompose].index, season_component)
    plt.xlabel('Time')
    plt.ylabel('Seasion Component')
    plt.title(f'Seasionality Analysis of commodity {selected_commodity_decompose} WPI')
    st.pyplot(plt)
    
    if selected_commodity_decompose != 'Entire Category':
        seasonality_test_result = 'Stationary' if decomposed.seasonal.dropna().std() < 0.1 else 'Seasonal'
        st.header(f"The '{selected_commodity_decompose}' commodity is likely {seasonality_test_result}.")

    
def inflation_plot():
    st.title('Over all Inflation rate Trend')
    monthly_inflation_rate = data['ALL COMMODITIES'].pct_change() * 100
    yearly_inflation_rate = data['ALL COMMODITIES'].pct_change(12) * 100
    yearly_inflation_rate.dropna(inplace=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(yearly_inflation_rate.index, yearly_inflation_rate, label='Yearly General Inflation Rate', color='red')
    plt.title('Yearly Inflation Rate Over Time')
    plt.xlabel('Time')
    plt.ylabel('Inflation Rate (%)')
    plt.axhline(0, color='black', linewidth=0.5)  # Add horizontal line at y=0
    plt.legend(loc='best')
    st.pyplot(plt)
    
def time_series_forecasting():
    st.title('Time Series Forecasting')
    st.write("Time series forecasting involves predicting future values based on past data.")
    st.write("In this process, the model learns patterns and trends from historical data to make future predictions.")

    selected_category_tf = st.sidebar.selectbox('Select Category for Time series Forecasting', list(categories_dict.keys()))
    selected_commodity_tf = st.sidebar.selectbox('Select Commodity for Time series Forecasting', ['General Index'] + ['Entire Category'] + list(categories_dict[selected_category_tf].columns))
    
    if selected_commodity_tf == 'General Index':
        forecasting_data = data['ALL COMMODITIES'].values
    elif selected_commodity_tf == 'Entire Category':
        forecasting_data = data[selected_category_tf].values
    else:
        forecasting_data = data[selected_commodity_tf].values
        
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
    
    
def home_page():
    # Create a Streamlit app title
    st.title("Wholesale Price Index Dataset Advanced Data Analysis")
    # Heading for WPI Definition
    st.header("Definition of Wholesale Price Index (WPI)")
    st.write("The Wholesale Price Index (WPI) is a measure of the average change in the prices of a basket of goods and services typically traded in bulk at the wholesale level. It is used to track inflation and economic trends.")
    st.write("The formula for calculating the WPI is as follows:")
    st.latex(r'WPI = \frac{\sum(P_{i} \cdot W_{i})}{\sum(W_{i})}')
    
    # Bullet Points for Interpretation
    st.header("Interpreting WPI Data")
    st.write("You can interpret WPI data in the following ways:")
    st.markdown("- **Inflation Analysis**: WPI helps in monitoring inflation trends by tracking changes in wholesale prices over time.")
    st.markdown("- **Economic Trends**: WPI data can provide insights into the overall health of the economy, as rising prices may indicate economic growth or overheating.")
    st.markdown("- **Sector-Specific Analysis**: WPI can be used to analyze price movements in specific sectors, helping businesses and policymakers make informed decisions.")

    st.header("Sample Data (First 5 Rows)")
    st.dataframe(data.head(5))
    
    heatmap_page1()
    clustering_page()
    
def main():
    st.sidebar.title('Navigation')
    pages = {
        "Home Page":home_page,
        "Trend Visualization": trend_visualization_page,
        "Calculate Volatility": calculate_volatility_page,
        "Correlation Analysis": correlation_analysis_commodity,
        "Inflation Analysis": monthy_yearly_rate_change_analysis,
        "Timeseries Analysis": time_series_analysis,
        "Timeseries Forecasting":time_series_forecasting
    }
    
    selected_page = st.sidebar.radio("Go to", list(pages.keys()))
    pages[selected_page]()

if __name__ == "__main__":
    main()
