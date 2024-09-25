import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
import os
from io import StringIO
import sys
import warnings
warnings.filterwarnings('ignore')



st.set_page_config(page_title="Superstore!!!", page_icon=":bar_chart:",layout="wide")

st.title(" :bar_chart: SuperStore EDA")
st.markdown('<style>div.block-container{padding-top:2.5rem;}</style>',unsafe_allow_html=True)

fl = st.file_uploader(":file_folder: Upload a file",type=(["csv","txt","xlsx","xls"]))
if fl is not None:
    filename = fl.name
    st.write(filename)
    df = pd.read_csv(filename, encoding = "ISO-8859-1")
else:
    os.chdir(r"E:\Python\Projects\Assignment")
    df = pd.read_csv("Superstore.csv", encoding = "ISO-8859-1")

col1, col2 = st.columns((2))
df["Order Date"] = pd.to_datetime(df["Order Date"])

# Getting the min and max date 
startDate = pd.to_datetime(df["Order Date"]).min()
endDate = pd.to_datetime(df["Order Date"]).max()

with col1:
    date1 = pd.to_datetime(st.date_input("Start Date", startDate))

with col2:
    date2 = pd.to_datetime(st.date_input("End Date", endDate))

df = df[(df["Order Date"] >= date1) & (df["Order Date"] <= date2)].copy()

st.sidebar.header("Choose your filter: ")
# Create for Region
region = st.sidebar.multiselect("Pick your Region", df["Region"].unique())
if not region:
    df2 = df.copy()
else:
    df2 = df[df["Region"].isin(region)]

# Create for State
state = st.sidebar.multiselect("Pick the State", df2["State"].unique())
if not state:
    df3 = df2.copy()
else:
    df3 = df2[df2["State"].isin(state)]

# Create for City
city = st.sidebar.multiselect("Pick the City",df3["City"].unique())



# Filter the data based on Region, State and City

if not region and not state and not city:
    filtered_df = df
elif not state and not city:
    filtered_df = df[df["Region"].isin(region)]
elif not region and not city:
    filtered_df = df[df["State"].isin(state)]
elif state and city:
    filtered_df = df3[df["State"].isin(state) & df3["City"].isin(city)]
elif region and city:
    filtered_df = df3[df["Region"].isin(region) & df3["City"].isin(city)]
elif region and state:
    filtered_df = df3[df["Region"].isin(region) & df3["State"].isin(state)]
elif city:
    filtered_df = df3[df3["City"].isin(city)]
else:
    filtered_df = df3[df3["Region"].isin(region) & df3["State"].isin(state) & df3["City"].isin(city)]

category_df = filtered_df.groupby(by = ["Category"], as_index = False)["Sales"].sum()

with col1:
    st.subheader("Category wise Sales")
    fig = px.bar(category_df, x = "Category", y = "Sales", text = ['${:,.2f}'.format(x) for x in category_df["Sales"]],
                 template = "seaborn")
    st.plotly_chart(fig,use_container_width=True, height = 200)

with col2:
    st.subheader("Region wise Sales")
    fig = px.pie(filtered_df, values = "Sales", names = "Region", hole = 0.5)
    fig.update_traces(text = filtered_df["Region"], textposition = "outside")
    st.plotly_chart(fig,use_container_width=True)

cl1, cl2 = st.columns((2))
with cl1:
    with st.expander("Category_ViewData"):
        st.write(category_df.style.background_gradient(cmap="Blues"))
        csv = category_df.to_csv(index = False).encode('utf-8')
        st.download_button("Download Data", data = csv, file_name = "Category.csv", mime = "text/csv",
                            help = 'Click here to download the data as a CSV file')

with cl2:
    with st.expander("Region_ViewData"):
        region = filtered_df.groupby(by = "Region", as_index = False)["Sales"].sum()
        st.write(region.style.background_gradient(cmap="Oranges"))
        csv = region.to_csv(index = False).encode('utf-8')
        st.download_button("Download Data", data = csv, file_name = "Region.csv", mime = "text/csv",
                        help = 'Click here to download the data as a CSV file')

filtered_df["month_year"] = filtered_df["Order Date"].dt.to_period("M")
st.subheader('Time Series Analysis')

linechart = pd.DataFrame(filtered_df.groupby(filtered_df["month_year"].dt.strftime("%Y : %b"))["Sales"].sum()).reset_index()
fig2 = px.line(linechart, x = "month_year", y="Sales", labels = {"Sales": "Amount"},height=500, width = 1000,template="gridon")
st.plotly_chart(fig2,use_container_width=True)

with st.expander("View Data of TimeSeries:"):
    st.write(linechart.T.style.background_gradient(cmap="Blues"))
    csv = linechart.to_csv(index=False).encode("utf-8")
    st.download_button('Download Data', data = csv, file_name = "TimeSeries.csv", mime ='text/csv')

# Create a treem based on Region, category, sub-Category
st.subheader("Hierarchical view of Sales using TreeMap")
fig3 = px.treemap(filtered_df, path = ["Region","Category","Sub-Category"], values = "Sales",hover_data = ["Sales"],
                  color = "Sub-Category")
fig3.update_layout(width = 800, height = 650)
st.plotly_chart(fig3, use_container_width=True)

chart1, chart2 = st.columns((2))
with chart1:
    st.subheader('Segment wise Sales')
    fig = px.pie(filtered_df, values = "Sales", names = "Segment", template = "plotly_dark")
    fig.update_traces(text = filtered_df["Segment"], textposition = "inside")
    st.plotly_chart(fig,use_container_width=True)

with chart2:
    st.subheader('Category wise Sales')
    fig = px.pie(filtered_df, values = "Sales", names = "Category", template = "gridon")
    fig.update_traces(text = filtered_df["Category"], textposition = "inside")
    st.plotly_chart(fig,use_container_width=True)

import plotly.figure_factory as ff
st.subheader(":point_right: Month wise Sub-Category Sales Summary")
with st.expander("Summary_Table"):
    df_sample = df[0:5][["Region","State","City","Category","Sales","Profit","Quantity"]]
    fig = ff.create_table(df_sample, colorscale = "Cividis")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("Month wise sub-Category Table")
    filtered_df["month"] = filtered_df["Order Date"].dt.month_name()
    sub_category_Year = pd.pivot_table(data = filtered_df, values = "Sales", index = ["Sub-Category"],columns = "month")
    st.write(sub_category_Year.style.background_gradient(cmap="Blues"))

# Create a scatter plot
data1 = px.scatter(filtered_df, x = "Sales", y = "Profit", size = "Quantity")
data1['layout'].update(title="Relationship between Sales and Profits using Scatter Plot.",
                       titlefont = dict(size=20),xaxis = dict(title="Sales",titlefont=dict(size=19)),
                       yaxis = dict(title = "Profit", titlefont = dict(size=19)))
st.plotly_chart(data1,use_container_width=True)

with st.expander("View Data"):
    st.write(filtered_df.iloc[:500,1:20:2].style.background_gradient(cmap="Oranges"))

# Download orginal DataSet
csv = df.to_csv(index = False).encode('utf-8')
st.download_button('Download Data', data = csv, file_name = "Data.csv",mime = "text/csv")


st.subheader(":point_right: Exploratory Data Analysis")
# Capture the output of df.info()
output = StringIO()
original_stdout = sys.stdout # Save the original standard output
sys.stdout = output # Redirect standard output to the StringIO object
df.info() # This will write to the StringIO object instead of standard output
sys.stdout = original_stdout 
st.write("##### DataFrame Information:")
st.text(output.getvalue())


eda1, eda2 = st.columns((2))
with eda1:
    st.write("##### Checking for null values:")
    missing_values_summary = df.isna().sum()
    st.text(missing_values_summary)

with eda2:
    st.write("##### Overviewing the datatypes:")
    data_types = df.dtypes
    st.text(data_types)


st.write("##### Descriptive statistics:")
statistics = df.describe()
st.dataframe(statistics)


df[['Ship Mode', 'Segment', 'Country', 'City', 'State', 'Region', 'Category', 'Sub-Category']] = df[['Ship Mode', 'Segment', 'Country', 'City', 'State', 'Region', 'Category', 'Sub-Category']].apply(lambda col:pd.Categorical(col).codes)

df_clean = df.drop(['Order ID', 'Order Date', 'Ship Date', 'Customer ID', 'Customer Name', 'Product ID', 'Product Name', 'Country','month_year', 'month'], axis=1)

import matplotlib.pyplot as plt
import seaborn as sns

st.subheader(":point_right: Correlation Analysis")
# Compute the correlation matrix
#st.text(X)

# Compute the correlation matrix
correlation_matrix = df_clean.corr()

# Create a heatmap using Plotly Express
fig = px.imshow(correlation_matrix,
                labels=dict(x="Features", y="Features", color="Correlation"),
                x=correlation_matrix.columns,
                y=correlation_matrix.columns,
                color_continuous_scale="twilight")

# Add annotations for correlation values with bold and black text
for i in range(len(correlation_matrix.columns)):
    for j in range(len(correlation_matrix.columns)):
        fig.add_annotation(
            x=i, y=j,
            text=str(round(correlation_matrix.iloc[i, j], 2)),
            font=dict(size=12, color='black'), # Updated font attributes
            showarrow=False,
            xref="x", yref="y")

# Modify layout to increase figure size
fig.update_layout(
    autosize=False,
    width=800,
    height=800
)

# Show the Plotly figure in Streamlit
st.plotly_chart(fig)



st.subheader(":point_right: Initiation of Machine Learning Regression Analysis")

import hvplot.pandas

X_profit = df_clean.drop(["Profit"],axis=1)
y_profit = df_clean[["Profit"]]    

X_sales = df_clean.drop(["Sales"],axis=1)
y_sales = df_clean[["Sales"]]

#Splitting of Variable
from sklearn.model_selection import train_test_split
X_trainPR, X_testPR, y_trainP, y_testP = train_test_split(X_profit, y_profit, test_size=0.3, random_state=42)

X_trainSA, X_testSA, y_trainS, y_testS = train_test_split(X_sales, y_sales, test_size=0.3, random_state=42)

#Model Development
from sklearn import metrics
from sklearn.model_selection import cross_val_score

def cross_valP(model):
    pred = cross_val_score(model, X_profit, y_profit, cv=10)
    return pred.mean()

def cross_valS(model):
    pred = cross_val_score(model, X_sales, y_sales, cv=10)
    return pred.mean()

def print_evaluate(true, predicted):  
    mae = metrics.mean_absolute_error(true, predicted)
    mse = metrics.mean_squared_error(true, predicted)
    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
    r2_square = metrics.r2_score(true, predicted)
    return f'MAE: {mae}\nMSE: {mse}\nRMSE: {rmse}\nR2 Square: {r2_square}\n__________________________________'

    
def evaluate(true, predicted):
    mae = metrics.mean_absolute_error(true, predicted)
    mse = metrics.mean_squared_error(true, predicted)
    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
    r2_square = metrics.r2_score(true, predicted)
    return mae, mse, rmse, r2_square

#Model Pipelining and Standard Scaler
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
pipeline = Pipeline([('std_scalar', StandardScaler())])
X_trainP = pipeline.fit_transform(X_trainPR)
X_testP = pipeline.transform(X_testPR)

X_trainS = pipeline.fit_transform(X_trainSA)
X_testS = pipeline.transform(X_testSA)

ML_Profit, ML_Sales = st.columns((2))
with ML_Profit:
    st.write("##### Implementing Linear Regression on Profit:")
    #Implementing Linear Regression
    from sklearn.linear_model import LinearRegression
    lin_regP = LinearRegression()
    lin_regP.fit(X_trainP,y_trainP)
    LinearRegression()
    Inter = lin_regP.intercept_
    st.text(f'Intercept: {Inter}')
    #Predicting data
    pred = lin_regP.predict(X_testP)

    #Obtaining accuracy of the model

    test_predP = lin_regP.predict(X_testP)
    train_predP = lin_regP.predict(X_trainP)

    st.write('Test set evaluation:\n_____________________________________')
    evaluation_textP = print_evaluate(y_testP, test_predP)
    st.text(evaluation_textP)

    st.write('Train set evaluation:\n_____________________________________')
    evaluation_textP = print_evaluate(y_trainP, train_predP)
    st.text(evaluation_textP)

    results_dfP = pd.DataFrame(data=[["Linear Regression", *evaluate(y_testP, test_predP) , cross_valP(LinearRegression())]], columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square', "Cross Validation"])

    st.dataframe(results_dfP)


    st.write("##### Implementing Robust Regression on Profit:")

    from sklearn.linear_model import RANSACRegressor, LinearRegression
    modelP = RANSACRegressor(LinearRegression(), max_trials=100)
    modelP.fit(X_trainP, y_trainP)

    #Prediction of data
    test_pred1P = modelP.predict(X_testP)
    train_pred1P = modelP.predict(X_trainP)

    #Obtaining accuracy of the model
    st.write('Test set evaluation:\n_____________________________________')
    evaluation_text1P = print_evaluate(y_testP, test_pred1P)
    st.text(evaluation_text1P)

    st.write('Train set evaluation:\n_____________________________________')
    evaluation_text1P = print_evaluate(y_trainP, train_pred1P)
    st.text(evaluation_text1P)

    results_df_1P = pd.DataFrame(data=[["Robust Regression", *evaluate(y_testP, test_pred1P) , cross_valP(RANSACRegressor())]], 
                                columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square', "Cross Validation"])


    results_dfP = results_dfP._append(results_df_1P, ignore_index=True)
    st.dataframe(results_dfP)


    st.write("##### Implementing Ridge Regression on Profit:")

    from sklearn.linear_model import Ridge
    model1P = Ridge(alpha=100, solver='cholesky', tol=0.0001, random_state=42)
    model1P.fit(X_trainP, y_trainP)

    #Prediction of data
    test_pred2P = model1P.predict(X_testP)
    train_pred2P = model1P.predict(X_trainP)

    #Obtaining accuracy of the model
    st.write('Test set evaluation:\n_____________________________________')
    evaluation_text2P = print_evaluate(y_testP, test_pred2P)
    st.text(evaluation_text2P)

    st.write('Train set evaluation:\n_____________________________________')
    evaluation_text2P = print_evaluate(y_trainP, train_pred2P)
    st.text(evaluation_text2P)

    results_df_3P = pd.DataFrame(data=[["Ridge Regression", *evaluate(y_testP, test_pred2P) , cross_valP(Ridge())]], 
                                columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square', "Cross Validation"])

    results_dfP = results_dfP._append(results_df_3P, ignore_index=True)
    st.dataframe(results_dfP)


    st.write("##### Implementing LASSO Regression on Profit:")

    from sklearn.linear_model import Lasso
    model2P = Lasso(alpha=0.1, 
                precompute=True,
                positive=True, 
                selection='random',
                random_state=42)
    model2P.fit(X_trainP, y_trainP)

    #Prediction of data
    test_pred3P = model2P.predict(X_testP)
    train_pred3P = model2P.predict(X_trainP)

    #Obtaining accuracy of the model

    st.write('Test set evaluation:\n_____________________________________')
    evaluation_text3P = print_evaluate(y_testP, test_pred3P)
    st.text(evaluation_text3P)

    st.write('Train set evaluation:\n_____________________________________')
    evaluation_text3P = print_evaluate(y_trainP, train_pred3P)
    st.text(evaluation_text3P)

    results_df_4P = pd.DataFrame(data=[["Lasso Regression", *evaluate(y_testP, test_pred3P) , cross_valP(Lasso())]], 
                                columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square', "Cross Validation"])

    results_dfP = results_dfP._append(results_df_4P, ignore_index=True)

    st.dataframe(results_dfP)

    st.write("##### Implementing Elastic Net Regression on Profit:")

    from sklearn.linear_model import ElasticNet
    model3P = ElasticNet(alpha=0.1, l1_ratio=0.9, selection='random', random_state=42)
    model3P.fit(X_trainP, y_trainP)

    #Prediction of data
    test_pred4P = model3P.predict(X_testP)
    train_pred4P = model3P.predict(X_trainP)

    #Obtaining accuracy of the model

    st.write('Test set evaluation:\n_____________________________________')
    evaluation_text4P = print_evaluate(y_testP, test_pred4P)
    st.text(evaluation_text4P)

    st.write('Train set evaluation:\n_____________________________________')
    evaluation_text4P = print_evaluate(y_trainP, train_pred4P)
    st.text(evaluation_text4P)

    results_df_5P = pd.DataFrame(data=[["Elastic Net Regression", *evaluate(y_testP, test_pred4P) , cross_valP(ElasticNet())]], 
                                columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square', "Cross Validation"])

    results_dfP = results_dfP._append(results_df_5P, ignore_index=True)
    st.dataframe(results_dfP)

    st.write("##### Implementing Stochastic Gradient Descent on Profit:")
    from sklearn.linear_model import SGDRegressor
    sgd_regP = SGDRegressor(n_iter_no_change=250, penalty=None, eta0=0.0001, max_iter=100000)
    sgd_regP.fit(X_trainP, y_trainP)

    #Prediction of data
    test_pred5P = sgd_regP.predict(X_testP)
    train_pred5P = sgd_regP.predict(X_trainP)

    #Obtaining accuracy of the model

    st.write('Test set evaluation:\n_____________________________________')
    evaluation_text5P = print_evaluate(y_testP, test_pred5P)
    st.text(evaluation_text5P)

    st.write('Train set evaluation:\n_____________________________________')
    evaluation_text5P = print_evaluate(y_trainP, train_pred5P)
    st.text(evaluation_text5P)

    results_df_6P = pd.DataFrame(data=[["Stochastic Gradient Descent", *evaluate(y_testP, test_pred5P), cross_valP(SGDRegressor())]], 
                                columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square', 'Cross Validation'])

    results_dfP = results_dfP._append(results_df_6P, ignore_index=True)
    st.dataframe(results_dfP)

    st.write("##### Implementing Random Forest Regressor on Profit:")
    from sklearn.ensemble import RandomForestRegressor
    rf_regP = RandomForestRegressor(n_estimators=1000)
    rf_regP.fit(X_trainP, y_trainP)

    #Prediction of data

    test_pred6P = rf_regP.predict(X_testP)
    train_pred6P = rf_regP.predict(X_trainP)

    #Obtaining accuracy of the model

    st.write('Test set evaluation:\n_____________________________________')
    evaluation_text6P = print_evaluate(y_testP, test_pred6P)
    st.text(evaluation_text6P)

    st.write('Train set evaluation:\n_____________________________________')
    evaluation_text6P = print_evaluate(y_trainP, train_pred6P)
    st.text(evaluation_text6P)

    results_df_7P = pd.DataFrame(data=[["Random Forest Regressor", *evaluate(y_testP, test_pred6P), cross_valP(RandomForestRegressor())]], 
                                columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square', 'Cross Validation'])

    results_dfP = results_dfP._append(results_df_7P, ignore_index=True)
    st.dataframe(results_dfP)

    st.write("##### Implementing Decision Tree Regressor on Profit:")
    from sklearn.tree import DecisionTreeRegressor

    # Initialize the model
    tree_regressorP = DecisionTreeRegressor(random_state=42)

    # Fit the model to training data
    tree_regressorP.fit(X_trainP, y_trainP)

    # Make predictions
    test_pred7P = tree_regressorP.predict(X_testP)
    train_pred7P = tree_regressorP.predict(X_trainP)

    #Obtaining accuracy of the model

    st.write('Test set evaluation:\n_____________________________________')
    evaluation_text7P = print_evaluate(y_testP, test_pred7P)
    st.text(evaluation_text7P)

    st.write('Train set evaluation:\n_____________________________________')
    evaluation_text7P = print_evaluate(y_trainP, train_pred7P)
    st.text(evaluation_text7P)

    results_df_8P = pd.DataFrame(data=[["Decision Tree Regressor", *evaluate(y_testP, test_pred7P), cross_valP(DecisionTreeRegressor())]], 
                                columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square', 'Cross Validation'])

    results_dfP = results_dfP._append(results_df_8P, ignore_index=True)
    st.dataframe(results_dfP)

    st.write("##### Implementing Support Vector Machine on Profit:")
    from sklearn.svm import SVR
    svm_regP = SVR(kernel='linear', C=1, epsilon=0.1)
    svm_regP.fit(X_trainP, y_trainP)

    #Prediction of data
    test_pred8P = svm_regP.predict(X_testP)
    train_pred8P = svm_regP.predict(X_trainP)

    #Obtaining accuracy of the model

    st.write('Test set evaluation:\n_____________________________________')
    evaluation_text8P = print_evaluate(y_testP, test_pred8P)
    st.text(evaluation_text8P)

    st.write('Train set evaluation:\n_____________________________________')
    evaluation_text8P = print_evaluate(y_trainP, train_pred8P)
    st.text(evaluation_text8P)

    results_df_9P = pd.DataFrame(data=[["SVM Regressor", *evaluate(y_testP, test_pred8P), cross_valP(SVR())]], 
                                columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square', 'Cross Validation'])

    results_dfP = results_dfP._append(results_df_9P, ignore_index=True)
    st.dataframe(results_dfP)

    st.write("##### Implementing Polynomial Regression on Profit:")

    from sklearn.preprocessing import PolynomialFeatures
    poly_regP = PolynomialFeatures(degree=2)
    X_train_2_dP = poly_regP.fit_transform(X_trainP)
    X_test_2_dP = poly_regP.transform(X_testP)

    # Creating a new Linear Regression model instance for polynomial features
    lin_reg_polyP = LinearRegression()

    # Fitting this new model with the transformed training data
    lin_reg_polyP.fit(X_train_2_dP, y_trainP)

    # Prediction of data with the new model instance
    test_pred9P = lin_reg_polyP.predict(X_test_2_dP)
    train_pred9P = lin_reg_polyP.predict(X_train_2_dP)

    #Obtaining accuracy of the model

    st.write('Test set evaluation:\n_____________________________________')
    evaluation_text9P = print_evaluate(y_testP, test_pred9P)
    st.text(evaluation_text9P)

    st.write('Train set evaluation:\n_____________________________________')
    evaluation_text9P = print_evaluate(y_trainP, train_pred9P)
    st.text(evaluation_text9P)

    results_df_10P = pd.DataFrame(data=[["Polynomail Regression", *evaluate(y_testP, test_pred9P), 0]], 
                                columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square', 'Cross Validation'])

    results_dfP = results_dfP._append(results_df_10P, ignore_index=True)
    st.dataframe(results_dfP)

    st.write("##### Implementing Artficial Neural Network on Profit:")

    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    import time

    X_train1P = np.array(X_trainP)
    X_test1P = np.array(X_testP)
    y_train1P = np.array(y_trainP)
    y_test1P = np.array(y_testP)

    model4P = Sequential()

    model4P.add(Dense(X_trainP.shape[1], activation='relu'))
    model4P.add(Dense(32, activation='relu'))
    model4P.add(Dense(64, activation='relu'))
    model4P.add(Dense(128, activation='relu'))
    model4P.add(Dense(512, activation='relu'))
    model4P.add(Dropout(0.1))
    model4P.add(Dense(1))
    model4P.compile(optimizer=Adam(0.00001), loss='mse')

    # Create a Streamlit text area to display training logs
    logs_areaP = st.empty()

    # Train your model and capture training logs
    model_historyP = model4P.fit(X_train1P, y_train1P,
                            validation_data=(X_test1P, y_test1P),
                            batch_size=1,
                            epochs=100,
                            verbose=2)  # Set verbose to 2 to capture training logs

    # Display training completion message
    st.success('Training completed!')

    # Accumulate training logs in a list
    training_logsP = []

    # Display training logs in real-time with Markdown and HTML
    for epoch in range(100):
        logs = f"Epoch {epoch + 1}/100\n"
        
        # Check if the epoch index exists in model_history.history['loss']
        if epoch < len(model_historyP.history['loss']):
            training_loss = model_historyP.history['loss'][epoch]
            logs += f"6995/6995 [==============================] - 9s 1ms/step - loss: {training_loss:.4f}"
        
        # Check if the epoch index exists in model_history.history['val_loss']
        if epoch < len(model_historyP.history['val_loss']):
            validation_loss = model_historyP.history['val_loss'][epoch]
            logs += f" - val_loss: {validation_loss:.4f}"

        # Use Markdown for basic formatting
        logs = f"```markdown\n{logs}\n```"
        
        # Append logs to the list
        training_logsP.append(logs)

        # Sleep to control the update frequency (adjust as needed)
        time.sleep(1)  # You can adjust the sleep duration for real-time updates

    # Display the accumulated logs
    logs_areaP.markdown('\n\n'.join(training_logsP), unsafe_allow_html=True)

    test_pred10P = model4P.predict(X_test1P)
    train_pred10P = model4P.predict(X_train1P)


    #Obtaining accuracy of the model

    st.write('Test set evaluation:\n_____________________________________')
    evaluation_text10P = print_evaluate(y_testP, test_pred10P)
    st.text(evaluation_text10P)

    st.write('Train set evaluation:\n_____________________________________')
    evaluation_text10P = print_evaluate(y_trainP, train_pred10P)
    st.text(evaluation_text10P)

    results_df_11P = pd.DataFrame(data=[["Artficial Neural Network", *evaluate(y_testP, test_pred10P), 0]], 
                                columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square', 'Cross Validation'])

    results_dfP = results_dfP._append(results_df_11P, ignore_index=True)
    st.dataframe(results_dfP)

    # Create a Streamlit chart to display loss curves
    chart = st.empty()
    # Lists to store loss and validation loss values
    loss_valuesP = model_historyP.history['loss']
    val_loss_valuesP = model_historyP.history['val_loss']

    # Plot the loss curves
    plt.figure(figsize=(8, 6))
    plt.plot(loss_valuesP, label='Training Loss')
    plt.plot(val_loss_valuesP, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss and Validation Loss Curves')
    plt.legend()
    plt.grid()

    # Display the chart in Streamlit
    chart.pyplot(plt)

with ML_Sales:
    st.write("##### Implementing Linear Regression on Sales:")
    #Implementing Linear Regression
    from sklearn.linear_model import LinearRegression
    lin_reg = LinearRegression()
    lin_reg.fit(X_trainS,y_trainS)
    LinearRegression()
    Inter = lin_reg.intercept_
    st.text(f'Intercept: {Inter}')
    #Predicting data
    pred = lin_reg.predict(X_testS)

    #Obtaining accuracy of the model

    test_pred = lin_reg.predict(X_testS)
    train_pred = lin_reg.predict(X_trainS)

    st.write('Test set evaluation:\n_____________________________________')
    evaluation_text = print_evaluate(y_testS, test_pred)
    st.text(evaluation_text)

    st.write('Train set evaluation:\n_____________________________________')
    evaluation_text = print_evaluate(y_trainS, train_pred)
    st.text(evaluation_text)

    results_df = pd.DataFrame(data=[["Linear Regression", *evaluate(y_testS, test_pred) , cross_valS(LinearRegression())]], columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square', "Cross Validation"])

    st.dataframe(results_df)


    st.write("##### Implementing Robust Regression on Sales:")

    from sklearn.linear_model import RANSACRegressor, LinearRegression
    model = RANSACRegressor(LinearRegression(), max_trials=100)
    model.fit(X_trainS, y_trainS)

    #Prediction of data
    test_pred1 = model.predict(X_testS)
    train_pred1 = model.predict(X_trainS)

    #Obtaining accuracy of the model
    st.write('Test set evaluation:\n_____________________________________')
    evaluation_text1 = print_evaluate(y_testS, test_pred1)
    st.text(evaluation_text1)

    st.write('Train set evaluation:\n_____________________________________')
    evaluation_text1 = print_evaluate(y_trainS, train_pred1)
    st.text(evaluation_text1)

    results_df_1 = pd.DataFrame(data=[["Robust Regression", *evaluate(y_testP, test_pred1) , cross_valS(RANSACRegressor())]], 
                                columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square', "Cross Validation"])


    results_df = results_df._append(results_df_1, ignore_index=True)
    st.dataframe(results_df)


    st.write("##### Implementing Ridge Regression on Sales:")

    from sklearn.linear_model import Ridge
    model1 = Ridge(alpha=100, solver='cholesky', tol=0.0001, random_state=42)
    model1.fit(X_trainS, y_trainS)

    #Prediction of data
    test_pred2 = model1.predict(X_testS)
    train_pred2 = model1.predict(X_trainS)

    #Obtaining accuracy of the model
    st.write('Test set evaluation:\n_____________________________________')
    evaluation_text2 = print_evaluate(y_testS, test_pred2)
    st.text(evaluation_text2)

    st.write('Train set evaluation:\n_____________________________________')
    evaluation_text2 = print_evaluate(y_trainS, train_pred2)
    st.text(evaluation_text2)

    results_df_3 = pd.DataFrame(data=[["Ridge Regression", *evaluate(y_testS, test_pred2) , cross_valS(Ridge())]], 
                                columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square', "Cross Validation"])

    results_df = results_df._append(results_df_3, ignore_index=True)
    st.dataframe(results_df)


    st.write("##### Implementing LASSO Regression on Sales:")

    from sklearn.linear_model import Lasso
    model2 = Lasso(alpha=0.1, 
                precompute=True,
                positive=True, 
                selection='random',
                random_state=42)
    model2.fit(X_trainS, y_trainS)

    #Prediction of data
    test_pred3 = model2.predict(X_testS)
    train_pred3 = model2.predict(X_trainS)

    #Obtaining accuracy of the model

    st.write('Test set evaluation:\n_____________________________________')
    evaluation_text3 = print_evaluate(y_testS, test_pred3)
    st.text(evaluation_text3)

    st.write('Train set evaluation:\n_____________________________________')
    evaluation_text3 = print_evaluate(y_trainS, train_pred3)
    st.text(evaluation_text3)

    results_df_4 = pd.DataFrame(data=[["Lasso Regression", *evaluate(y_testS, test_pred3) , cross_valS(Lasso())]], 
                                columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square', "Cross Validation"])

    results_df = results_df._append(results_df_4, ignore_index=True)

    st.dataframe(results_df)

    st.write("##### Implementing Elastic Net Regression on Sales:")

    from sklearn.linear_model import ElasticNet
    model3 = ElasticNet(alpha=0.1, l1_ratio=0.9, selection='random', random_state=42)
    model3.fit(X_trainS, y_trainS)

    #Prediction of data
    test_pred4 = model3.predict(X_testS)
    train_pred4 = model3.predict(X_trainS)

    #Obtaining accuracy of the model

    st.write('Test set evaluation:\n_____________________________________')
    evaluation_text4 = print_evaluate(y_testS, test_pred4)
    st.text(evaluation_text4)

    st.write('Train set evaluation:\n_____________________________________')
    evaluation_text4 = print_evaluate(y_trainS, train_pred4)
    st.text(evaluation_text4)

    results_df_5 = pd.DataFrame(data=[["Elastic Net Regression", *evaluate(y_testS, test_pred4) , cross_valS(ElasticNet())]], 
                                columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square', "Cross Validation"])

    results_df = results_df._append(results_df_5, ignore_index=True)
    st.dataframe(results_df)

    st.write("##### Implementing Stochastic Gradient Descent on Sales:")
    from sklearn.linear_model import SGDRegressor
    sgd_reg = SGDRegressor(n_iter_no_change=250, penalty=None, eta0=0.0001, max_iter=100000)
    sgd_reg.fit(X_trainS, y_trainS)

    #Prediction of data
    test_pred5 = sgd_reg.predict(X_testS)
    train_pred5 = sgd_reg.predict(X_trainS)

    #Obtaining accuracy of the model

    st.write('Test set evaluation:\n_____________________________________')
    evaluation_text5 = print_evaluate(y_testS, test_pred5)
    st.text(evaluation_text5)

    st.write('Train set evaluation:\n_____________________________________')
    evaluation_text4 = print_evaluate(y_trainS, train_pred5)
    st.text(evaluation_text5)

    results_df_6 = pd.DataFrame(data=[["Stochastic Gradient Descent", *evaluate(y_testS, test_pred5), cross_valS(SGDRegressor())]], 
                                columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square', 'Cross Validation'])

    results_df = results_df._append(results_df_6, ignore_index=True)
    st.dataframe(results_df)

    st.write("##### Implementing Random Forest Regressor on Sales:")
    from sklearn.ensemble import RandomForestRegressor
    rf_reg = RandomForestRegressor(n_estimators=1000)
    rf_reg.fit(X_trainS, y_trainS)

    #Prediction of data

    test_pred6 = rf_reg.predict(X_testS)
    train_pred6 = rf_reg.predict(X_trainS)

    #Obtaining accuracy of the model

    st.write('Test set evaluation:\n_____________________________________')
    evaluation_text6 = print_evaluate(y_testS, test_pred6)
    st.text(evaluation_text6)

    st.write('Train set evaluation:\n_____________________________________')
    evaluation_text6 = print_evaluate(y_trainS, train_pred6)
    st.text(evaluation_text6)

    results_df_7 = pd.DataFrame(data=[["Random Forest Regressor", *evaluate(y_testS, test_pred6), cross_valS(RandomForestRegressor())]], 
                                columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square', 'Cross Validation'])

    results_df = results_df._append(results_df_7, ignore_index=True)
    st.dataframe(results_df)

    st.write("##### Implementing Decision Tree Regressor on Sales:")
    from sklearn.tree import DecisionTreeRegressor

    # Initialize the model
    tree_regressor = DecisionTreeRegressor(random_state=42)

    # Fit the model to training data
    tree_regressor.fit(X_trainS, y_trainS)

    # Make predictions
    test_pred7 = tree_regressor.predict(X_testS)
    train_pred7 = tree_regressor.predict(X_trainS)

    #Obtaining accuracy of the model

    st.write('Test set evaluation:\n_____________________________________')
    evaluation_text7 = print_evaluate(y_testS, test_pred7)
    st.text(evaluation_text7)

    st.write('Train set evaluation:\n_____________________________________')
    evaluation_text7 = print_evaluate(y_trainS, train_pred7)
    st.text(evaluation_text7)

    results_df_8 = pd.DataFrame(data=[["Decision Tree Regressor", *evaluate(y_testS, test_pred7), cross_valS(DecisionTreeRegressor())]], 
                                columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square', 'Cross Validation'])

    results_df = results_df._append(results_df_8, ignore_index=True)
    st.dataframe(results_df)

    st.write("##### Implementing Support Vector Machine on Sales:")
    from sklearn.svm import SVR
    svm_reg = SVR(kernel='linear', C=1, epsilon=0.1)
    svm_reg.fit(X_trainS, y_trainS)

    #Prediction of data
    test_pred8 = svm_reg.predict(X_testS)
    train_pred8 = svm_reg.predict(X_trainS)

    #Obtaining accuracy of the model

    st.write('Test set evaluation:\n_____________________________________')
    evaluation_text8 = print_evaluate(y_testS, test_pred8)
    st.text(evaluation_text8)

    st.write('Train set evaluation:\n_____________________________________')
    evaluation_text8 = print_evaluate(y_trainS, train_pred8)
    st.text(evaluation_text8)

    results_df_9 = pd.DataFrame(data=[["SVM Regressor", *evaluate(y_testS, test_pred8), cross_valS(SVR())]], 
                                columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square', 'Cross Validation'])

    results_df = results_df._append(results_df_9, ignore_index=True)
    st.dataframe(results_df)

    st.write("##### Implementing Polynomial Regression on Sales:")

    from sklearn.preprocessing import PolynomialFeatures
    poly_reg = PolynomialFeatures(degree=2)
    X_train_2_d = poly_reg.fit_transform(X_trainS)
    X_test_2_d = poly_reg.transform(X_testS)

    # Creating a new Linear Regression model instance for polynomial features
    lin_reg_poly = LinearRegression()

    # Fitting this new model with the transformed training data
    lin_reg_poly.fit(X_train_2_d, y_trainS)

    # Prediction of data with the new model instance
    test_pred9 = lin_reg_poly.predict(X_test_2_d)
    train_pred9 = lin_reg_poly.predict(X_train_2_d)

    #Obtaining accuracy of the model

    st.write('Test set evaluation:\n_____________________________________')
    evaluation_text9 = print_evaluate(y_testS, test_pred9)
    st.text(evaluation_text9)

    st.write('Train set evaluation:\n_____________________________________')
    evaluation_text9 = print_evaluate(y_trainS, train_pred9)
    st.text(evaluation_text9)

    results_df_10 = pd.DataFrame(data=[["Polynomail Regression", *evaluate(y_testS, test_pred9), 0]], 
                                columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square', 'Cross Validation'])

    results_df = results_df._append(results_df_10, ignore_index=True)
    st.dataframe(results_df)

    st.write("##### Implementing Artficial Neural Network on Sales:")

    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    import time

    X_train1 = np.array(X_trainS)
    X_test1 = np.array(X_testS)
    y_train1 = np.array(y_trainS)
    y_test1 = np.array(y_testS)

    model4 = Sequential()

    model4.add(Dense(X_trainS.shape[1], activation='relu'))
    model4.add(Dense(32, activation='relu'))
    model4.add(Dense(64, activation='relu'))
    model4.add(Dense(128, activation='relu'))
    model4.add(Dense(512, activation='relu'))
    model4.add(Dropout(0.1))
    model4.add(Dense(1))
    model4.compile(optimizer=Adam(0.00001), loss='mse')

    # Create a Streamlit text area to display training logs
    logs_area = st.empty()

    # Train your model and capture training logs
    model_history = model4.fit(X_train1, y_train1,
                            validation_data=(X_test1, y_test1),
                            batch_size=1,
                            epochs=100,
                            verbose=2)  # Set verbose to 2 to capture training logs

    # Display training completion message
    st.success('Training completed!')

    # Accumulate training logs in a list
    training_logs = []

    # Display training logs in real-time with Markdown and HTML
    for epoch in range(100):
        logs = f"Epoch {epoch + 1}/100\n"
        
        # Check if the epoch index exists in model_history.history['loss']
        if epoch < len(model_history.history['loss']):
            training_loss = model_history.history['loss'][epoch]
            logs += f"6995/6995 [==============================] - 9s 1ms/step - loss: {training_loss:.4f}"
        
        # Check if the epoch index exists in model_history.history['val_loss']
        if epoch < len(model_history.history['val_loss']):
            validation_loss = model_history.history['val_loss'][epoch]
            logs += f" - val_loss: {validation_loss:.4f}"

        # Use Markdown for basic formatting
        logs = f"```markdown\n{logs}\n```"
        
        # Append logs to the list
        training_logs.append(logs)

        # Sleep to control the update frequency (adjust as needed)
        time.sleep(1)  # You can adjust the sleep duration for real-time updates

    # Display the accumulated logs
    logs_area.markdown('\n\n'.join(training_logs), unsafe_allow_html=True)

    test_pred10 = model4.predict(X_test1)
    train_pred10 = model4.predict(X_train1)


    #Obtaining accuracy of the model

    st.write('Test set evaluation:\n_____________________________________')
    evaluation_text10 = print_evaluate(y_testS, test_pred10)
    st.text(evaluation_text10)

    st.write('Train set evaluation:\n_____________________________________')
    evaluation_text10 = print_evaluate(y_trainS, train_pred10)
    st.text(evaluation_text10)

    results_df_11 = pd.DataFrame(data=[["Artficial Neural Network", *evaluate(y_testS, test_pred10), 0]], 
                                columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square', 'Cross Validation'])

    results_df = results_df._append(results_df_11, ignore_index=True)
    st.dataframe(results_df)

    # Create a Streamlit chart to display loss curves
    chart = st.empty()
    # Lists to store loss and validation loss values
    loss_values = model_history.history['loss']
    val_loss_values = model_history.history['val_loss']

    # Plot the loss curves
    plt.figure(figsize=(8, 6))
    plt.plot(loss_values, label='Training Loss')
    plt.plot(val_loss_values, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss and Validation Loss Curves')
    plt.legend()
    plt.grid()

    # Display the chart in Streamlit
    chart.pyplot(plt)




st.subheader(":point_right: Initiation of Machine Learning Classification Analysis")

X_region = df_clean.drop(["Region"],axis=1)
y_region = df_clean[["Region"]]
X_train, X_test, y_train, y_test = train_test_split(X_region, y_region, test_size=0.2, random_state=42)

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, NuSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

color_maps = [
    'aggrnyl', 'agsunset', 'algae', 'amp', 'armyrose', 'balance', 
    'twilight', 'viridis', 'ylgn', 'ylgnbu', 'ylorbr', 'ylorrd'
]

# Models to be used
models = {
    "KNeighborsClassifier": KNeighborsClassifier(),
    "SVC": SVC(random_state=42),
    "NuSVC": NuSVC(random_state=42),
    "AdaBoostClassifier": AdaBoostClassifier(random_state=42),
    "GradientBoostingClassifier": GradientBoostingClassifier(random_state=42),
    "GaussianNB": GaussianNB(),
    "LinearDiscriminantAnalysis": LinearDiscriminantAnalysis(),
    "QuadraticDiscriminantAnalysis": QuadraticDiscriminantAnalysis(),
    "RandomForestClassifier": RandomForestClassifier(random_state=42),
    "LogisticRegression": LogisticRegression(random_state=42),
    "DecisionTreeClassifier": DecisionTreeClassifier(random_state=42),
    "XGBoost": xgb.XGBClassifier(random_state=42)
}

# Fit and predict with each model
results = {}
for idx, (model_name, model) in enumerate(models.items()):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=['Central', 'East', 'South', 'North'], output_dict=True)
    results[model_name] = report

    # Show classification report
    st.subheader(f"{model_name} Classification Report")
    st.text(classification_report(y_test, y_pred, target_names=['Central', 'East', 'South', 'North']))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot confusion matrix with Plotly Express
    fig = px.imshow(cm,
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=['Central', 'East', 'South', 'North'],
                    y=['Central', 'East', 'South', 'North'],
                    color_continuous_scale=color_maps[idx % len(color_maps)],
                    text_auto=True)
    
    fig.update_layout(title_text=f'{model_name} Confusion Matrix', title_x=0.5)
    
    # Display the plotly figure in Streamlit
    st.plotly_chart(fig)


# Show model accuracy comparison
st.subheader("Model Accuracy Comparison")
accuracy_data = {"Model": [], "Accuracy (%)": []}
for model_name, report in results.items():
    accuracy = report['accuracy'] * 100
    accuracy_data["Model"].append(model_name)
    accuracy_data["Accuracy (%)"].append(accuracy)

# Create a DataFrame for accuracy comparison and display it
accuracy_df = pd.DataFrame(accuracy_data)
st.table(accuracy_df)

