# %%
# Import necessary libraries
import streamlit as st
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
# Data Visualization Libraries
import matplotlib.pyplot as plt
import seaborn as sns
#
from sklearn.preprocessing import LabelEncoder

# Machine Learning Libraries
import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

st.title("E-commerce Customer Satisfaction Analysis and Prediction")
st.write("This application analyzes customer satisfaction data from an e-commerce platform and predicts CSAT scores based on various features.")    

# Load the dataset
df = pd.read_csv('ecommerce.csv')


# %%
df.head()

# %%
df.isnull().mean() * 100

# %%
# Data Cleaning: Handle missing values
# For simplicity we will drop some columns with high missing values and fill others with mean 
# Drop columns with more than 70% missing values

columns_to_drop = df.columns[df.isnull().mean() > 0.8]
df.drop(columns=columns_to_drop, inplace=True)
# Fill remaining missing values with mean for numerical columns
df['Item_price'] = df['Item_price'].fillna(df['Item_price'].mean())

# Fill categorical columns with mode
categorical_cols = df.select_dtypes(include=['object','string']).columns
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])


# Verify that there are no more missing values
df.isnull().sum()




# %%
st.subheader("Data Overview")
st.write("Here's a preview of the dataset:")
st.dataframe(df.head())

# Data Visualization
#UNIVARIANT CHART 1

# Distribution of CSAT Score
plt.hist(df['CSAT Score'], color= 'red', alpha=0.7,bins=10)
plt.title('Distribution of CSAT Score')
plt.xlabel('CSAT Score')
plt.ylabel('Frequency')
st.pyplot(plt)
plt.close()
#WHY: To understand the distribution of customer satisfaction scores and identify any skewness or outliers.
#INSIGHT: The histogram shows that the majority of customers have a CSAT Score between 3 and 5, indicating generally high satisfaction.
# There are fewer customers with lower scores, suggesting that most customers are satisfied with their purchases.
#BUSINESS DECISION: Focus on maintaining high customer satisfaction by analyzing the factors contributing to high CSAT Scores and addressing any issues that may lead to lower scores.

# UNIVARIANT CHART 2
# Item price distribution
plt.hist(df['Item_price'])
plt.xlim(0,df['Item_price'].max())
plt.xlabel('Item price')
plt.ylabel('Frequency')
plt.title('Price Distribution')
plt.grid()
st.pyplot(plt)
plt.close()
#WHY: To analyze the distribution of item prices and identify any trends or outliers in pricing.
#INSIGHT: The histogram reveals that most items are priced below 20,000, with a few outliers at higher price points.
#  This suggests that the majority of products are in the affordable range, while a small number of premium products exist.
#BUSINESS DECISION: Consider offering a wider range of products at different price points to cater to various customer segments.
# Additionally, analyze the performance of higher-priced items to determine if they contribute significantly to revenue and
# customer satisfaction.


# BIVARIANT CHART 3
# Relationship between CSAT Score and Item Price
plt.bar(df['CSAT Score'], df['Item_price'], color='Brown')
plt.title('CSAT Score vs Item Price')
plt.xlabel('CSAT Score')
plt.ylabel('Item Price')
plt.grid()
st.pyplot(plt)
plt.close()
#WHY: To explore if there is any correlation between the price of items and customer satisfaction.
#INSIGHT: The bar chart indicates that higher-priced items tend to have higher CSAT Scores
#BUSINESS DECISION: Consider offering premium products or improving the quality of higher-priced items to enhance
#  customer satisfaction and potentially increase sales.



#BIVARIANT CHART 4
#Product category vs item price
plt.scatter(df['Product_category'],df['Item_price'],color = "pink")
plt.xticks(rotation = 45 , ha = 'right')
plt.xlabel('Product category')
plt.ylabel('Item Price')
plt.title('Product_Category vs Item price')
plt.grid()
st.pyplot(plt)
plt.close()

#WHY: To analyze how different product categories are priced and identify which categories have higher average prices.
#INSIGHT: The bar chart reveals that certain categories, such as Electronics and mobile,    have higher average item prices compared to others like Clothing and Home & Kitchen.
# BUSINESS DECISION: Focus marketing efforts on higher-priced categories to maximize revenue, and consider offering promotions or discounts on lower-priced categories to boost sales.    

#BIVARIANT CHART 5
# ITEM PRICE VS CATEGORY
data =df
sns.barplot(data ,y = 'CSAT Score',x = 'channel_name')
plt.title('CSAT Score vs Channel name')
plt.xlabel('Channel Name')
plt.ylabel('CSAT Score')
plt.grid()
st.pyplot(plt)
plt.close()
#WHY: To examine the relationship between sales channels and customer satisfaction scores.
#INSIGHT: The bar chart indicates that certain channels, such as Online and Mobile App, have higher average CSAT Scores compared to others like In-store.
#BUSINESS DECISION: Invest in improving the customer experience on channels with lower CSAT Scores, such as In-store, by providing better training for staff or enhancing the shopping environment. Additionally, leverage the strengths of channels with higher CSAT Scores to further boost customer satisfaction and loyalty.


# MULTIVARIANT CHART 6
# AVG PRICE BY CHANNEL NAME
df.groupby('channel_name')['Item_price'].mean().plot(kind = 'bar',color = 'grey')
plt.title('Aveage price VS Channel name')
plt.xlabel('Channel Name')
plt.ylabel('Average Item Price')
plt.grid()
st.pyplot(plt)
plt.close()
#WHY: To analyze how the average item price varies across different sales channels.
#INSIGHT: The bar chart reveals that the average item price is higher for the Online channel compared to In-store and Mobile App channels.
#BUSINESS DECISION: Consider optimizing pricing strategies for the Online channel to capitalize on higher average prices, while also exploring opportunities to increase the average item price for In-store and Mobile App channels through targeted promotions or bundling strategies.

#MULTIVARIANT CHART 7
plt.figure (figsize=(6,4))
sns.heatmap(df.select_dtypes(include= np.number).corr(),annot = True, cmap="viridis")
plt.title("Correlation matrix")
plt.xlabel("Numerical Variables")
plt.ylabel("Numerical Variables")
plt.grid()
plt.tight_layout()
st.pyplot(plt)
plt.close()
#WHY: To examine the correlation between numerical variables in the dataset and identify any strong relationships.

#INSIGHT: The heatmap reveals that there is a moderate positive correlation between Item_price and CSAT Score,
# suggesting that higher-priced items may lead to higher customer satisfaction.

#BUSINESS DECISION: Consider investing in higher-quality products that can be priced higher to enhance customer satisfaction
#and potentially increase revenue. Additionally, analyze other factors that may contribute to customer satisfaction to further 
# improve the overall customer experience.


# Data Preprocessing for Machine Learning
df.select_dtypes(include='object').columns
# Drop unnecessary columns
df.drop(['Order_id','Customer Remarks','Agent_name','Agent Shift'],axis = 1,inplace = True)
df.drop(['order_date_time','Issue_reported at','issue_responded','Survey_response_Date','Tenure Bucket'],axis =1,inplace=True)


# Encode categorical variables
cat_cols = df.select_dtypes(include ='object').columns
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Split features and target variable
x = df.drop('CSAT Score', axis =1)
y = df['CSAT Score']

#  scale the features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x = scaler.fit_transform(x)

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

# Define the neural network model
model = Sequential()

model.add(Dense(128, activation='relu', input_shape=(x_train.shape[1],)))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(32, activation='relu'))

model.add(Dense(6, activation='softmax'))



# Compile the model
optimizer = Adam(learning_rate=0.0005)
model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Set up early stopping
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=6,
    restore_best_weights=True
)

# Train the model with caching to avoid retraining on every run
@st.cache_resource
def train_model():
    st.write("Training the model... Please wait.")
    model.fit(
        x_train,
        y_train,
        epochs=20,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stop]
    )
    return model
model = train_model()
  


# Evaluate the model
st.subheader("Model Performance")
loss, acc = model.evaluate(x_test,y_test)
st.success(f"Model Accuracy: {acc*100:.2f}%")

# Predict CSAT on test data
y_pred = model.predict(x_test)
y_pred_class = np.argmax(y_pred, axis=1)+1

result = pd.DataFrame({
    "Actual CSAT": y_test.values,
    "Predicted CSAT": y_pred_class
})

st.write("Actual vs Predicted CSAT")
st.dataframe(result.head(20))

result["Difference"] = result["Actual CSAT"] - result["Predicted CSAT"]

st.write("Prediction Difference")
st.dataframe(result.head(20))

# User Friendly Input for Prediction
# Categorical input
product_category = st.selectbox("Product Category", ["Electronics", "Mobile","Home Appliances","Books & General merchandise","GiftCards", "Furniture", "Accessories"])
sub_category = st.selectbox("Sub Category",(['Life Insurance', 'Product Specific Information',
       'Installation/demo', 'Reverse Pickup Enquiry', 'Not Needed',
       'Fraudulent User']))
item_price = st.number_input("Item Price", value=12000)

if st.button("Predict CSAT"):

    input_data = pd.DataFrame({
        "Product_category":[product_category],
        "Sub_category":[sub_category],
        "Item_price":[item_price]
    })

    # Encode categorical data
    for col in input_data.select_dtypes(include='object').columns:
        le = encoders[col]
        input_data[col] = le.transform(input_data[col])
    # Scale
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_scaled)
    predicted_csat = np.argmax(prediction)+1

    st.success(f"Predicted CSAT Score: {predicted_csat}")

#WHY: To provide a user-friendly interface for predicting CSAT scores based on input features.
#INSIGHT: Users can input product category, sub-category, and item price to get a predicted CSAT score, which can help in understanding customer satisfaction for specific products.
#BUSINESS DECISION: Use the prediction tool to identify products that may have lower predicted CSAT scores and take proactive measures to improve customer satisfaction, such as enhancing product quality or providing better customer support.




