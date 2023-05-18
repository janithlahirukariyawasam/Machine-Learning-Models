#import libraries
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error

#Import csv file
df=pd.read_csv("/content/Bengaluru_House_Data (2).csv")
data = df.dropna()


#encoding categorical data into numericals
label_encoder = LabelEncoder()
encodedAreaType= label_encoder.fit_transform(data['area_type'])
balcony = data['balcony']
#encodeSize = label_encoder.fit_transform(data['size'])  # three columns

binaryAvailability = data['availability'].apply(lambda x: 1 if x == 'Ready To Move' else 0)

df = pd.DataFrame(data['size'])

# Extract the numerical values from the 'size' column
bedrooms = df['size'].apply(lambda x: int(x.split()[0]))
hall = df['size'].apply(lambda x: 1 if 'hall' in x else 0)
kitchen = df['size'].apply(lambda x: 1 if 'kitchen' in x else 0)

def extract_sqft_value(value):
    numerical_values = re.findall(r'(\d+\.?\d*)', value)
    if len(numerical_values) == 2:
        lower_value = float(numerical_values[0])
        upper_value = float(numerical_values[1])
        return (lower_value + upper_value) / 2
    elif len(numerical_values) == 1:
        return float(numerical_values[0])
    else:
        return np.nan

sqft = data['total_sqft'].apply(extract_sqft_value)

train_X = np.column_stack((encodedAreaType, binaryAvailability,sqft,balcony,bedrooms,hall,kitchen))
train_y = data['price']

selected_X = train_X[4200:]
selected_Y = train_y[4200:]

model = LinearRegression()

model.fit(selected_X, selected_Y)



predicted_price = model.predict([selected_X[7]])
print(predicted_price)


test2_X = train_X[:4200]
test2_y = train_y[:4200]

predictions = model.predict(test2_X)

mae = mean_absolute_error(test2_y, predictions)
mse = mean_squared_error(test2_y, predictions)
rmse = np.sqrt(mse)

print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)