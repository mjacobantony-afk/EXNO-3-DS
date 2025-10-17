## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, PowerTransformer
from scipy.stats import boxcox

df = pd.read_csv("D:\\Data Science\\CSV files\\Encoding Data.csv")

df.dropna(inplace=True) 

ord_2_mapping = {'Cold': 0, 'Warm': 1, 'Hot': 2}
df['ord_2_encoded'] = df['ord_2'].map(ord_2_mapping)

le = LabelEncoder()
df['bin_1_encoded'] = le.fit_transform(df['bin_1'])
df['bin_2_encoded'] = le.fit_transform(df['bin_2'])

df = pd.get_dummies(df, columns=['nom_0'], prefix='nom_0')

df['ord_2_encoded'] = df['ord_2_encoded'] + 1

df['ord_2_log'] = np.log(df['ord_2_encoded'])

df['ord_2_reciprocal'] = 1 / df['ord_2_encoded']

df['ord_2_sqrt'] = np.sqrt(df['ord_2_encoded'])

df['ord_2_boxcox'], _ = boxcox(df['ord_2_encoded'])

pt = PowerTransformer(method='yeo-johnson')
df['ord_2_yeojohnson'] = pt.fit_transform(df[['ord_2_encoded']])

df.to_csv("Fully_Transformed_Encoding_Data.csv", index=False)
print("All transformations applied and saved to 'Fully_Transformed_Encoding_Data.csv'")
```
OUTPUT
We have saved the changes to a dataset named Fully_Transformed_Encoded_Data.csv. 
# RESULT:
We have performed the above mentioned data encoding and transformation processes on the dataset and obtained new dataset. 

       
