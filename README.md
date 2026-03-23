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
       # INCLUDE YOUR CODING AND OUTPUT SCREENSHOTS HERE
   import numpy as np
from scipy import stats
df=pd.read_csv("data.csv")
df
<img width="579" height="429" alt="image" src="https://github.com/user-attachments/assets/16bc7cbf-5e02-45f3-968a-08ed3ba01c59" />
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
climate=['Cold','Warm','Hot','Very Hot']
ele=OrdinalEncoder(categories=[climate])
ele.fit_transform(df[["Ord_1"]])
<img width="155" height="232" alt="image" src="https://github.com/user-attachments/assets/701e1ee1-9c47-4f6d-9fd1-875fc278dc99" />
df['bo2']=ele.fit_transform(df[['Ord_1']])
df
<img width="604" height="433" alt="image" src="https://github.com/user-attachments/assets/557a3688-d18f-4b2e-a09c-152344f5ba2d" />
le=LabelEncoder()
df2=df.copy()
df2['Ord_2']=le.fit_transform(df2['Ord_2'])
df2
<img width="566" height="431" alt="image" src="https://github.com/user-attachments/assets/709f2e74-d523-47ce-a898-88086c816d6a" />
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder()
df3=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df[['City']]))
df2=pd.concat([enc,df3],axis=1)
df2
<img width="1039" height="427" alt="image" src="https://github.com/user-attachments/assets/51d3b4c0-2bbd-4e2b-bef1-efe511fec51a" />
pd.get_dummies(df,columns=['City'])
<img width="1027" height="439" alt="image" src="https://github.com/user-attachments/assets/475e3259-f647-41b4-8628-e02d1c3192f8" />
pip install --upgrade category_encoders
<img width="1543" height="353" alt="image" src="https://github.com/user-attachments/assets/33c1476e-b949-4279-b992-1f6719aa746f" />
from category_encoders import BinaryEncoder
import pandas as pd
df=pd.read_csv("C:\Users\Downloads\data.csv")
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
df1=pd.concat([df,nd],axis=1)
df1=df.copy()
df1
<img width="656" height="448" alt="image" src="https://github.com/user-attachments/assets/92bc226e-60e2-4d7f-8709-c69b42389128" />
from category_encoders import TargetEncoder
te=TargetEncoder()
cc=df.copy()
new=te.fit_transform(X=cc["City"],y=cc["Target"])
cc=pd.concat([cc,new],axis=1)
cc
<img width="750" height="465" alt="image" src="https://github.com/user-attachments/assets/8c439817-bd09-40f9-ae5a-6ede801cb84e" />
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("Data_to_Transform.csv")
df
<img width="1041" height="535" alt="image" src="https://github.com/user-attachments/assets/557d5cbd-c8df-411d-aeb7-bf9b920629b9" />
df.skew()
<img width="392" height="128" alt="image" src="https://github.com/user-attachments/assets/8d2b9d07-c58b-4d5d-a3ed-892fd0b8ada5" />
np.log(df["Highly Positive Skew"])
<img width="650" height="305" alt="image" src="https://github.com/user-attachments/assets/5608181e-a659-4d9c-b43f-75d09b3c03ba" />
np.reciprocal(df["Highly Positive Skew"])
<img width="740" height="302" alt="image" src="https://github.com/user-attachments/assets/222aed81-795f-439d-943a-c90503d89e75" />
np.reciprocal(df["Moderate Positive Skew"])
<img width="667" height="302" alt="image" src="https://github.com/user-attachments/assets/914d801c-3f90-4fee-b1cb-061ff210c09a" />
np.square(df["Highly Positive Skew"])
<img width="658" height="286" alt="image" src="https://github.com/user-attachments/assets/6248fd00-bfee-41d5-bba3-a1f8bec5d6e8" />
df["Highly Positive Skew_boxcox"],parameters=stats.boxcox(df["Highly Positive Skew"])
df
df["Moderate Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df['Moderate Negative Skew'])
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
<img width="1611" height="582" alt="image" src="https://github.com/user-attachments/assets/10d8db8b-01d8-41f8-941c-713d50d55980" />
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats
sm.qqplot(df['Moderate Negative Skew'],line='45')
plt.show()
<img width="1018" height="598" alt="image" src="https://github.com/user-attachments/assets/35c2ba71-8150-44f1-baff-4722b7a8a5ef" />
sm.qqplot(df['Moderate Negative Skew_1'],line='45')
<img width="1058" height="592" alt="image" src="https://github.com/user-attachments/assets/b23edc0a-76ea-4ddf-8460-50b3ab0433f1" />
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df['Highly Negative Skew'],line='45')
plt.show()
<img width="1002" height="596" alt="image" src="https://github.com/user-attachments/assets/8134f08a-3fb2-4fae-8096-f16b14aa18a6" />
sm.qqplot(df['Highly Negative Skew_1'],line='45')
plt.show()
<img width="1066" height="587" alt="image" src="https://github.com/user-attachments/assets/74364c17-05bf-478b-8f3d-5dd8fa7a2105" />
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
<img width="936" height="577" alt="image" src="https://github.com/user-attachments/assets/81227b3a-50f0-44e9-a0bf-d484ce208f2b" />



















# RESULT:
  Thus the given data,Feature Encoding,Transformation process and save the data to a file was performed successfully

       
