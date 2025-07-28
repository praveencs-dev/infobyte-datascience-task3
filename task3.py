import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
df=pd.read_csv("D:\\datascience\\task3\\car.csv")
df=pd.get_dummies(df,drop_first=True)
x=df.drop('Selling_Price',axis=1)
y=df['Selling_Price']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print("model accuracy is:",r2_score(y_pred,y_test))
car=x.iloc[4:5]
price=model.predict(car)
print("the car price is:",price)