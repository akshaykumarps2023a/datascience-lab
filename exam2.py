import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('slr.csv')
x = dataset.iloc[:, :1].values
y = dataset.iloc[:, 2].values
print(x)
print(y)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=.3)

from sklearn.linear_model import LinearRegression

regression = LinearRegression()
regression.fit(x_train, y_train)
regression.predict(x_test)

plt.scatter(x_train, y_train, c='red')
plt.plot(x_train, regression.predict(x_train), c='blue')
plt.show()

plt.scatter(x_test, y_test, c='red')
plt.plot(x_train, regression.predict(x_train), c='blue')
plt.title('SALARY')
plt.xlabel('Advt')
plt.ylabel('Sales')
plt.show()
