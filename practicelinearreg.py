import numpy as np
import pandas as pd
from sklearn import model_selection, preprocessing, linear_model
from palmerpenguins import load_penguins

penguins = load_penguins()
#this is a pandas dataframe
penguins = penguins.dropna()

label_encoder = preprocessing.LabelEncoder()

species = label_encoder.fit_transform(list(penguins['species']))
island = label_encoder.fit_transform(list(penguins['island']))
sex = label_encoder.fit_transform(list(penguins['sex']))

Y_category = list(species)
X_category = list(zip(island, np.array(penguins['bill_length_mm']), np.array(penguins['bill_depth_mm']), np.array(penguins['flipper_length_mm']), np.array(penguins['body_mass_g']), sex, np.array(penguins['year'])))


X_train, X_test, y_train, y_test = model_selection.train_test_split(X_category, Y_category)
linear_regression_model = linear_model.LinearRegression()
linear_regression_model.fit(X_train, y_train)

acc = linear_regression_model.score(X_test, y_test)
print(acc)
predictions = linear_regression_model.predict(X_test)
for i in range(len(predictions)):
    print(X_test[i], predictions[i], y_test[i])



