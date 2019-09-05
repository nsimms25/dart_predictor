"""
Model darts thrown by different competitors using a neural network.
darts.csv:
    features - xCoord and yCoord
    label - competitor (Kate, Steve, Michael, Susan)
"""

# Import
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.layers import Dense
from keras.models import Sequential

# Import to_categorical from keras utils module
from keras.utils import to_categorical

# Read in CSV file
darts = pd.read_csv('darts.csv')

# Instantiate a sequential model
model = Sequential()

# Add 3 dense layers of 128, 64 and 32 neurons each
model.add(Dense(128, input_shape=(2,), activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))

# Add a dense layer with as many neurons as competitors
model.add(Dense(4, activation='softmax'))

# Compile your model using categorical_crossentropy loss
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Transform into a categorical variable
darts.competitor = pd.Categorical(darts.competitor)

# Assign a number to each category (label encoding)
darts.competitor = darts.competitor.cat.codes

# Use to_categorical on your labels.
# Use one-hot encoding for the competitors labels instead of strings as the identifiers.
coordinates = darts.drop(['competitor'], axis=1)
competitors = to_categorical(darts.competitor)

'''
# double check the features and labels
# Now print the to_categorical() result
print('One-hot encoded competitors: \n', competitors)
print(coordinates)
'''

coord_train, coord_test, competitors_train, competitors_test = train_test_split(coordinates,
                                                                                competitors,
                                                                                random_state=65)

# Train your model on the training data for 200 epochs
model.fit(coord_train, competitors_train, epochs=200)

# Evaluate your model accuracy on the test data
accuracy = model.evaluate(coord_test, competitors_test)[1]

# Print accuracy
print('Accuracy:', accuracy)

'''
# Output (Accuracy will change based on the pseudo-random splitting of data).
 32/200 [===>..........................] - ETA: 0s
200/200 [==============================] - 0s 150us/step
Accuracy: 0.8
'''
