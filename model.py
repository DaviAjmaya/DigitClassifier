import tensorflow as tf

# Initialize model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

# Load data from MNIST
mnist = tf.keras.datasets.mnist
tf.logging.set_verbosity(tf.logging.ERROR)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Downscale pixel values to a range of 0 to 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# Compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train and evaluate the model
model.fit(x_train, y_train, epochs=15, verbose=2)
model.evaluate(x_test, y_test, verbose=2)

model.save('model.h5')