import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

print(f"TensorFlow Version: {tf.__version__}")
if tf.__version__.startswith('1'):
    print("Warning: You are using TensorFlow 1.x. Consider upgrading to TensorFlow 2.x for eager execution.")

tf.compat.v1.enable_eager_execution()

model = tf.keras.models.load_model("model.keras")

train_data = pd.read_csv('train.csv') 

x = train_data.drop(columns=['label']).values 
y = train_data['label'].values  

x = x / 255.0

x = x.reshape(-1, 28, 28, 1)

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

try:
    model.fit(x_train, y_train, epochs=5, validation_data=(x_val, y_val))
except Exception as e:
    print(f"Error during model training: {e}")

model.save("Model_Predict_Digit.keras")

y_pred = model.predict(x_val)
y_pred_classes = np.argmax(y_pred, axis=1)

result_df = pd.DataFrame({
    'ImageId': np.arange(1, len(x_val) + 1), 
    'Label': y_pred_classes,  
    'Actual': y_val  
})

result_df.to_csv('predictions.csv', index=False)
print("Predictions saved to 'predictions.csv'.")

accuracy = accuracy_score(y_val, y_pred_classes)

print(f"Accuracy: {accuracy * 100:.2f}%")

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_val, y_pred_classes)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
