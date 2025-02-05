# Assignment Details
**Deliveries**: A notebook with your models for part A, and B.  
**Submission deadline**: Thursday February 12 at 11: 59 pm  
**Brief presentation**: 2-3 minutes - February Wednesday 12 at class time.

Use convnet to classify whether images contain either a dog or a cat. This is easy for humans, dogs, and cats. Your computer will find it a bit more difficult. (Textbook Chapte 8)

**Goals**  
- Understanding convolutional neural networks (convnets)  
- Using data augmentation to mitigate overfitting

**Programming environment**: Google-Colab

Use the Kaggle API to programmatically download the Dogs vs. Cats dataset. Follows textbook (Section 8.2.2) or here: https://medium.com/@wl8380/unlocking-kaggle-datasets-a-guide-to-obtaining-and-installing-your-api-key-65ca25a7ac7c  

Dataset: https://www.kaggle.com/c/dogs-vs-cats/data

Follow the chapter to have 2,000 training images, 1,000 validation images, and 2,000 test images. Each split contains the same number of samples froeach class: this is a balanced binary-classification problem, which means classificatioaccuracy will be an appropriate measure of success.

Re-scale your images if need it to use the following architecture. And use an end-to-end workflow presented in Canvas->Modules->week5->end-to-end CNN.ipynb

```python
from tensorflow import keras  
from tensorflow.keras import layers  

model = keras.Sequential([  
	keras.Input(shape=(32, 32, 3)), # Explicit Input Layer  
	layers.Conv2D(32, (3,3), activation="relu", padding="same"),  
	layers.MaxPooling2D((2,2)),  
	layers.Conv2D(64, (3,3), activation="relu", padding="same"),  
	layers.MaxPooling2D((2,2)),  
	layers.Conv2D(128, (3,3), activation="relu", padding="same"),  
	layers.Flatten(),  
	layers.Dense(128, activation="relu"),  
	layers.Dropout(0.5), # Dropout for regularization  
	layers.Dense(10, activation="softmax") # Output layer (10 classes)  
])  

# Compile the Model  
model.compile(  
	optimizer="adam",  
	loss="categorical_crossentropy",  
	metrics=["accuracy"]  
)  

# Display Model Summary  
model.summary()
```

**Train your model using callbacks (Section 8.11)**  

Use 30 epochs  

**Plot accuracy for training and validation, and loss for training and validation.**  

You will find overfitting.  

**Evaluate your model (section 8.13)**  

**Let’s try to correct the overfitting:** Typical techniques include:  
- modify the architecture.
- use more training data or data augmentation
- simplify the model (using regularization)

Let’s use data augmentation and drop out. However, you can increase the size of the training set. (Section 8.16)  

Let’s plot again the metrics for training and validation.
