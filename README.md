




🔬 Project Title: Hematovision

Subtitle: Advanced Blood Cell Classification Using Transfer Learning
Language: Python
Libraries: TensorFlow/Keras, OpenCV, NumPy, Matplotlib, scikit-learn, Pandas



 Project Structure

hematovision/
│
├── data/
│   └── blood_cells/         # Dataset folders (train/test/val with images)
├── models/
│   └── hematovision_model.h5
├── notebooks/
│   └── EDA_and_Training.ipynb
├── utils/
│   └── image_preprocessing.py
├── main.py
├── train.py
├── evaluate.py
├── requirements.txt
└── README.md





You can use the Blood Cell Images dataset from Kaggle which includes:

Neutrophil

Eosinophil

Monocyte

Lymphocyte





tensorflow>=2.10
numpy
matplotlib
pandas
opencv-python
scikit-learn

📚 train.py

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
import os

def train_model():
    img_size = 224
    batch_size = 32

    base_path = "data/blood_cells"
    train_path = os.path.join(base_path, "train")
    val_path = os.path.join(base_path, "val")

    train_gen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=15,
                                   zoom_range=0.1,
                                   horizontal_flip=True)

    val_gen = ImageDataGenerator(rescale=1./255)

    train_data = train_gen.flow_from_directory(train_path, 
                                               target_size=(img_size, img_size), 
                                               batch_size=batch_size,
                                               class_mode='categorical')

    val_data = val_gen.flow_from_directory(val_path,
                                           target_size=(img_size, img_size),
                                           batch_size=batch_size,
                                           class_mode='categorical')

    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(128, activation='relu')(x)
    output = Dense(4, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=output)

    for layer in base_model.layers:
        layer.trainable = False  # Freeze base model

    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_data, epochs=10, validation_data=val_data)
    model.save("models/hematovision_model.h5")

    return model


---

📈 evaluate.py

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def evaluate_model(model=None):
    if not model:
        model = load_model("models/hematovision_model.h5")

    img_size = 224
    test_path = "data/blood_cells/test"
    test_gen = ImageDataGenerator(rescale=1./255)

    test_data = test_gen.flow_from_directory(test_path,
                                             target_size=(img_size, img_size),
                                             batch_size=32,
                                             class_mode='categorical',
                                             shuffle=False)

    preds = model.predict(test_data)
    y_pred = np.argmax(preds, axis=1)
    y_true = test_data.classes

    print("Classification Report:\n", classification_report(y_true, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred


📄 README.md (Excerpt)

# Hematovision: Advanced Blood Cell Classification Using Transfer Learning

This project uses a pre-trained CNN (EfficientNetB0) to classify 4 types of blood cells using transfer learning. It aims to assist hematologists and medical researchers in fast and accurate detection.

## Classes
- Neutrophil
- Eosinophil
- Monocyte
- Lymphocyte

## Tech Stack
- Python
- TensorFlow / Keras
- OpenCV
- scikit-
- 
