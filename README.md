# Tea Leaf Disease Detection

A deep learning-based tea leaf disease detection system that classifies tea leaf images into different disease categories using a Convolutional Neural Network (CNN).

## Detected Classes

The model currently detects 7 classes:

* Anthracnose
* Bird Eye Spot
* Brown Blight
* Gray Light
* Healthy
* Red Leaf Spot
* White Spot

## Technologies Used

* Python
* PyTorch
* OpenCV
* NumPy
* Scikit-learn
* Albumentations
* Streamlit

## Project Structure

```text
Tea_Leaf_Disease_Model-main/
│
├── DataSetFolder/
│   ├── Anthracnose/
│   ├── bird eye spot/
│   ├── brown blight/
│   ├── gray light/
│   ├── healthy/
│   ├── red leaf spot/
│   └── white spot/
│
├── models/
│   ├── cnn_model.py
│   ├── denseNet121.py
│   ├── EfficientNetV2B3.py
│   ├── Hybrid_CNN_Transformer.py
│   ├── MobileNetV3.py
│   ├── model_factory.py
│   ├── ResNet50.py
│   ├── VGG19.py
│   └── ViTB16.py
│
├── utils/
│   ├── data_loader.py
│   └── train_evaluation.py
│
├── run.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Dataset

The dataset contains 772 tea leaf images distributed across 7 classes.

The dataset is divided into:

* 80% Training
* 10% Validation
* 10% Testing

Image augmentation techniques such as rotation, horizontal flipping, brightness/contrast adjustment, CLAHE, and sharpening are used during training.

## Installation

Clone the repository and create a virtual environment:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Tea_Leaf_Disease_Model-main

python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt
```

## Train the Model

To train the Custom CNN model:

```bash
python run.py --data-dir DataSetFolder
```

You can also specify the model and other training parameters.

## Run the Streamlit Application

After training, run:

```bash
python -m streamlit run app.py
```

If the application requires a specific trained model/checkpoint, make sure the checkpoint is available in the expected location.

## Model Performance

The model is evaluated using:

* Accuracy
* Loss
* AUC
* Validation performance
* Test performance

Training results and model checkpoints are saved in the configured result directory.

## Future Improvements

* Improve classification accuracy with a larger dataset
* Add more tea leaf disease classes
* Deploy the application online
* Improve model inference speed
* Add recommendations for disease management

