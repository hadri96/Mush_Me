# Mush_Me Project Overview
The "Mush_Me" project is a deep learning tool designed to help mushroom enthusiasts avoid poisonous varieties through image recognition. Built with TensorFlow and Python, the project utilizes Google Cloud Platform for training and storing the model.

## Technical Stack
* **Programming Language**: Python
* **Machine Learning Framework**: TensorFlow (Keras)
* **Cloud Services**: Google Cloud Storage
* **Model Architecture**
    * **Base Model**: ResNet50 used for feature extraction with transfer learning.
    * **Classifier**: Custom dense layers for specific class predictions.
* Workflow
    * **Data Handling**: Automated data fetching and preprocessing from Google Cloud Storage.
    * **Training**: Model trained with augmented image data to enhance accuracy and robustness.
    * **Evaluation**: Performance assessed using a separate test dataset.
    * **Deployment**: Model saved and uploaded to Google Cloud for accessibility.
## Setup Instructions
* Clone the repository and install required packages from requirements.txt.
* Set up Google Cloud Storage with the necessary datasets.
* Train the model by running the main script.
*Evaluate and deploy the model as needed.

## Contributing
Contributions to the "Mush_Me" project are welcome! Please read CONTRIBUTING.md for guidelines on how to make a contribution.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
