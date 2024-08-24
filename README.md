
# Crop Recommendation Using Machine Learning



This project implements a crop recommendation system leveraging machine learning techniques, specifically the Random Forest Algorithm, to offer precise crop suggestions. The model, trained on a dataset of 2201 samples across 22 classes, achieves an exceptional accuracy of 99.24%. The application is deployed using Streamlit, and features robust logging and exception handling to ensure reliable performance.



![App Screenshot](https://github.com/Rishi-Sutar/Crop_Recommendation/blob/main/images/Screenshot%20(2).png)


## Technologies and Skills

- **Machine Learning:** Random Forest Algorithm
- **Dataset:** 2201 samples, 22 classes
- **Deployment:** Streamlit
- **Error Handling:** Comprehensive logging and exception management
## Installation

To set up and run this project locally, please follow these steps:

- **Clone the Repository:**

```bash
    git clone https://github.com/Rishi-Sutar/Crop_Recommendation.git
    cd Crop_Recommendation
```

- **Install Dependencies:**
    
    Install the necessary Python packages by running:

```bash
    pip install -r requirements.txt

```

- **Train the Model:**
    
    Execute the following command to train the model and generate the model.pkl file:

```bash
    python main.py

```

The trained model will be saved in the artifacts directory.

- **Launch the Application:**
    
    To start the Streamlit application and access the graphical user interface, run:

```bash
    python app.py

```

This command will open the Streamlit app in your default web browser, where you can interact with the crop recommendation system.
## How It Works

- Model Training:

    The main.py script trains the Random Forest model using the provided dataset and    outputs the model as model.pkl in the artifacts directory.

- User Interface:

    The app.py script deploys a web-based interface using Streamlit, allowing users to input data and receive crop recommendations based on the trained model.
## Features

- **High Accuracy:** The Random Forest model provides accurate crop recommendations with an accuracy of 99.24%.
- **Robust Deployment:** The application is deployed on Streamlit for a seamless user experience.
- **Logging and Error Handling:** Includes comprehensive logging and exception handling to ensure smooth operation.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

