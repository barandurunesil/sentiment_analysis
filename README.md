# Sentiment Analysis with ML.NET

- A simple sentiment analysis project built with ML.NET using SDCA logistic regression.
- The model predicts whether a sentence expresses a **positive** or **negative** sentiment.

## ML Task

- Binary classification is the task of classifying items into one of two classes.
- In this project, sentences are classified as:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**Positive**
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**Negative**

- The training algorithm used is **SdcaLogisticRegression**, which is a variation of logistic regression based on the Stochastic Dual Coordinate Ascent (SDCA) method.

## Dataset

- This project uses a **modified version** of the Amazon Polarity dataset.
- Original source: https://huggingface.co/datasets/mteb/amazon_polarity
- The dataset was adapted to a **TSV format** and simplified to include only the **numeric label** and **text** fields.

## Dataset and Pretrained Model

- The dataset and pretrained model are **not included in this repository** due to file size limits. You need to download them to run the project.
- Download them from: https://drive.google.com/drive/folders/10CphYB0wHedczE85qHGOoJifdwM-ORY0?usp=sharing
- After downloading, extract the files and place the **Data** folder inside the project directory (where `Program.cs` is located).

## Results

- **Test accuracy:** ~92.5%
