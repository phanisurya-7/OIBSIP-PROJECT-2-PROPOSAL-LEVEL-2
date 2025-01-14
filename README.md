# OIBSIP-PROJECT-2-PROPOSAL-LEVEL-2


```markdown
# Wine Quality Prediction

## 📜 Project Overview
This project focuses on predicting the quality of wine based on its chemical characteristics using machine learning techniques. By analyzing a dataset of wines with features such as acidity, density, and alcohol content, we aim to classify wines as good or bad quality. This application showcases how machine learning can be used in real-world scenarios such as viticulture and beverage quality control.

---

## 🎯 Objective
The primary goal is to build and evaluate machine learning models to accurately classify wine quality and provide insights into the key chemical factors affecting wine quality.

---

## 📂 Dataset
The project uses the [Wine Quality Dataset](https://www.kaggle.com/datasets/yasserh/wine-quality-dataset) from Kaggle. The dataset includes the following:
- **Features:** Chemical characteristics of the wine (e.g., fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, etc.).
- **Target:** Wine quality score on a scale (0–10).

---

## 🛠️ Tools and Technologies
1. **Programming Language:** Python
2. **Libraries:**
   - **Data Analysis:** Pandas, NumPy
   - **Visualization:** Matplotlib, Seaborn
   - **Machine Learning:** Scikit-Learn
3. **Version Control:** Git, GitHub

---

## 🔬 Methodology
1. **Data Preprocessing:**
   - Cleaned the dataset by handling missing values (if any).
   - Converted wine quality scores into binary categories: 
     - **Good Quality:** Quality ≥ 7
     - **Bad Quality:** Quality < 7
2. **Exploratory Data Analysis (EDA):**
   - Visualized data distribution and correlations to identify significant features.
3. **Model Building:**
   - Trained and evaluated three classifiers:
     - Random Forest
     - Stochastic Gradient Descent (SGD)
     - Support Vector Classifier (SVC)
4. **Model Evaluation:**
   - Used metrics like accuracy and classification reports to evaluate model performance.

---

## 📊 Results
- **Random Forest Classifier:** Accuracy - XX%
- **SGD Classifier:** Accuracy - XX%
- **SVC:** Accuracy - XX%

> Detailed classification reports are available in the project output.

---

## 🖥️ How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/phanisurya-7/OIBSIP-PROJECT-2-PROPOSAL-LEVEL-2.git
   cd wine-quality-prediction
   ```
2. Install required libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. Place the dataset (`winequality.csv`) in the project folder.
4. Run the script:
   ```bash
   python wine_quality_prediction.py
   ```

---

## 📈 Key Insights
- **Alcohol Content:** Wines with higher alcohol content tend to have better quality.
- **Acidity Levels:** Balanced acidity is a significant indicator of good wine quality.
- **Residual Sugar:** Minimal impact on quality, but higher levels are often found in dessert wines.

---

## 🌟 Future Enhancements
- Experiment with additional models like Gradient Boosting or XGBoost.
- Perform hyperparameter tuning for better model accuracy.
- Develop a web interface for users to predict wine quality using custom inputs.

---

## 🤝 Acknowledgments
- Dataset provided by Kaggle ([Wine Quality Dataset](https://www.kaggle.com/datasets/yasserh/wine-quality-dataset)).
- Inspiration from real-world applications in viticulture and food quality control.

---

Feel free to contribute or share your thoughts!
``
