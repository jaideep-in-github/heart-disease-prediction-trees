# 🧠 Decision Trees & Random Forests – Heart Disease Prediction

This project is part of an AI & ML internship task where I explored tree-based models to classify heart disease presence using a real-world dataset. I went beyond just model training and tried to understand *how* and *why* these models make decisions.

---

## 📌 Objective

To train and evaluate Decision Tree and Random Forest models on a heart disease dataset, visualize the decision process, analyze model performance, and interpret feature importance.

---

## 🛠️ Tools & Libraries Used

- **Python (Pandas, NumPy)** – for data handling
- **scikit-learn** – model building, evaluation
- **matplotlib & seaborn** – data and feature importance visualization
- **graphviz** – for rendering decision trees

---

## 📂 What I Did

### ✅ Step 1: Loaded the Dataset
I worked with the `heart.csv` dataset containing medical features like cholesterol, blood pressure, age, etc. The goal was to predict the `target` column (1 = disease, 0 = no disease).

### ✅ Step 2: Trained a Basic Decision Tree
Started with a simple decision tree. Accuracy was okay, but I noticed it overfitted — classic behavior when depth isn’t restricted.

### ✅ Step 3: Controlled Overfitting
By limiting the tree depth (`max_depth=3`), I got a more general model, though slightly less accurate. It helped in balancing bias and variance.

### ✅ Step 4: Trained a Random Forest
Moved on to Random Forest (an ensemble of decision trees). Accuracy improved significantly. It handled overfitting better and gave more stable results.

### ✅ Step 5: Feature Importance
I plotted feature importances using `rf.feature_importances_`. It was interesting to see that `cp`, `thalach`, and `ca` had the most influence on predictions.

### ✅ Step 6: Cross-Validation
To be sure about my model's performance, I did 5-fold cross-validation. It confirmed that Random Forest was more reliable across different data splits.

---

## 📈 Results

| Model              | Accuracy (Test Set)         |
|--------------------|-----------------------------|
| Decision Tree      | 0.9853658536585366          |
| Limited Depth Tree | 0.7804878048780488          |
| Random Forest      | 0.9853658536585366          |
| 5-Fold CV (RF)     | 0.9970731707317073          |

---

## 🖼️ Files in this Repo

- `main.py` — Full source code
- `heart.csv` — Dataset
- `decision_tree.png` — Visual of trained decision tree
- `feature_importances.png` — Bar chart of feature influence
- `README.md` — You’re reading it :

---

## 🔚 Final Thoughts

This task was a great hands-on way to understand how tree-based models work. It taught me not just how to train them, but how to interpret their decisions, avoid overfitting, and trust their predictions. Looking forward to applying this in more real-world problems!

