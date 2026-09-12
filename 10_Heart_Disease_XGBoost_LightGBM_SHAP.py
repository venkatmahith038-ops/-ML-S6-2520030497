from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import xgboost as xgb
import lightgbm as lgb
import shap
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

data = fetch_openml(name='heart-statlog', version=1, as_frame=True, parser='auto')
X, y = data.data, data.target

le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)

lgb_model = lgb.LGBMClassifier()
lgb_model.fit(X_train, y_train)

explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test, show=False)
plt.savefig('shap_summary.png')
