from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"data"/"logistics_predictive_dataset.csv"
OUT=ROOT/"outputs"; CHARTS=OUT/"charts"; TABLES=OUT/"tables"; MODELS=ROOT/"models"
for p in [CHARTS,TABLES,MODELS]: p.mkdir(parents=True,exist_ok=True)
df=pd.read_csv(DATA)
target="actual_delivery_days"
features=["region","transport_mode","shipment_type","distance_km","shipment_volume_units","weight_kg","traffic_level_1_to_5","weather_condition"]
X=df[features]; y=df[target]
cat=["region","transport_mode","shipment_type","weather_condition"]
num=["distance_km","shipment_volume_units","weight_kg","traffic_level_1_to_5"]
pre=ColumnTransformer([
("num",Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler())]),num),
("cat",Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]),cat)])
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=42)
models={"Linear Regression":LinearRegression(),
"Decision Tree":DecisionTreeRegressor(random_state=42,max_depth=7,min_samples_leaf=5),
"Random Forest":RandomForestRegressor(random_state=42,n_estimators=180,min_samples_leaf=2),
"Gradient Boosting":GradientBoostingRegressor(random_state=42,n_estimators=120,max_depth=2,learning_rate=.05)}
rows=[]
for name,model in models.items():
 pipe=Pipeline([("preprocess",pre),("model",model)])
 pipe.fit(X_train,y_train); pred=pipe.predict(X_test)
 rows.append({"model":name,"MAE_days":mean_absolute_error(y_test,pred),"RMSE_days":mean_squared_error(y_test,pred)**.5,"R2":r2_score(y_test,pred)})
res=pd.DataFrame(rows).sort_values("RMSE_days")
res.to_csv(TABLES/"model_comparison.csv",index=False)
cv=Pipeline([("preprocess",pre),("model",RandomForestRegressor(random_state=42,n_estimators=150,min_samples_leaf=2))])
scores=-cross_val_score(cv,X_train,y_train,cv=KFold(n_splits=5,shuffle=True,random_state=42),scoring="neg_mean_absolute_error")
pd.DataFrame({"fold":range(1,6),"MAE_days":scores}).to_csv(TABLES/"cross_validation_scores.csv",index=False)
gridpipe=Pipeline([("preprocess",pre),("model",RandomForestRegressor(random_state=42))])
grid=GridSearchCV(gridpipe,{"model__n_estimators":[100,180],"model__max_depth":[None,12],"model__min_samples_leaf":[1,3]},cv=KFold(n_splits=5,shuffle=True,random_state=42),scoring="neg_mean_absolute_error",n_jobs=-1)
grid.fit(X_train,y_train); best=grid.best_estimator_; pred=best.predict(X_test)
metrics={"model":"Tuned Random Forest","MAE_days":mean_absolute_error(y_test,pred),"RMSE_days":mean_squared_error(y_test,pred)**.5,"R2":r2_score(y_test,pred)}
pd.DataFrame([metrics]).to_csv(TABLES/"tuned_model_metrics.csv",index=False)
joblib.dump(best,MODELS/"delivery_time_model.joblib")
plt.figure(figsize=(8,5));plt.bar(res.model,res.RMSE_days,edgecolor="black");plt.ylabel("RMSE (days)");plt.title("Model Comparison: Test RMSE");plt.xticks(rotation=20,ha="right");plt.tight_layout();plt.savefig(CHARTS/"model_comparison_rmse.png",dpi=150);plt.close()
plt.figure(figsize=(7,6));plt.scatter(y_test,pred,alpha=.65);lims=[min(y_test.min(),pred.min()),max(y_test.max(),pred.max())];plt.plot(lims,lims,linestyle="--");plt.xlabel("Actual delivery days");plt.ylabel("Predicted delivery days");plt.title("Actual vs Predicted Delivery Time");plt.tight_layout();plt.savefig(CHARTS/"actual_vs_predicted.png",dpi=150);plt.close()
plt.figure(figsize=(8,5));plt.hist(y_test-pred,bins=20,edgecolor="black");plt.xlabel("Prediction error (actual - predicted days)");plt.ylabel("Shipment count");plt.title("Prediction Error Distribution");plt.tight_layout();plt.savefig(CHARTS/"prediction_error_distribution.png",dpi=150);plt.close()
names=best.named_steps["preprocess"].get_feature_names_out(); imp=best.named_steps["model"].feature_importances_
fi=pd.DataFrame({"feature":names,"importance":imp}).sort_values("importance",ascending=False).head(12);fi.to_csv(TABLES/"feature_importance.csv",index=False)
plt.figure(figsize=(8,5));plt.barh(fi.feature[::-1],fi.importance[::-1],edgecolor="black");plt.xlabel("Feature importance");plt.title("Top Predictors of Delivery Time");plt.tight_layout();plt.savefig(CHARTS/"feature_importance.png",dpi=150);plt.close()
test=X_test.reset_index(drop=True).copy();test["actual_delivery_days"]=y_test.reset_index(drop=True);test["predicted_delivery_days"]=pred;test["prediction_error_days"]=test.actual_delivery_days-test.predicted_delivery_days
test["priority"]=np.where(test.predicted_delivery_days>=test.predicted_delivery_days.quantile(.75),"High","Normal")
test.to_csv(TABLES/"test_predictions_and_priority.csv",index=False)
df.groupby("transport_mode").agg(shipments=("shipment_id","count"),avg_delivery_days=("actual_delivery_days","mean"),avg_cost_inr=("transportation_cost_inr","mean")).round(2).to_csv(TABLES/"mode_tradeoff_summary.csv")
print("Model training and evaluation complete.")
print(res.to_string(index=False))
print("Tuned Random Forest:",metrics)
print("Best parameters:",grid.best_params_)
print("Outputs saved under outputs/ and models/.")
