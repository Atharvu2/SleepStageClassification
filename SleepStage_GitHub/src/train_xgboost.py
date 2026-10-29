"""Run XGBoost using leave-one-subject-out validation on the saved feature matrix."""
from pathlib import Path
import json, time
import numpy as np, pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
import matplotlib.pyplot as plt
import seaborn as sns
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'results'; FIG=ROOT/'figures'
NAMES=['Wake','N1','N2','N3','REM']
def main():
    d=np.load(OUT/'features.npz',allow_pickle=True); X=d['X']; y=d['y'].astype(int); groups=d['groups'].astype(str)
    logo=LeaveOneGroupOut(); yt=[]; yp=[]; folds=[]; start=time.time()
    for fold,(tr,te) in enumerate(logo.split(X,y,groups),1):
        model=XGBClassifier(n_estimators=25,max_depth=3,learning_rate=.1,subsample=.85,colsample_bytree=.85,objective='multi:softprob',num_class=5,eval_metric='mlogloss',tree_method='hist',n_jobs=1,random_state=42)
        weights=compute_sample_weight(class_weight='balanced',y=y[tr])
        model.fit(X[tr],y[tr],sample_weight=weights)
        pred=model.predict(X[te]).astype(int); yt.extend(y[te].tolist()); yp.extend(pred.tolist())
        folds.append({'held_out_subject':str(groups[te][0]),'n_test':len(te),'accuracy':float(accuracy_score(y[te],pred)),'macro_f1':float(f1_score(y[te],pred,average='macro',zero_division=0))})
        (OUT/'xgboost_progress.json').write_text(json.dumps({'completed_folds':folds},indent=2))
        print(f"fold {fold}/5 {groups[te][0]} acc={folds[-1]['accuracy']:.4f}",flush=True)
    metrics={'model':'XGBoost','accuracy':float(accuracy_score(yt,yp)),'macro_f1':float(f1_score(yt,yp,average='macro',zero_division=0)),'weighted_f1':float(f1_score(yt,yp,average='weighted',zero_division=0)),'cohen_kappa':float(cohen_kappa_score(yt,yp)),'seconds':round(time.time()-start,1),'n_epochs':len(yt),'folds':folds}
    (OUT/'xgboost_results.json').write_text(json.dumps(metrics,indent=2))
    report=classification_report(yt,yp,labels=[0,1,2,3,4],target_names=NAMES,output_dict=True,zero_division=0)
    (OUT/'xgboost_classification_report.json').write_text(json.dumps(report,indent=2))
    pd.DataFrame({'y_true':yt,'y_pred':yp}).to_csv(OUT/'xgboost_predictions.csv',index=False)
    cm=confusion_matrix(yt,yp,labels=[0,1,2,3,4]); plt.figure(figsize=(7,6)); sns.heatmap(cm,annot=True,fmt='d',xticklabels=NAMES,yticklabels=NAMES,cmap='Blues'); plt.xlabel('Predicted'); plt.ylabel('True'); plt.title('XGBoost — subject-independent predictions'); plt.tight_layout(); plt.savefig(FIG/'confusion_XGBoost.png',dpi=160); plt.close()
    print('FINAL',json.dumps(metrics),flush=True)
if __name__=='__main__': main()
