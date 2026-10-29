from pathlib import Path
import json, time
import numpy as np, pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
import matplotlib.pyplot as plt, seaborn as sns
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'results'; FIG=ROOT/'figures'; OUT.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
NAMES=['Wake','N1','N2','N3','REM']
def main():
 d=np.load(OUT/'features.npz',allow_pickle=True); X=d['X']; y=d['y'].astype(int); groups=d['groups'].astype(str)
 models={
 'RandomForest':RandomForestClassifier(n_estimators=180,min_samples_leaf=2,class_weight='balanced_subsample',n_jobs=-1,random_state=42),
 'SVM_RBF':make_pipeline(SimpleImputer(),StandardScaler(),SVC(C=3,kernel='rbf',class_weight='balanced',cache_size=1500)),
 'KNN':make_pipeline(SimpleImputer(),StandardScaler(),KNeighborsClassifier(n_neighbors=9,weights='distance',n_jobs=-1)),
 'ExtraTrees':ExtraTreesClassifier(n_estimators=180,min_samples_leaf=2,class_weight='balanced',n_jobs=-1,random_state=42),
 'XGBoost':XGBClassifier(n_estimators=25,max_depth=3,learning_rate=0.1,subsample=0.85,colsample_bytree=0.85,objective='multi:softprob',num_class=5,eval_metric='mlogloss',tree_method='hist',n_jobs=1,random_state=42)
 }
 logo=LeaveOneGroupOut(); rows=[]; allpred={}; reports={}
 for name,model in models.items():
  print('\nMODEL',name,flush=True); yt=[]; yp=[]; folds=[]; t=time.time()
  for fold,(tr,te) in enumerate(logo.split(X,y,groups),1):
   if name=='XGBoost':
    weights=compute_sample_weight(class_weight='balanced',y=y[tr])
    model.fit(X[tr],y[tr],sample_weight=weights)
   else:
    model.fit(X[tr],y[tr])
   pred=model.predict(X[te]); yt.extend(y[te]); yp.extend(pred)
   folds.append({'held_out_subject':str(groups[te][0]),'n_test':len(te),'accuracy':accuracy_score(y[te],pred),'macro_f1':f1_score(y[te],pred,average='macro',zero_division=0)})
   print(f' fold {fold}/{len(np.unique(groups))} subject={groups[te][0]} acc={folds[-1]["accuracy"]:.4f}',flush=True)
  metrics={'model':name,'accuracy':accuracy_score(yt,yp),'macro_f1':f1_score(yt,yp,average='macro',zero_division=0),'weighted_f1':f1_score(yt,yp,average='weighted',zero_division=0),'cohen_kappa':cohen_kappa_score(yt,yp),'seconds':round(time.time()-t,1),'n_epochs':len(yt),'folds':folds}
  rows.append(metrics); allpred[name]=(np.asarray(yt),np.asarray(yp)); reports[name]=classification_report(yt,yp,labels=[0,1,2,3,4],target_names=NAMES,output_dict=True,zero_division=0)
  cm=confusion_matrix(yt,yp,labels=[0,1,2,3,4]); plt.figure(figsize=(7,6)); sns.heatmap(cm,annot=True,fmt='d',xticklabels=NAMES,yticklabels=NAMES,cmap='Blues'); plt.xlabel('Predicted'); plt.ylabel('True'); plt.title(f'{name} — subject-independent predictions'); plt.tight_layout(); plt.savefig(FIG/f'confusion_{name}.png',dpi=160); plt.close()
  print('RESULT',name,metrics['accuracy'],metrics['macro_f1'],metrics['cohen_kappa'],flush=True)
 pd.DataFrame([{k:v for k,v in r.items() if k!='folds'} for r in rows]).to_csv(OUT/'model_comparison.csv',index=False)
 (OUT/'fold_metrics.json').write_text(json.dumps(rows,indent=2)); (OUT/'classification_reports.json').write_text(json.dumps(reports,indent=2))
 # comparison chart
 df=pd.DataFrame(rows); ax=df.set_index('model')[['accuracy','macro_f1','weighted_f1']].plot(kind='bar',figsize=(10,5),ylim=(0,1)); ax.set_ylabel('Score'); ax.set_title('Subject-independent model comparison'); plt.xticks(rotation=20,ha='right'); plt.tight_layout(); plt.savefig(FIG/'model_comparison.png',dpi=160); plt.close()
 print('\nFINAL RESULTS\n',df[['model','accuracy','macro_f1','weighted_f1','cohen_kappa','seconds']].to_string(index=False),flush=True)
if __name__=='__main__': main()
