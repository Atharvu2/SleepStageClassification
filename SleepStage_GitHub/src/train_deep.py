"""Optional raw-EEG deep-learning experiment. Run only after classical baseline.
Builds 30-s, 100-Hz C4-A1 epochs and evaluates CNN and CNN-LSTM with LOSO CV.
This is computationally heavier than train_classical.py."""
from pathlib import Path
from collections import Counter
import numpy as np, pandas as pd, mne, torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
import json, time
ROOT=Path(__file__).resolve().parents[1]; RAW=ROOT/'data'/'raw'; OUT=ROOT/'results'; OUT.mkdir(exist_ok=True)
FS=100; EPOCH=3000; DEVICE='cuda' if torch.cuda.is_available() else 'cpu'

def read_labels(night,n):
 arr=[]
 for p in sorted(night.glob('*Expert*.csv')):
  try: arr.append(pd.read_csv(p,header=None).iloc[:,0].astype(int).to_numpy())
  except Exception: pass
 if not arr: return np.full(n,-1)
 L=min(n,*map(len,arr)); mat=np.vstack([a[:L] for a in arr]); out=np.full(n,-1)
 for i in range(L): out[i]=Counter(mat[:,i]).most_common(1)[0][0]
 return out

def load_raw_epochs():
 xs=[]; ys=[]; gs=[]
 for p in sorted(RAW.rglob('*.edf')):
  subj=p.parts[-3]; raw=mne.io.read_raw_edf(p,preload=True,verbose='ERROR')
  ch=next((c for c in ['C4-A1','F4-A1','O2-A1','P4-A1'] if c in raw.ch_names),None)
  if ch is None: continue
  raw.pick([ch]); raw.resample(FS,verbose='ERROR'); raw.filter(.5,40,method='iir',iir_params={'order':4,'ftype':'butter'},verbose='ERROR')
  sig=raw.get_data()[0]; n=len(sig)//EPOCH; sig=sig[:n*EPOCH].reshape(n,EPOCH); y=read_labels(p.parent,n)
  n=min(len(sig),len(y)); keep=np.isin(y[:n],[0,1,2,3,4]); sig=sig[:n][keep]; y=y[:n][keep]
  # Per-epoch normalisation limits amplitude scale variation; computed independently per epoch.
  sig=(sig-sig.mean(axis=1,keepdims=True))/(sig.std(axis=1,keepdims=True)+1e-6)
  xs.append(sig.astype('float32')); ys.append(y.astype('int64')); gs.extend([subj]*len(y))
  print('loaded',p.stem,len(y),flush=True)
 return np.concatenate(xs),np.concatenate(ys),np.asarray(gs)

class Net(nn.Module):
 def __init__(self,kind):
  super().__init__(); self.kind=kind
  self.conv=nn.Sequential(nn.Conv1d(1,16,15,padding=7),nn.BatchNorm1d(16),nn.ReLU(),nn.MaxPool1d(4),nn.Conv1d(16,32,9,padding=4),nn.BatchNorm1d(32),nn.ReLU(),nn.MaxPool1d(4),nn.Conv1d(32,64,7,padding=3),nn.ReLU(),nn.AdaptiveAvgPool1d(64))
  if kind=='CNN': self.head=nn.Sequential(nn.Flatten(),nn.Linear(64*64,128),nn.ReLU(),nn.Dropout(.4),nn.Linear(128,5))
  else: self.rnn=nn.LSTM(64,64,batch_first=True,bidirectional=True); self.head=nn.Sequential(nn.Dropout(.4),nn.Linear(128,5))
 def forward(self,x):
  z=self.conv(x)
  if self.kind=='CNN': return self.head(z)
  z=z.transpose(1,2); z,_=self.rnn(z); return self.head(z.mean(dim=1))

def main():
 X,y,g=load_raw_epochs(); logo=LeaveOneGroupOut(); results=[]
 for kind in ['CNN','CNN-LSTM']:
  true=[]; pred=[]; start=time.time()
  for fold,(tr,te) in enumerate(logo.split(X,y,g),1):
   model=Net(kind).to(DEVICE); opt=torch.optim.AdamW(model.parameters(),lr=1e-3); lossfn=nn.CrossEntropyLoss()
   counts=np.bincount(y[tr],minlength=5); weights=len(y[tr])/(5*np.maximum(counts,1)); lossfn=nn.CrossEntropyLoss(weight=torch.tensor(weights,dtype=torch.float32,device=DEVICE))
   dl=DataLoader(TensorDataset(torch.from_numpy(X[tr,None]),torch.from_numpy(y[tr])),batch_size=64,shuffle=True)
   for epoch in range(8):
    model.train()
    for xb,yb in dl:
     xb,yb=xb.to(DEVICE),yb.to(DEVICE); opt.zero_grad(); loss=lossfn(model(xb),yb); loss.backward(); opt.step()
   model.eval(); pp=[]
   with torch.no_grad():
    for i in range(0,len(te),128): pp.extend(model(torch.from_numpy(X[te[i:i+128],None]).to(DEVICE)).argmax(1).cpu().numpy())
   true.extend(y[te]); pred.extend(pp); print(kind,'fold',fold,'held out',g[te][0],flush=True)
  results.append({'model':kind,'accuracy':accuracy_score(true,pred),'macro_f1':f1_score(true,pred,average='macro',zero_division=0),'weighted_f1':f1_score(true,pred,average='weighted',zero_division=0),'cohen_kappa':cohen_kappa_score(true,pred),'seconds':round(time.time()-start,1),'device':DEVICE})
  print(results[-1],flush=True)
 (OUT/'deep_model_comparison.json').write_text(json.dumps(results,indent=2)); print(results)
if __name__=='__main__': main()
