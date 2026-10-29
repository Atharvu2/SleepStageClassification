from pathlib import Path
from collections import Counter
import numpy as np, pandas as pd, mne
from scipy.signal import welch, butter, sosfiltfilt, resample_poly
from scipy.stats import skew, kurtosis, entropy
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'data'/'raw'; OUT=ROOT/'results'; OUT.mkdir(exist_ok=True)
FS=100; EPOCH_SEC=30; N=FS*EPOCH_SEC
BANDS={'delta':(0.5,4),'theta':(4,8),'alpha':(8,12),'sigma':(12,16),'beta':(16,30)}

def feat_batch(epochs, fs=FS):
    x=np.asarray(epochs,dtype=np.float64); x=x-x.mean(axis=1,keepdims=True); eps=1e-12
    f,p=welch(x,fs=fs,nperseg=min(x.shape[1],fs*4),axis=1)
    totalmask=(f>=0.5)&(f<=40); total=np.trapezoid(p[:,totalmask],f[totalmask],axis=1)+eps
    dx=np.diff(x,axis=1); ddx=np.diff(dx,axis=1)
    v0=np.var(x,axis=1)+eps; v1=np.var(dx,axis=1)+eps; v2=np.var(ddx,axis=1)+eps
    out=[x.mean(axis=1),x.std(axis=1),x.min(axis=1),x.max(axis=1),np.sqrt(np.mean(x*x,axis=1)),skew(x,axis=1),kurtosis(x,axis=1),np.sqrt(v0),np.sqrt(v1/v0),np.sqrt(v2/v1),np.sqrt(np.maximum(0,v0*v2-v1*v1))/(v1+eps)]
    for lo,hi in BANDS.values():
        m=(f>=lo)&(f<hi); power=np.trapezoid(p[:,m],f[m],axis=1) if m.sum()>1 else np.zeros(len(x))
        out += [power,power/total]
    pp=p[:,totalmask]+eps; probs=pp/pp.sum(axis=1,keepdims=True); out += [-(probs*np.log(probs)).sum(axis=1)]
    return np.nan_to_num(np.column_stack(out),nan=0,posinf=0,neginf=0)

def labels_for(night, n_epochs):
    files=sorted(night.glob('*Expert*.csv'))
    arr=[]
    for f in files:
        try:
            a=pd.read_csv(f,header=None).iloc[:,0].to_numpy(dtype=int)
            arr.append(a)
        except Exception as e: print('Label read warning',f,e)
    if not arr: return np.full(n_epochs,-1,dtype=int)
    L=min([len(a) for a in arr]+[n_epochs]); mat=np.vstack([a[:L] for a in arr])
    # Majority vote; ties resolve to the smallest stage code and are noted in methods.
    voted=np.array([Counter(mat[:,i]).most_common()[-1][0] if False else Counter(mat[:,i]).most_common(1)[0][0] for i in range(L)])
    out=np.full(n_epochs,-1,dtype=int); out[:L]=voted
    return out

def main():
    X=[]; y=[]; groups=[]; meta=[]; used=0
    edfs=sorted(RAW.rglob('*.edf'))
    print(f'Found {len(edfs)} EDF recordings')
    for path in edfs:
        subj=path.parts[-3]; night=path.parent
        try:
            raw=mne.io.read_raw_edf(path,preload=True,verbose='ERROR')
            candidates=[c for c in ['C4-A1','F4-A1','O2-A1','P4-A1'] if c in raw.ch_names]
            if not candidates: candidates=[c for c in raw.ch_names if 'EEG' in c.upper()]
            if not candidates: print('No EEG channel, skip',path); continue
            raw.pick([candidates[0]]); raw.resample(FS,verbose='ERROR')
            # Bandpass EEG 0.5–40 Hz. Apply after resampling.
            raw.filter(0.5,40,method='iir',iir_params={'order':4,'ftype':'butter'},verbose='ERROR')
            sig=raw.get_data()[0]; n_epochs=len(sig)//N
            if n_epochs==0: continue
            sig=sig[:n_epochs*N].reshape(n_epochs,N)
            lab=labels_for(night,n_epochs)
            n=min(len(sig),len(lab)); sig=sig[:n]; lab=lab[:n]
            keep=np.isin(lab,[0,1,2,3,4])
            sig=sig[keep]; lab=lab[keep]
            if not len(lab): continue
            feats=feat_batch(sig)
            X.append(feats); y.append(lab); groups.extend([subj]*len(lab));
            meta.extend([(subj,path.stem,i,int(lbl)) for i,lbl in enumerate(lab)])
            used+=1; print(f'{path.stem}: {len(lab)} labelled epochs, channel={candidates[0]}')
        except Exception as e: print('ERROR processing',path,e)
    if not X: raise RuntimeError('No data processed. Check EDF and annotation files.')
    X=np.vstack(X); y=np.concatenate(y); groups=np.asarray(groups)
    np.savez_compressed(OUT/'features.npz',X=X,y=y,groups=groups)
    pd.DataFrame(meta,columns=['subject','recording','epoch_index_after_filter','label']).to_csv(OUT/'epoch_metadata.csv',index=False)
    print('Saved features:',X.shape,'labels:',Counter(y),'subjects:',Counter(groups))
if __name__=='__main__': main()
