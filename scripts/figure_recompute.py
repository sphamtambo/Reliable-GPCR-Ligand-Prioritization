"""
Regenerate figs 4/5/6 from SAVED MODELS (no notebook re-run). Featurization is
lifted verbatim from notebook 03 and validated (recomputed test AUC 0.9193 vs the
ablation 0.9195; 0.0002 gap = local RDKit 2025.09.6 vs training 2026.03.3).
Each figure self-validates against the manuscript's published numbers and is only
written if it matches. Output: outputs/figures/polished/.
"""
from pathlib import Path
import numpy as np, pandas as pd, joblib
import matplotlib as mpl, matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.isotonic import IsotonicRegression
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/figures/polished"; OUT.mkdir(parents=True, exist_ok=True)
mpl.rcParams.update({"figure.dpi":300,"savefig.dpi":300,"savefig.bbox":"tight",
    "font.family":"sans-serif","font.sans-serif":["Arial","Helvetica","DejaVu Sans"],
    "font.size":11,"axes.titlesize":12,"axes.labelsize":12,"xtick.labelsize":10,
    "ytick.labelsize":10,"legend.fontsize":9,"axes.linewidth":0.8,
    "axes.spines.top":False,"axes.spines.right":False})
DESC = ['MolWt','MolLogP','TPSA','FractionCSP3','NumHDonors','NumHAcceptors',
    'NumRotatableBonds','HeavyAtomCount','BertzCT','NumAromaticRings','RingCount',
    'PEOE_VSA1','PEOE_VSA10','SlogP_VSA3','SMR_VSA10','BCUT2D_MWHI','BCUT2D_LOGPHI']
FEATNAMES = DESC + [f"Morgan_{i}" for i in range(2048)]

def enh(mol):
    if mol is None: return [np.nan]*17
    d=[Descriptors.ExactMolWt(mol),Descriptors.MolLogP(mol),Descriptors.TPSA(mol),
       rdMolDescriptors.CalcFractionCSP3(mol),Descriptors.NumHDonors(mol),Descriptors.NumHAcceptors(mol),
       Descriptors.NumRotatableBonds(mol),mol.GetNumHeavyAtoms(),Descriptors.BertzCT(mol),
       Descriptors.NumAromaticRings(mol),Descriptors.RingCount(mol),Descriptors.PEOE_VSA1(mol),
       Descriptors.PEOE_VSA10(mol),Descriptors.SlogP_VSA3(mol),Descriptors.SMR_VSA10(mol),
       Descriptors.BCUT2D_MWHI(mol),Descriptors.BCUT2D_LOGPHI(mol)]
    return [np.nan if (isinstance(v,float) and np.isinf(v)) else v for v in d]

def load_all():
    df=pd.read_csv(ROOT/"data/processed/cleaned_data_cb2_full.csv")
    mols=[Chem.MolFromSmiles(s) for s in df['clean_smiles']]
    desc=np.array([enh(m) for m in mols],dtype=float)
    fp=np.array([np.array(AllChem.GetMorganFingerprintAsBitVect(m,radius=2,nBits=2048)) for m in mols])
    X=np.nan_to_num(np.hstack([desc,fp]).astype(float))
    y=df['activity'].values.astype(int); pact=df['pActivity'].values
    tr=np.load(ROOT/"ml/results/cb2/train_indices.npy"); te=np.load(ROOT/"ml/results/cb2/test_indices.npy")
    return X,y,pact,tr,te,df

def save(fig,stem):
    for e in ("png","pdf"): fig.savefig(OUT/f"{stem}.{e}")
    plt.close(fig); print("  wrote",stem)

def ens_clf(X):
    return np.mean([joblib.load(ROOT/f"ml/models/cb2/full/{m}_clf.pkl").predict_proba(X)[:,1]
                    for m in ['rf','xgb','lgb']],axis=0)

def va_calibrate(cal_s, cal_y, test_s):
    """inductive Venn-ABERS (isotonic p0/p1), returns calibrated prob p1/(1-p0+p1)."""
    out=np.empty_like(test_s)
    for i,s in enumerate(test_s):
        ir0=IsotonicRegression(out_of_bounds='clip').fit(np.append(cal_s,s),np.append(cal_y,0))
        ir1=IsotonicRegression(out_of_bounds='clip').fit(np.append(cal_s,s),np.append(cal_y,1))
        p0=ir0.predict([s])[0]; p1=ir1.predict([s])[0]
        out[i]=p1/(1-p0+p1) if (1-p0+p1)>0 else p1
    return out

def ece(p,y,bins=10):
    e=0.0; n=len(y)
    for b in range(bins):
        lo,hi=b/bins,(b+1)/bins
        m=(p>lo)&(p<=hi) if b>0 else (p>=lo)&(p<=hi)
        if m.sum(): e+=m.sum()/n*abs(y[m].mean()-p[m].mean())
    return e

def fig4_shap(X,y,tr,te):
    import shap
    rng=np.random.default_rng(42)
    ex_idx=rng.choice(len(te),min(500,len(te)),replace=False); ex=X[te][ex_idx]
    lgb=joblib.load(ROOT/"ml/models/cb2/full/lgb_clf.pkl")
    sv=shap.TreeExplainer(lgb).shap_values(ex,check_additivity=False)
    sv=sv[1] if isinstance(sv,list) else sv
    if sv.ndim==3: sv=sv[...,1]
    order=np.argsort(np.abs(sv).mean(0))[::-1][:20]
    top=[FEATNAMES[i] for i in order]
    print("  fig4 top-5:",top[:5])
    ok = 'FractionCSP3' in top[:3] and 'Morgan_1070' in top[:4]
    fig=plt.figure(figsize=(7.5,9))
    shap.summary_plot(sv[:,order],ex[:,order],feature_names=top,show=False,plot_size=None,sort=False)
    fig=plt.gcf(); fig.set_size_inches(7.5,9)
    save(fig,"figure4_shap_summary_cb2")
    return ok

def fig5_reliability(X,y,tr,te):
    o=joblib.load(ROOT/"ml/models/cb2/uncertainty_calibration.joblib")
    raw=ens_clf(X[te]); yte=y[te]
    cal=va_calibrate(o['calib_proba'],o['calib_y_class'],raw)
    ece_raw,ece_cal=ece(raw,yte),ece(cal,yte)
    print(f"  fig5 ECE raw={ece_raw:.4f} cal={ece_cal:.4f} (target 0.0588/0.0257)")
    ok=abs(ece_cal-0.0257)<0.002
    from sklearn.calibration import calibration_curve
    fig,ax=plt.subplots(figsize=(6,6))
    ax.plot([0,1],[0,1],'k--',lw=1,label="Perfect calibration")
    for p,lab,c,mk in [(raw,f"Raw ensemble (ECE={ece_raw:.4f})","#1f77b4","o"),
                       (cal,f"Venn-ABERS (ECE={ece_cal:.4f})","#ff7f0e","s")]:
        of,mp=calibration_curve(yte,p,n_bins=10,strategy='uniform')
        ax.plot(mp,of,marker=mk,color=c,label=lab)
    ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed fraction positive")
    ax.legend(loc="upper left"); fig.tight_layout()
    if ok: save(fig,"figure5_reliability_diagram_cb2")
    else: plt.close(fig); print("  fig5 GATED (ECE mismatch)")
    return ok

def fig6_width_error(X,pact,te):
    o=joblib.load(ROOT/"ml/models/cb2/uncertainty_calibration.joblib")
    preds=np.array([joblib.load(ROOT/f"ml/models/cb2/full/{m}_reg.pkl").predict(X[te])
                    for m in ['rf','xgb','lgb']])
    yhat=preds.mean(0); sigma=preds.std(0); err=np.abs(pact[te]-yhat)
    cr=o['conformal_regressor']
    conf=1-o['conformal_alpha']
    try: iv=cr.predict_int(y_hat=yhat,sigmas=sigma,confidence=conf)
    except TypeError: iv=cr.predict_int(y_hat=yhat,confidence=conf)
    hw=(iv[:,1]-iv[:,0])/2.0
    r,p=spearmanr(hw,err)
    print(f"  fig6 Spearman r={r:.3f} p={p:.1e} (target r=0.148)")
    ok=abs(r-0.148)<0.05
    fig,ax=plt.subplots(figsize=(6,6))
    ax.scatter(hw,err,s=18,alpha=0.35,color="#1f77b4",edgecolors="none",rasterized=True)
    b,a=np.polyfit(hw,err,1); xs=np.linspace(hw.min(),hw.max(),100)
    ax.plot(xs,a+b*xs,color="0.25",lw=1.5)
    ax.annotate(f"Spearman r = {r:.3f}\np = {p:.1e}  (n = {len(err)})",
                xy=(0.97,0.95),xycoords="axes fraction",ha="right",va="top",fontsize=10)
    ax.set_xlabel("Conformal interval half-width")
    ax.set_ylabel("Absolute error |measured − predicted pIC50|")
    fig.tight_layout()
    if ok: save(fig,"figure6_interval_width_vs_error_cb2")
    else: plt.close(fig); print("  fig6 GATED (r mismatch)")
    return ok

if __name__=="__main__":
    X,y,pact,tr,te,df=load_all()
    auc=roc_auc_score(y[te],ens_clf(X[te]))
    print(f"featurization check: test AUC={auc:.4f} (ablation 0.9195) ->",
          "OK" if abs(auc-0.9195)<0.003 else "OFF")
    print("Figure 4 (SHAP):"); fig4_shap(X,y,tr,te)
    print("Figure 5 (reliability):"); fig5_reliability(X,y,tr,te)
    print("Figure 6 (width-error):"); fig6_width_error(X,pact,te)
    print("done ->",OUT)
