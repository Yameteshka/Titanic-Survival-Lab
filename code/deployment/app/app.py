import os
import numpy as np
import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
COLS = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
PORTS = {"C": "Cherbourg", "Q": "Queenstown", "S": "Southampton"}
PRESETS = {
    "Custom": (1, "female", 29.0, 1, 0, 71.0, "S"),
    "First-class adult": (1, "female", 38.0, 1, 0, 71.28, "C"),
    "Third-class young adult": (3, "male", 22.0, 1, 0, 7.25, "S"),
    "Family traveler": (2, "female", 32.0, 1, 2, 41.58, "C"),
    "Child passenger": (2, "female", 8.0, 1, 2, 30.0, "S"),
}
INPUT_KEYS = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]

st.set_page_config(page_title="Titanic Survival Lab", page_icon="⚓", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');
:root{--b0:#070d18;--b1:#0b1424;--b2:#101d33;--bd:rgba(151,193,232,.14);--bd2:rgba(151,193,232,.28);--tx:#eaf3fc;--mu:#a7bad0;--dim:#6d8199;--cy:#5cd0f6;--cy2:#1e9ed8;--gold:#e7bb62;--gr:#41d99a;--rd:#f4796b}
.stApp{background:radial-gradient(900px 600px at 8% -10%,rgba(30,158,216,.18),transparent 62%),radial-gradient(700px 500px at 95% 12%,rgba(92,208,246,.08),transparent 58%),var(--b0);color:var(--tx);font-family:'IBM Plex Sans',sans-serif}.stApp:before{content:"";position:fixed;inset:0;pointer-events:none;background-image:linear-gradient(rgba(151,193,232,.035) 1px,transparent 1px),linear-gradient(90deg,rgba(151,193,232,.035) 1px,transparent 1px);background-size:46px 46px;mask-image:radial-gradient(1000px 700px at 25% 0,#000,transparent 80%)}
[data-testid="stHeader"]{background:transparent}.block-container{max-width:1220px;padding:2.2rem 2.2rem 3rem}h1,h2,h3{font-family:'Space Grotesk',sans-serif!important;color:var(--tx)!important}p{color:var(--mu)}
[data-testid="stSidebar"]{width:265px!important;background:linear-gradient(180deg,#0e1a2e,#09101e);border-right:1px solid var(--bd)}[data-testid="stSidebar"] .block-container{padding:1.35rem 1rem}.brand{display:flex;align-items:center;gap:12px;padding:3px 5px 20px;border-bottom:1px solid var(--bd);margin-bottom:12px}.mark{width:42px;height:42px;display:grid;place-items:center;border-radius:12px;background:linear-gradient(140deg,#a5e6ff,var(--cy2));color:#062331;font-size:21px}.brand strong{font:700 15px 'Space Grotesk';display:block}.brand small{font:500 9px 'IBM Plex Mono';letter-spacing:.12em;color:var(--dim)}
[data-testid="stSidebar"] [role="radiogroup"]{gap:6px}[data-testid="stSidebar"] label[data-baseweb="radio"]{padding:11px;border:1px solid transparent;border-radius:12px;color:var(--mu)}[data-testid="stSidebar"] label[data-baseweb="radio"]:has(input:checked){background:rgba(92,208,246,.11);border-color:var(--bd2);color:var(--tx)}[data-testid="stSidebar"] label[data-baseweb="radio"]>div:first-child{display:none}.side{margin-top:22px;padding:15px 8px;border-top:1px solid var(--bd)}.online{color:var(--gr);font-weight:600}.offline{color:var(--rd);font-weight:600}.dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:currentColor;margin-right:8px}.mono{font-family:'IBM Plex Mono';color:var(--dim);font-size:10px}
.head{display:flex;justify-content:space-between;align-items:flex-end;gap:24px;flex-wrap:wrap;margin-bottom:25px}.eye{font:500 11px 'IBM Plex Mono';letter-spacing:.22em;text-transform:uppercase;color:var(--cy);margin-bottom:8px}.eye:before{content:"";display:inline-block;width:25px;height:1px;background:var(--cy);margin:0 9px 3px 0}.head h1{font-size:clamp(27px,3vw,37px);margin:0}.head p{max-width:650px;margin:.45rem 0 0}.chips{display:flex;gap:8px}.chip{padding:7px 11px;border-radius:99px;border:1px solid var(--bd);background:rgba(151,193,232,.045);font:500 11px 'IBM Plex Mono';color:var(--mu)}.chip.ok:before{content:"";display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--gr);margin-right:7px}
[data-testid="stVerticalBlockBorderWrapper"]{background:linear-gradient(180deg,rgba(24,41,67,.72),rgba(13,23,41,.82));border:1px solid var(--bd);border-radius:18px;box-shadow:0 18px 44px -22px #020814;padding:7px}.title{font:700 19px 'Space Grotesk';color:var(--tx)}.sub{color:var(--dim);font-size:13px;margin:3px 0 14px}.kick{font:600 10px 'IBM Plex Mono';letter-spacing:.09em;text-transform:uppercase;color:var(--dim)}
div[data-baseweb="select"]>div,[data-testid="stNumberInput"] input,[data-testid="stFileUploaderDropzone"]{background:rgba(7,13,24,.48)!important;border-color:var(--bd2)!important}div[role="radiogroup"] label{background:rgba(7,13,24,.35);border:1px solid var(--bd);border-radius:9px;padding:7px 12px}.stButton>button,.stDownloadButton>button{border-radius:10px;border:1px solid var(--bd2);font-weight:700}button[data-testid="stBaseButton-primaryForm"],button[data-testid="stBaseButton-primary"]{background:linear-gradient(135deg,var(--cy),var(--cy2))!important;color:#062331!important;border:0!important;min-height:50px;box-shadow:0 10px 24px -10px rgba(30,158,216,.8)}
.empty{min-height:410px;display:flex;flex-direction:column;justify-content:center;align-items:center;text-align:center}.sonar{width:76px;height:76px;border:1px solid rgba(92,208,246,.28);border-radius:50%;display:grid;place-items:center;box-shadow:0 0 0 15px rgba(92,208,246,.035),0 0 0 30px rgba(92,208,246,.018);color:var(--cy);font-size:25px;margin-bottom:28px}.empty h3{margin:0}.empty p{font-size:13px;color:var(--dim)}
.verdict{text-align:center;padding-top:8px}.badge{display:inline-block;border-radius:99px;padding:7px 14px;font:600 13px 'IBM Plex Mono'}.pos{color:var(--gr);background:rgba(65,217,154,.12);border:1px solid rgba(65,217,154,.28)}.neg{color:var(--rd);background:rgba(244,121,107,.12);border:1px solid rgba(244,121,107,.28)}.gauge{--p:50;--a:var(--cy);width:190px;height:190px;margin:20px auto;border-radius:50%;display:grid;place-items:center;background:conic-gradient(var(--a) calc(var(--p)*1%),rgba(151,193,232,.08) 0);position:relative}.gauge:after{content:"";position:absolute;inset:15px;background:#0d192c;border-radius:50%;border:1px solid var(--bd)}.gauge>div{z-index:1;text-align:center}.gauge strong{font:700 40px 'Space Grotesk'}.gauge span{display:block;width:110px;font-size:10px;color:var(--dim);line-height:1.2}.prow{display:flex;justify-content:space-between;font-size:12px;margin-top:8px}.bar{height:7px;background:rgba(151,193,232,.08);border-radius:10px;overflow:hidden}.bar i{display:block;height:100%;background:linear-gradient(90deg,var(--cy2),var(--cy))}.bar.red i{background:linear-gradient(90deg,#a05248,var(--rd))}.threshold{text-align:center;margin:15px;color:var(--dim);font:500 11px 'IBM Plex Mono'}.pills{display:flex;gap:6px;flex-wrap:wrap}.pill{border:1px solid var(--bd);border-radius:99px;padding:5px 9px;font:500 10px 'IBM Plex Mono';color:var(--mu)}
.derived{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}.ditem{padding:9px;border:1px solid var(--bd);border-radius:9px;background:rgba(7,13,24,.32)}.ditem span{display:block;font-size:10px;color:var(--dim)}.ditem strong{font:600 12px 'IBM Plex Mono'}.grid3,.grid4{display:grid;gap:12px;margin:14px 0}.grid3{grid-template-columns:repeat(3,1fr)}.grid4{grid-template-columns:repeat(4,1fr)}.stat{padding:16px;border:1px solid var(--bd);border-radius:14px;background:linear-gradient(180deg,rgba(24,41,67,.7),rgba(13,23,41,.8));position:relative;overflow:hidden}.stat:before{content:"";position:absolute;left:0;right:0;top:0;height:2px;background:linear-gradient(90deg,var(--cy2),var(--cy))}.sl{font:500 10px 'IBM Plex Mono';color:var(--dim);text-transform:uppercase}.sv{font:700 22px 'Space Grotesk';margin:7px 0 2px}.ss{font-size:11px;color:var(--dim)}.track{height:4px;background:rgba(151,193,232,.08);border-radius:5px;margin-top:10px}.track i{display:block;height:100%;background:linear-gradient(90deg,var(--cy2),var(--cy))}.note{padding:13px;border-radius:10px;background:rgba(92,208,246,.07);border:1px solid rgba(92,208,246,.16);font:500 11px 'IBM Plex Mono';color:var(--mu)}
.pipe{display:flex;align-items:center;min-width:920px}.step{flex:1;text-align:center;padding:13px 8px;border:1px solid var(--bd);background:var(--b2);border-radius:11px;font-size:11px}.arr{color:var(--cy);padding:0 7px}.scroll{overflow-x:auto;padding:12px 0}.info{display:grid;grid-template-columns:1fr 1fr;gap:0 20px}.ir{display:flex;justify-content:space-between;padding:10px 0;border-bottom:1px solid var(--bd);font-size:12px}.ir span:first-child{color:var(--dim)}
@media(max-width:1000px){.grid4{grid-template-columns:repeat(2,1fr)}}@media(max-width:760px){.block-container{padding:1.4rem 1rem}.grid3,.grid4{grid-template-columns:1fr}.derived{grid-template-columns:repeat(2,1fr)}.chips{display:none}}
</style>""", unsafe_allow_html=True)


def api_get(path):
    try:
        r = requests.get(API_URL + path, timeout=5); r.raise_for_status(); return r.json()
    except requests.RequestException:
        return None


def header(eye, title, desc, health):
    chips = '<span class="chip ok">API Healthy</span><span class="chip">◆ Model Loaded</span>' if health else '<span class="chip">API Offline</span>'
    st.markdown(f'<div class="head"><div><div class="eye">{eye}</div><h1>{title}</h1><p>{desc}</p></div><div class="chips">{chips}</div></div>', unsafe_allow_html=True)


health = api_get("/health")
info = api_get("/model-info") if health else None
with st.sidebar:
    st.markdown('<div class="brand"><div class="mark">⚓</div><div><strong>Titanic Survival Lab</strong><small>MLOPS PREDICTION CONSOLE</small></div></div>', unsafe_allow_html=True)
    page = st.radio("Navigation", ["◉  Predict · single passenger", "◇  Batch · csv inference", "⌘  Model & Pipeline"], label_visibility="collapsed")
    cls, txt = ("online", "API Online") if health else ("offline", "API Offline")
    version = health.get("model_version", "not loaded") if health else "not loaded"
    st.markdown(f'<div class="side"><div class="{cls}"><span class="dot"></span>{txt}</div><div class="mono" style="margin:7px 0 18px 16px">{version}</div><div class="mono">PMLDL · ASSIGNMENT 1</div></div>', unsafe_allow_html=True)


def apply_preset():
    for key, value in zip(INPUT_KEYS, PRESETS[st.session_state.preset]):
        st.session_state[key] = value
        st.session_state[f"saved_{key}"] = value
    st.session_state.pop("pred", None)
    st.session_state.pop("pred_input", None)


if st.session_state.get("ui_revision") != 2:
    for key, value in zip(INPUT_KEYS, PRESETS["Custom"]):
        st.session_state[key] = value
        st.session_state[f"saved_{key}"] = value
    st.session_state.ui_revision = 2

previous_page = st.session_state.get("previous_page")
if previous_page and previous_page.startswith("◉") and not page.startswith("◉"):
    for key in INPUT_KEYS:
        if key in st.session_state:
            st.session_state[f"saved_{key}"] = st.session_state[key]
elif page.startswith("◉") and previous_page and not previous_page.startswith("◉"):
    for key, default in zip(INPUT_KEYS, PRESETS["Custom"]):
        st.session_state[key] = st.session_state.get(f"saved_{key}", default)
st.session_state.previous_page = page

if page.startswith("◉"):
    header("PMLDL · Deployment Pipeline", "Survival Prediction Console", "Explore a single-passenger prediction through the deployed FastAPI model service.", health)
    left, right = st.columns([1.03,.97], gap="large")
    with left, st.container(border=True):
        st.markdown('<div class="title">Passenger profile</div><div class="sub">Enter passenger information used by the deployed model.</div>', unsafe_allow_html=True)
        st.selectbox("Example profiles", list(PRESETS), key="preset", on_change=apply_preset)
        with st.form("passenger"):
            st.markdown('<div class="kick">Passenger class</div>', unsafe_allow_html=True)
            st.radio("Class", [1,2,3], key="pclass", horizontal=True, format_func=lambda x:f"{x}{'st' if x==1 else 'nd' if x==2 else 'rd'}", label_visibility="collapsed")
            st.markdown('<div class="kick">Sex</div>', unsafe_allow_html=True)
            st.radio("Sex", ["female","male"], key="sex", horizontal=True, format_func=str.title, label_visibility="collapsed")
            a,b = st.columns(2)
            a.number_input("Age",0.0,120.0,step=1.0,key="age",help="Age in years")
            b.number_input("Fare (£)",0.0,10000.0,step=1.0,key="fare",help="Ticket fare")
            a.number_input("Siblings / spouses",0,20,key="sibsp")
            b.number_input("Parents / children",0,20,key="parch")
            st.markdown('<div class="kick">Embarkation port</div>', unsafe_allow_html=True)
            st.radio("Port",["C","Q","S"],key="embarked",horizontal=True,format_func=lambda x:f"{x} · {PORTS[x]}",label_visibility="collapsed")
            family=st.session_state.sibsp+st.session_state.parch+1; age=st.session_state.age
            group="child" if age<=12 else "teenager" if age<=17 else "young adult" if age<=29 else "adult" if age<=59 else "senior"
            with st.expander("✦ Derived features · AUTO"):
                vals=[("Family size",family),("Is alone","Yes" if family==1 else "No"),("Fare / person",f"£{st.session_state.fare/family:.2f}"),("Age group",group),("log(Fare)",f"{np.log1p(st.session_state.fare):.2f}")]
                st.markdown('<div class="derived">'+''.join(f'<div class="ditem"><span>{k}</span><strong>{v}</strong></div>' for k,v in vals)+'</div>',unsafe_allow_html=True)
            go=st.form_submit_button("✦  Predict survival   →",type="primary",use_container_width=True)
        if go:
            payload={k:st.session_state[k] for k in INPUT_KEYS}
            try:
                r=requests.post(API_URL+"/predict",json=payload,timeout=10);r.raise_for_status();st.session_state.pred=r.json();st.session_state.pred_input=payload
                for key in INPUT_KEYS:
                    st.session_state[f"saved_{key}"] = payload[key]
            except requests.RequestException as e: st.error(f"Prediction service error: {e}")
    with right, st.container(border=True):
        version=health.get("model_version","—") if health else "—"
        st.markdown(f'<div class="title">Model prediction <span class="mono" style="float:right">{version}</span></div><div class="sub">Inference via deployed FastAPI service.</div>',unsafe_allow_html=True)
        result=st.session_state.get("pred")
        if not result:
            st.markdown('<div class="empty"><div class="sonar">◉</div><h3>Ready for prediction</h3><p>Complete the passenger profile and run inference.</p></div>',unsafe_allow_html=True)
        else:
            p=result["probability_survived"]; survived=result["predicted_class"]==1; label="Likely survived" if survived else "Likely did not survive"; badge="pos" if survived else "neg"; accent="var(--gr)" if survived else "var(--rd)"
            pills=''.join(f'<span class="pill">{k}: {v}</span>' for k,v in st.session_state.pred_input.items())
            st.markdown(f'<div class="verdict"><span class="badge {badge}">{label}</span><p style="font-size:11px">Model prediction · not ground truth</p></div><div class="gauge" style="--p:{p*100:.2f};--a:{accent}"><div><strong>{p:.0%}</strong><span>Predicted survival probability</span></div></div><div class="prow"><span>Survived</span><span class="mono">{p:.3f}</span></div><div class="bar"><i style="width:{p*100:.2f}%"></i></div><div class="prow"><span>Did not survive</span><span class="mono">{1-p:.3f}</span></div><div class="bar red"><i style="width:{(1-p)*100:.2f}%"></i></div><div class="threshold">Decision threshold · {result["threshold"]:.2f}</div><div class="pills">{pills}</div>',unsafe_allow_html=True)
        st.info("This result demonstrates the deployed MLOps pipeline.",icon="ℹ️")

elif page.startswith("◇"):
    header("Batch inference","Batch prediction","Upload a CSV file to score multiple passengers through the same deployed API.",health)
    sample=pd.DataFrame([{"Pclass":3,"Sex":"male","Age":22,"SibSp":1,"Parch":0,"Fare":7.25,"Embarked":"S"},{"Pclass":1,"Sex":"female","Age":38,"SibSp":1,"Parch":0,"Fare":71.28,"Embarked":"C"}])
    _,dl=st.columns([3,1]);dl.download_button("↓ Download sample CSV",sample.to_csv(index=False),"sample_passengers.csv","text/csv",use_container_width=True)
    with st.container(border=True):
        st.markdown('<div class="title">Passenger CSV</div><div class="sub">Drop a file below or click to browse.</div>',unsafe_allow_html=True)
        upload=st.file_uploader("CSV",type="csv",label_visibility="collapsed")
        st.markdown('<div class="note">Expected columns · Pclass · Sex · Age · SibSp · Parch · Fare · Embarked</div>',unsafe_allow_html=True)
    if upload is None:
        st.markdown('<div class="grid3">'+''.join(f'<div class="stat"><div class="sl">{x}</div><div class="sv">—</div></div>' for x in ["Rows","Valid","Invalid"])+'</div>',unsafe_allow_html=True)
    else:
        try:
            frame=pd.read_csv(upload); errors={}; missing=sorted(set(COLS)-set(frame.columns))
            for i,row in frame.iterrows():
                bad=[]
                if missing: bad.append("missing "+", ".join(missing))
                else:
                    if row.Pclass not in [1,2,3]: bad.append("Pclass")
                    if row.Sex not in ["male","female"]: bad.append("Sex")
                    if row.Embarked not in ["C","Q","S"]: bad.append("Embarked")
                    for c in ["Age","SibSp","Parch","Fare"]:
                        v=pd.to_numeric(row[c],errors="coerce")
                        if pd.isna(v) or v<0: bad.append(c)
                if bad: errors[int(i)]=bad
            valid=len(frame)-len(errors)
            values=[("Rows",len(frame),"uploaded"),("Valid",valid,"ready for scoring"),("Invalid",len(errors),"with issues")]
            st.markdown('<div class="grid3">'+''.join(f'<div class="stat"><div class="sl">{a}</div><div class="sv">{b}</div><div class="ss">{c}</div></div>' for a,b,c in values)+'</div>',unsafe_allow_html=True)
            with st.container(border=True):
                st.markdown('<div class="title">Data preview</div><div class="sub">First rows of the uploaded file.</div>',unsafe_allow_html=True);st.dataframe(frame.head(20),use_container_width=True,hide_index=True)
                if errors: st.error("Validation: "+"; ".join(f"row {i+1}: {', '.join(e)}" for i,e in list(errors.items())[:8]))
                run=st.button("✦ Run batch prediction",type="primary",use_container_width=True,disabled=bool(errors))
            if run:
                r=requests.post(API_URL+"/predict-batch",files={"file":("passengers.csv",frame.to_csv(index=False).encode(),"text/csv")},timeout=30);r.raise_for_status();pred=r.json()["predictions"];out=frame.copy();out["prediction"]=[x["predicted_class"] for x in pred];out["survival_probability"]=[x["probability_survived"] for x in pred];st.session_state.batch=out
            if "batch" in st.session_state:
                with st.container(border=True):
                    out=st.session_state.batch;st.success(f"Batch completed · {len(out)} rows predicted");st.markdown('<div class="title">Batch results</div>',unsafe_allow_html=True);st.dataframe(out,use_container_width=True,hide_index=True);st.download_button("↓ Download predictions.csv",out.to_csv(index=False),"predictions.csv","text/csv",use_container_width=True)
        except (ValueError,requests.RequestException) as e: st.error(f"Could not process batch: {e}")

else:
    header("Registry · Orchestration","Model & Pipeline","Inspect the deployed model and the automated workflow behind it.",health)
    if info:
        m=info["test_metrics"];sizes=info["dataset_sizes"]
        cards=[("API","Healthy","/health · 200 OK"),("Model","Single neuron","PyTorch · nn.Linear"),("Last training",info["trained_at"][:16].replace("T"," "),"Airflow artifact"),("Deployment","2 containers","FastAPI + Streamlit")]
        st.markdown('<div class="grid4">'+''.join(f'<div class="stat"><div class="sl">{a}</div><div class="sv">{b}</div><div class="ss">{c}</div></div>' for a,b,c in cards)+'</div>',unsafe_allow_html=True)
        st.markdown('<h2>Test metrics</h2><p>Measured on the held-out test.csv split.</p>',unsafe_allow_html=True)
        specs=[("Accuracy",m["accuracy"]),("Precision",m["precision"]),("Recall",m["recall"]),("F1",m["f1"]),("ROC-AUC",m["roc_auc"]),("Log Loss",m["log_loss"])]
        st.markdown('<div class="grid3">'+''.join(f'<div class="stat"><div class="sl">{a}</div><div class="sv">{v:.3f}</div><div class="track"><i style="width:{min(v,1)*100:.1f}%"></i></div></div>' for a,v in specs)+'</div>',unsafe_allow_html=True)
        x,y=st.columns(2)
        with x,st.container(border=True):
            rows=[("Model type","Single-neuron logistic"),("Framework","PyTorch"),("Features",len(info["feature_names"])),("Threshold","0.50"),("Version",info["model_version"])]
            st.markdown('<div class="title">Model information</div><div class="info">'+''.join(f'<div class="ir"><span>{a}</span><span>{b}</span></div>' for a,b in rows)+'</div>',unsafe_allow_html=True)
        with y,st.container(border=True): st.markdown('<div class="title">Dataset split</div>',unsafe_allow_html=True);st.metric("Training rows",sizes["train"]);st.metric("Testing rows",sizes["test"])
    else: st.warning("Model information is unavailable until the API is healthy.")
    st.markdown('<h2>Automated pipeline</h2><p>One connected DAG runs every five minutes.</p>',unsafe_allow_html=True)
    steps=["Raw Data","Cleaning","Train / Test","Feature Engineering","Training","Evaluation","FastAPI","Streamlit"]
    st.markdown('<div class="scroll"><div class="pipe">'+'<span class="arr">→</span>'.join(f'<span class="step">{s}</span>' for s in steps)+'</div></div>',unsafe_allow_html=True)
