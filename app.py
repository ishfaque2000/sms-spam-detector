import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import re
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score, confusion_matrix)
from scipy.sparse import hstack, csr_matrix

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── GLOBAL STYLES ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;900&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

section[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
}
section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

.stApp { background: #0d1117; color: #e2e8f0; }

.metric-card {
    background: linear-gradient(135deg, #1e2a3a, #243447);
    border: 1px solid #2d4a6e;
    border-radius: 14px;
    padding: 18px 22px;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
}
.metric-card .val {
    font-size: 2rem; font-weight: 900;
    background: linear-gradient(90deg, #00d2ff, #7b2ff7);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.metric-card .lbl { font-size: 0.8rem; color: #8892a4; margin-top: 4px; }

.sec-header {
    font-size: 1.6rem; font-weight: 800;
    background: linear-gradient(90deg, #00d2ff, #7b2ff7, #ff6b6b);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 8px;
}

.pred-box {
    border-radius: 16px; padding: 24px; text-align: center;
    font-size: 2.4rem; font-weight: 900; margin: 10px 0;
    box-shadow: 0 8px 32px rgba(0,0,0,0.5);
}
.pred-spam { background: linear-gradient(135deg, #ff416c, #ff4b2b); color: white; }
.pred-ham  { background: linear-gradient(135deg, #11998e, #38ef7d); color: white; }

.grad-divider {
    height: 3px;
    background: linear-gradient(90deg, #00d2ff, #7b2ff7, #ff6b6b);
    border-radius: 2px; margin: 16px 0 24px 0;
}

.stButton > button {
    background: linear-gradient(90deg, #00d2ff, #7b2ff7);
    color: white; border: none; border-radius: 10px;
    font-weight: 700; padding: 10px 28px; font-size: 1rem;
}
.stButton > button:hover { opacity: 0.85; }

.stTabs [data-baseweb="tab"] {
    background: #1e2a3a; border-radius: 8px 8px 0 0; color: #8892a4; font-weight: 600;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #00d2ff22, #7b2ff722) !important;
    color: #00d2ff !important; border-bottom: 2px solid #00d2ff !important;
}
</style>
""", unsafe_allow_html=True)

# ─── CONSTANTS ──────────────────────────────────────────────────────────────────
STOPWORDS = {
    'the','a','an','to','of','and','in','is','it','you','i','for','me','my',
    'your','we','that','on','at','are','this','have','with','not','was','be',
    'but','they','from','or','do','so','if','up','will','just','as','he','she',
    'what','can','all','when','get','out','go','got','no','ll','ur','u','r','s'
}
SPAM_KEYWORDS = [
    'free','win','winner','prize','claim','call','text','reply','stop',
    'urgent','congratulations','selected','offer','cash','pounds','click',
    'mobile','service','subscription','awarded','guaranteed','apply','now',
    'today','limited','exclusive','ringtone','bonus','loan','credit','dear'
]
GRAD_COLORS = ['#00d2ff','#7b2ff7','#ff6b6b']
EDA_COLS    = ['char_count','word_count','exclamation','currency','digits','uppercase']

# ─── HELPERS ────────────────────────────────────────────────────────────────────
def add_features(df):
    df = df.copy()
    df['char_count']  = df['message'].apply(len)
    df['word_count']  = df['message'].apply(lambda x: len(x.split()))
    df['exclamation'] = df['message'].apply(lambda x: x.count('!'))
    df['currency']    = df['message'].apply(lambda x: x.count('$') + x.count('£'))
    df['digits']      = df['message'].apply(lambda x: sum(c.isdigit() for c in x))
    df['uppercase']   = df['message'].apply(lambda x: sum(c.isupper() for c in x))
    return df

def dark_fig(figsize=(8,4)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#111827')
    ax.tick_params(colors='#8892a4')
    ax.xaxis.label.set_color('#8892a4')
    ax.yaxis.label.set_color('#8892a4')
    ax.title.set_color('#e2e8f0')
    for sp in ax.spines.values(): sp.set_edgecolor('#2d4a6e')
    return fig, ax

# ─── DATA LOADING ───────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("spam.csv", encoding='latin-1')[['v1','v2']]
    df.columns = ['label','message']
    df = df.dropna().drop_duplicates()
    df['label_encoded'] = (df['label'] == 'spam').astype(int)
    return add_features(df)

# ─── MODEL TRAINING ─────────────────────────────────────────────────────────────
@st.cache_resource
def train_models(df):
    df_train, df_test = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['label_encoded']
    )
    tfidf  = TfidfVectorizer(max_features=1000, stop_words='english')
    scaler = MinMaxScaler()

    X_tf_tr = tfidf.fit_transform(df_train['message'])
    X_tf_te = tfidf.transform(df_test['message'])
    X_ed_tr = csr_matrix(scaler.fit_transform(df_train[EDA_COLS].values))
    X_ed_te = csr_matrix(scaler.transform(df_test[EDA_COLS].values))

    X_train = hstack([X_tf_tr, X_ed_tr])
    X_test  = hstack([X_tf_te, X_ed_te])
    y_train = df_train['label_encoded'].values
    y_test  = df_test['label_encoded'].values

    MODELS = {
        'Naive Bayes':         MultinomialNB(),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'KNN (k=5)':           KNeighborsClassifier(n_neighbors=5, metric='cosine'),
    }

    results, all_preds, trained = {}, {}, {}
    for name, model in MODELS.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        trained[name]   = model
        all_preds[name] = preds
        results[name]   = {
            'Accuracy':  round(accuracy_score(y_test, preds)*100, 2),
            'Precision': round(precision_score(y_test, preds)*100, 2),
            'Recall':    round(recall_score(y_test, preds)*100, 2),
            'F1 Score':  round(f1_score(y_test, preds)*100, 2),
        }
    return trained, tfidf, scaler, results, all_preds, y_test

# ─── SINGLE PREDICTION ──────────────────────────────────────────────────────────
def predict_message(message, model_name, trained, tfidf, scaler):
    row = {
        'char_count': len(message), 'word_count': len(message.split()),
        'exclamation': message.count('!'),
        'currency': message.count('$') + message.count('£'),
        'digits': sum(c.isdigit() for c in message),
        'uppercase': sum(c.isupper() for c in message),
    }
    X_tf  = tfidf.transform([message])
    X_ed  = csr_matrix(scaler.transform([[row[c] for c in EDA_COLS]]))
    X     = hstack([X_tf, X_ed])
    model = trained[model_name]
    pred  = model.predict(X)[0]

    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X)[0]
        conf  = proba[pred] * 100
        pd_   = {'Ham': round(proba[0]*100,1), 'Spam': round(proba[1]*100,1)}
    elif hasattr(model, 'decision_function'):
        score = float(model.decision_function(X)[0])
        conf  = min(abs(score)*10+50, 99.9)
        pd_   = {'Ham': round(100-conf,1) if pred==1 else round(conf,1),
                 'Spam': round(conf,1) if pred==1 else round(100-conf,1)}
    else:
        conf = 85.0
        pd_  = {'Ham':15.0,'Spam':85.0} if pred==1 else {'Ham':85.0,'Spam':15.0}

    feat_words = [w for w in re.findall(r'\b\w+\b', message.lower()) if w in SPAM_KEYWORDS]
    return pred, round(conf,1), pd_, feat_words, row

# ─── BATCH PREDICTION ───────────────────────────────────────────────────────────
def predict_batch(messages, model_name, trained, tfidf, scaler):
    rows = [[len(m), len(m.split()), m.count('!'), m.count('$')+m.count('£'),
             sum(c.isdigit() for c in m), sum(c.isupper() for c in m)]
            for m in messages]
    X_tf  = tfidf.transform(messages)
    X_ed  = csr_matrix(scaler.transform(np.array(rows)))
    X     = hstack([X_tf, X_ed])
    model = trained[model_name]
    preds = model.predict(X)
    if hasattr(model, 'predict_proba'):
        probas = model.predict_proba(X)
        confs  = [round(probas[i][p]*100, 1) for i,p in enumerate(preds)]
    else:
        confs = [85.0]*len(preds)
    return preds, confs

# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:10px 0 20px;'>
        <div style='font-size:3rem;'>📱</div>
        <div style='font-size:1.3rem; font-weight:800; color:#00d2ff;'>SMS Spam Detector</div>
        <div style='font-size:0.75rem; color:#8892a4; margin-top:4px;'>NLP · ML · Streamlit</div>
    </div>
    <div style='height:1px; background:linear-gradient(90deg,#00d2ff,#7b2ff7); margin-bottom:16px;'></div>
    """, unsafe_allow_html=True)

    page = st.radio("Navigate", [
        "🏠  Home",
        "📊  Dataset Explorer",
        "🔍  EDA & Visualizations",
        "🤖  Model Comparison",
        "🎯  Predict a Message",
        "📂  Upload & Predict CSV",
    ], label_visibility="collapsed")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.72rem; color:#4a5568; text-align:center; line-height:1.8;'>
        Dataset: UCI SMS Spam Collection<br>
        5,573 messages · 3 ML models<br>
        Naive Bayes · Logistic Regression · KNN
    </div>
    """, unsafe_allow_html=True)

# ─── LOAD DATA & MODELS ─────────────────────────────────────────────────────────
try:
    df = load_data()
    with st.spinner("🔧 Training models…"):
        trained, tfidf, scaler, results, all_preds, y_test = train_models(df)
    data_loaded = True
except FileNotFoundError:
    data_loaded = False

if not data_loaded:
    st.error("⚠️  `spam.csv` not found. Place it in the same directory as `app.py`.")
    st.stop()

results_df = pd.DataFrame(results).T.sort_values('F1 Score', ascending=False)

# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — HOME
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠  Home":
    st.markdown("<div class='sec-header'>📱 SMS Spam Detection</div>", unsafe_allow_html=True)
    st.markdown("<div class='grad-divider'></div>", unsafe_allow_html=True)
    st.markdown("""
    <p style='color:#b0bec5; font-size:1.05rem; line-height:1.8;'>
    Classifies SMS messages as <b style='color:#ff6b6b;'>spam</b> or
    <b style='color:#6bcb77;'>ham</b> using Natural Language Processing and
    3 Machine Learning classifiers. Explore the dataset, analyze patterns,
    compare models, and run live predictions — even on your own uploaded CSV.
    </p>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    spam_n = (df['label']=='spam').sum()
    ham_n  = (df['label']=='ham').sum()
    c1,c2,c3,c4 = st.columns(4)
    for col,val,lbl in zip([c1,c2,c3,c4],
        [f"{len(df):,}", f"{spam_n:,}", f"{ham_n:,}", f"{results_df['F1 Score'].iloc[0]:.1f}%"],
        ["Total Messages","Spam Messages","Ham Messages","Best F1 Score"]):
        col.markdown(f"<div class='metric-card'><div class='val'>{val}</div><div class='lbl'>{lbl}</div></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    steps = [
        ("📂","Load Data","5,573 SMS messages"),
        ("🔎","EDA","Patterns & distributions"),
        ("⚙️","Features","TF-IDF + 6 hand-crafted"),
        ("🤖","3 Models","NB · LR · KNN"),
        ("📈","Evaluate","Accuracy, F1, etc."),
        ("🎯","Predict","Live + CSV upload"),
    ]
    cols = st.columns(6)
    for col,(icon,title,desc) in zip(cols,steps):
        col.markdown(f"""
        <div style='background:#1e2a3a; border:1px solid #2d4a6e; border-radius:12px;
                    padding:14px 10px; text-align:center;'>
            <div style='font-size:1.8rem;'>{icon}</div>
            <div style='color:#00d2ff; font-weight:700; font-size:0.85rem; margin:6px 0 4px;'>{title}</div>
            <div style='color:#8892a4; font-size:0.72rem; line-height:1.4;'>{desc}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div style='color:#00d2ff; font-weight:700; font-size:1.1rem; margin-bottom:10px;'>🏆 Model Leaderboard</div>", unsafe_allow_html=True)
    medals = ["🥇","🥈","🥉"]
    for i,(model,row) in enumerate(results_df.iterrows()):
        st.markdown(f"""
        <div style='background:#1e2a3a; border:1px solid #2d4a6e; border-radius:10px;
                    padding:10px 16px; margin-bottom:8px; display:flex; align-items:center; gap:14px;'>
            <span style='font-size:1.3rem;'>{medals[i]}</span>
            <span style='color:#e2e8f0; font-weight:600; width:170px;'>{model}</span>
            <div style='flex:1; background:#111827; border-radius:6px; height:10px;'>
                <div style='width:{int(row["F1 Score"])}%; background:linear-gradient(90deg,#00d2ff,#7b2ff7);
                            height:10px; border-radius:6px;'></div>
            </div>
            <span style='color:#00d2ff; font-weight:700; width:55px; text-align:right;'>{row["F1 Score"]:.1f}%</span>
        </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — DATASET EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊  Dataset Explorer":
    st.markdown("<div class='sec-header'>📊 Dataset Explorer</div>", unsafe_allow_html=True)
    st.markdown("<div class='grad-divider'></div>", unsafe_allow_html=True)

    c1,c2,c3 = st.columns(3)
    c1.markdown(f"<div class='metric-card'><div class='val'>{len(df):,}</div><div class='lbl'>Total Rows</div></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='metric-card'><div class='val'>{(df['label']=='spam').sum():,}</div><div class='lbl'>Spam Messages</div></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='metric-card'><div class='val'>0</div><div class='lbl'>Null Values</div></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    f1,f2,f3 = st.columns([1,1,2])
    with f1: label_filter = st.selectbox("Label", ["All","Ham","Spam"])
    with f2:
        min_l,max_l = int(df['char_count'].min()), int(df['char_count'].max())
        len_range = st.slider("Char Length", min_l, max_l, (min_l,max_l))
    with f3: search = st.text_input("🔎 Search", placeholder="keyword…")

    fdf = df.copy()
    if label_filter != "All": fdf = fdf[fdf['label']==label_filter.lower()]
    fdf = fdf[(fdf['char_count']>=len_range[0]) & (fdf['char_count']<=len_range[1])]
    if search: fdf = fdf[fdf['message'].str.contains(search, case=False, na=False)]

    st.markdown(f"<div style='color:#8892a4; font-size:0.85rem; margin-bottom:8px;'>Showing <b style='color:#00d2ff;'>{len(fdf):,}</b> messages</div>", unsafe_allow_html=True)
    disp = ['label','message','char_count','word_count','exclamation','currency','digits','uppercase']
    st.dataframe(fdf[disp].rename(columns={
        'label':'Label','message':'Message','char_count':'Chars',
        'word_count':'Words','exclamation':'!','currency':'$£',
        'digits':'Digits','uppercase':'CAPS'
    }).reset_index(drop=True), use_container_width=True, height=430)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div style='color:#00d2ff; font-weight:700; font-size:1rem; margin-bottom:8px;'>📋 Avg Features by Label</div>", unsafe_allow_html=True)
    st.dataframe(df.groupby('label')[EDA_COLS].mean().round(2), use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 3 — EDA & VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔍  EDA & Visualizations":
    st.markdown("<div class='sec-header'>🔍 Exploratory Data Analysis</div>", unsafe_allow_html=True)
    st.markdown("<div class='grad-divider'></div>", unsafe_allow_html=True)

    tab1,tab2,tab3,tab4 = st.tabs(["📊 Class Distribution","📏 Message Length","🔤 Top Words","☁️ Word Clouds"])

    with tab1:
        spam_n=(df['label']=='spam').sum(); ham_n=(df['label']=='ham').sum()
        c1,c2 = st.columns(2)
        with c1:
            fig,ax = dark_fig((6,4))
            bars = ax.bar(['Ham','Spam'],[ham_n,spam_n], color=['#6bcb77','#ff6b6b'], width=0.45, edgecolor='none', zorder=3)
            ax.set_title('Message Count by Class', fontsize=13, pad=12)
            ax.grid(axis='y', alpha=0.2, color='#2d4a6e')
            for bar,v in zip(bars,[ham_n,spam_n]):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+20, f'{v:,}', ha='center', color='#e2e8f0', fontsize=11, fontweight='bold')
            ax.set_ylim(0, max(ham_n,spam_n)*1.15)
            st.pyplot(fig); plt.close()
        with c2:
            fig,ax = dark_fig((5,4))
            wedges,texts,autotexts = ax.pie([ham_n,spam_n], labels=['Ham','Spam'],
                colors=['#6bcb77','#ff6b6b'], autopct='%1.1f%%', startangle=90,
                wedgeprops=dict(edgecolor='#0d1117', linewidth=2))
            for t in texts+autotexts: t.set_color('#e2e8f0')
            ax.set_title('Class Proportion', fontsize=13, pad=12)
            fig.patch.set_facecolor('#0d1117')
            st.pyplot(fig); plt.close()
        st.markdown(f"<div style='background:#1e2a3a; border-left:4px solid #00d2ff; border-radius:8px; padding:12px 16px; color:#b0bec5; font-size:0.9rem;'>⚠️ <b>Class Imbalance:</b> {spam_n/(spam_n+ham_n)*100:.1f}% spam — Precision and F1 are more meaningful than raw Accuracy.</div>", unsafe_allow_html=True)

    with tab2:
        fig, axes = plt.subplots(1,2,figsize=(13,4))
        fig.patch.set_facecolor('#0d1117')
        for ax,col,xlabel in zip(axes,['char_count','word_count'],['Character Count','Word Count']):
            ax.set_facecolor('#111827')
            for sp in ax.spines.values(): sp.set_edgecolor('#2d4a6e')
            ax.tick_params(colors='#8892a4')
            ax.hist(df[df['label']=='ham'][col],  bins=40, alpha=0.75, color='#6bcb77', label='Ham')
            ax.hist(df[df['label']=='spam'][col], bins=40, alpha=0.75, color='#ff6b6b', label='Spam')
            ax.set_xlabel(xlabel, color='#8892a4'); ax.set_ylabel('Count', color='#8892a4')
            ax.set_title(f'{xlabel} Distribution', color='#e2e8f0', fontsize=12)
            ax.legend(facecolor='#1e2a3a', edgecolor='#2d4a6e', labelcolor='#e2e8f0')
            ax.grid(alpha=0.15, color='#2d4a6e')
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with tab3:
        c1,c2 = st.columns(2)
        for col,lbl,color in [(c1,'ham','#6bcb77'),(c2,'spam','#ff6b6b')]:
            with col:
                text  = ' '.join(df[df['label']==lbl]['message'].str.lower())
                words = [w for w in re.findall(r'\b[a-z]{3,}\b', text) if w not in STOPWORDS]
                freq  = Counter(words).most_common(15)
                lw    = [f[0] for f in freq][::-1]
                cnts  = [f[1] for f in freq][::-1]
                fig,ax = dark_fig((6,5))
                ax.barh(lw, cnts, color=color, alpha=0.85, edgecolor='none')
                ax.set_title(f'Top 15 — {lbl.capitalize()}', color='#e2e8f0', fontsize=12)
                ax.grid(axis='x', alpha=0.15, color='#2d4a6e')
                st.pyplot(fig); plt.close()

    with tab4:
        c1,c2 = st.columns(2)
        for col,lbl,cmap,title in [(c1,'ham','GnBu','✅ Ham'),(c2,'spam','OrRd','🚨 Spam')]:
            with col:
                st.markdown(f"<div style='text-align:center; color:#e2e8f0; font-weight:700; margin-bottom:8px;'>{title}</div>", unsafe_allow_html=True)
                text = ' '.join(df[df['label']==lbl]['message'])
                wc = WordCloud(width=600, height=360, background_color='#111827', colormap=cmap,
                               max_words=120, stopwords=STOPWORDS, collocations=False).generate(text)
                fig,ax = plt.subplots(figsize=(7,4))
                fig.patch.set_facecolor('#0d1117'); ax.imshow(wc, interpolation='bilinear'); ax.axis('off')
                st.pyplot(fig); plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 4 — MODEL COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🤖  Model Comparison":
    st.markdown("<div class='sec-header'>🤖 Model Training & Comparison</div>", unsafe_allow_html=True)
    st.markdown("<div class='grad-divider'></div>", unsafe_allow_html=True)

    best = results_df.iloc[0]
    c1,c2,c3,c4,c5 = st.columns(5)
    for col,(mname,mval) in zip([c1,c2,c3,c4,c5],[
        ('Best Model', results_df.index[0]),
        ('Accuracy',   f"{best['Accuracy']:.1f}%"),
        ('Precision',  f"{best['Precision']:.1f}%"),
        ('Recall',     f"{best['Recall']:.1f}%"),
        ('F1 Score',   f"{best['F1 Score']:.1f}%"),
    ]):
        col.markdown(f"<div class='metric-card'><div class='val' style='font-size:1.3rem;'>{mval}</div><div class='lbl'>{mname}</div></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    tab1,tab2,tab3 = st.tabs(["📊 Bar Comparison","📋 Results Table","🔢 Confusion Matrices"])

    with tab1:
        metrics = ['Accuracy','Precision','Recall','F1 Score']
        models  = results_df.index.tolist()
        x = np.arange(len(models)); w = 0.18
        fig,ax = dark_fig((11,5))
        for i,(metric,color) in enumerate(zip(metrics,['#00d2ff','#7b2ff7','#ff6b6b','#ffd93d'])):
            ax.bar(x+i*w-1.5*w, [results[m][metric] for m in models], w,
                   label=metric, color=color, alpha=0.85, edgecolor='none', zorder=3)
        ax.set_xticks(x); ax.set_xticklabels(models, rotation=10, ha='right', color='#8892a4', fontsize=11)
        ax.set_ylim(0,115); ax.set_ylabel('Score (%)', color='#8892a4')
        ax.set_title('Model Performance Comparison', color='#e2e8f0', fontsize=13)
        ax.legend(facecolor='#1e2a3a', edgecolor='#2d4a6e', labelcolor='#e2e8f0', fontsize=9)
        ax.grid(axis='y', alpha=0.15, color='#2d4a6e', zorder=0)
        st.pyplot(fig); plt.close()

        fig,ax = dark_fig((11,4))
        for i,model in enumerate(models):
            vals = [results[model][m] for m in metrics]
            ax.plot(metrics, vals, marker='o', linewidth=2.5, color=GRAD_COLORS[i], label=model, markersize=7, zorder=3)
            ax.fill_between(metrics, vals, alpha=0.07, color=GRAD_COLORS[i])
        ax.set_ylim(70,105); ax.set_title('Metric Profiles', color='#e2e8f0', fontsize=13)
        ax.legend(facecolor='#1e2a3a', edgecolor='#2d4a6e', labelcolor='#e2e8f0', fontsize=9)
        ax.grid(alpha=0.2, color='#2d4a6e')
        st.pyplot(fig); plt.close()

    with tab2:
        st.dataframe(results_df.style.background_gradient(cmap='Blues', axis=0).format("{:.2f}"), use_container_width=True)

    with tab3:
        fig, axes = plt.subplots(1,3, figsize=(15,5))
        fig.patch.set_facecolor('#0d1117')
        for i,(name,preds) in enumerate(all_preds.items()):
            ax = axes[i]
            ax.set_facecolor('#111827')
            for sp in ax.spines.values(): sp.set_edgecolor('#2d4a6e')
            cm = confusion_matrix(y_test, preds)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                        xticklabels=['Ham','Spam'], yticklabels=['Ham','Spam'],
                        linewidths=0.5, linecolor='#0d1117')
            ax.set_title(name, color='#e2e8f0', fontsize=11, pad=8)
            ax.tick_params(colors='#8892a4')
            ax.set_xlabel('Predicted', color='#8892a4'); ax.set_ylabel('Actual', color='#8892a4')
        plt.tight_layout(pad=2); st.pyplot(fig); plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 5 — PREDICT A MESSAGE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🎯  Predict a Message":
    st.markdown("<div class='sec-header'>🎯 Live Spam Prediction</div>", unsafe_allow_html=True)
    st.markdown("<div class='grad-divider'></div>", unsafe_allow_html=True)

    # Init session state
    if 'predict_msg' not in st.session_state:
        st.session_state.predict_msg = ''
    if 'msg_key' not in st.session_state:
        st.session_state.msg_key = 0

    col_l, col_r = st.columns([1.6,1])
    with col_l:
        model_choice = st.selectbox("🤖 Choose Classifier", list(trained.keys()),
                                    index=list(trained.keys()).index(results_df.index[0]))

        s1,s2,s3 = st.columns(3)
        if s1.button("🚨 Spam Sample", use_container_width=True):
            sample = df[df['label']=='spam']['message'].sample(1).values[0]
            st.session_state.predict_msg = sample
            st.session_state.msg_key += 1
        if s2.button("✅ Ham Sample", use_container_width=True):
            sample = df[df['label']=='ham']['message'].sample(1).values[0]
            st.session_state.predict_msg = sample
            st.session_state.msg_key += 1
        if s3.button("🤔 Tricky", use_container_width=True):
            tricky_keywords = ['free','prize','winner','congratulations','awarded','claim','offer']
            tricky_pool = df[df['label']=='spam']['message']
            tricky_pool = tricky_pool[tricky_pool.str.lower().str.contains('|'.join(tricky_keywords[:3]))]
            st.session_state.predict_msg = tricky_pool.sample(1).values[0]
            st.session_state.msg_key += 1

        user_msg = st.text_area("✉️ Enter SMS Message",
            value=st.session_state.predict_msg,
            height=130,
            placeholder="Type or paste a message… or use a sample button above!",
            key=f"msg_input_{st.session_state.msg_key}")

        predict_btn = st.button("🔍 Predict", use_container_width=True)

    with col_r:
        st.markdown("""
        <div style='background:#1e2a3a; border:1px solid #2d4a6e; border-radius:12px;
                    padding:16px; font-size:0.82rem; color:#8892a4; line-height:1.9;'>
        <b style='color:#00d2ff;'>ℹ️ How it works</b><br>
        1. Message → TF-IDF (1,000 features)<br>
        2. 6 EDA features extracted (!, $, digits…)<br>
        3. Selected model predicts ham/spam<br>
        4. Confidence from decision/proba score<br>
        5. Spam keywords are highlighted
        </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style='background:#1e2a3a; border:1px solid #2d4a6e; border-radius:12px; padding:14px;'>
        <div style='color:#00d2ff; font-weight:700; font-size:0.9rem; margin-bottom:8px;'>🏆 Rankings</div>
        {"".join([f'<div style="color:#8892a4; font-size:0.8rem; padding:2px 0;">{i+1}. <b style="color:#e2e8f0;">{m}</b> — F1: <span style="color:#00d2ff;">{results_df.loc[m,"F1 Score"]:.1f}%</span></div>' for i,m in enumerate(results_df.index)])}
        </div>""", unsafe_allow_html=True)

    if predict_btn and user_msg.strip():
        pred, conf, pd_, feat_words, row = predict_message(user_msg, model_choice, trained, tfidf, scaler)
        label = "🚨 SPAM" if pred==1 else "✅ HAM"
        css   = 'pred-spam' if pred==1 else 'pred-ham'

        st.markdown("<div class='grad-divider'></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='pred-box {css}'>{label}</div>", unsafe_allow_html=True)

        r1,r2,r3 = st.columns(3)
        r1.markdown(f"<div class='metric-card'><div class='val'>{conf:.1f}%</div><div class='lbl'>Confidence</div></div>", unsafe_allow_html=True)
        r2.markdown(f"<div class='metric-card'><div class='val'>{row['char_count']}</div><div class='lbl'>Characters</div></div>", unsafe_allow_html=True)
        r3.markdown(f"<div class='metric-card'><div class='val'>{row['exclamation']}</div><div class='lbl'>Exclamations</div></div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        pc1,pc2 = st.columns(2)

        with pc1:
            st.markdown("<div style='color:#00d2ff; font-weight:700; font-size:1rem; margin-bottom:8px;'>📊 Probability Distribution</div>", unsafe_allow_html=True)
            fig,ax = dark_fig((5,2.8))
            cats = list(pd_.keys()); vals = list(pd_.values())
            bars = ax.barh(cats, vals, color=['#6bcb77','#ff6b6b'], alpha=0.85, edgecolor='none', height=0.4)
            ax.set_xlim(0,115); ax.set_xlabel('Probability (%)', color='#8892a4')
            ax.grid(axis='x', alpha=0.15, color='#2d4a6e')
            for bar,v in zip(bars,vals):
                ax.text(v+1.5, bar.get_y()+bar.get_height()/2, f'{v:.1f}%',
                        va='center', color='#e2e8f0', fontweight='bold', fontsize=11)
            st.pyplot(fig); plt.close()

        with pc2:
            st.markdown("<div style='color:#00d2ff; font-weight:700; font-size:1rem; margin-bottom:8px;'>⚙️ Extracted Features</div>", unsafe_allow_html=True)
            max_val = max(df[EDA_COLS].max().max(), 1)
            for fname,fval in [('Characters',row['char_count']),('Words',row['word_count']),
                               ('Exclamations (!)',row['exclamation']),('Currency ($£)',row['currency']),
                               ('Digits',row['digits']),('Uppercase',row['uppercase'])]:
                bar_w = min(int(fval/max_val*100), 100)
                st.markdown(f"""
                <div style='display:flex; align-items:center; gap:10px; margin-bottom:6px;'>
                    <span style='color:#8892a4; font-size:0.8rem; width:130px;'>{fname}</span>
                    <div style='flex:1; background:#111827; border-radius:4px; height:8px;'>
                        <div style='width:{bar_w}%; background:linear-gradient(90deg,#00d2ff,#7b2ff7); height:8px; border-radius:4px;'></div>
                    </div>
                    <span style='color:#00d2ff; font-size:0.8rem; width:30px; text-align:right;'>{fval}</span>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div style='color:#00d2ff; font-weight:700; font-size:1rem; margin-bottom:8px;'>🔑 Spam Keywords Found</div>", unsafe_allow_html=True)
        if feat_words:
            html_w = " ".join([f"<span style='display:inline-block; background:rgba(255,107,107,0.25); border:1px solid #ff6b6b; border-radius:6px; padding:2px 8px; margin:3px; font-size:0.85rem; color:#ff9a9a;'>{w}</span>" for w in set(feat_words)])
            st.markdown(f"<div style='line-height:2.2;'>{html_w}</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='color:#8892a4; font-size:0.9rem;'>✅ No common spam keywords detected.</div>", unsafe_allow_html=True)

        highlighted = user_msg
        for w in set(feat_words):
            highlighted = re.sub(rf'\b({re.escape(w)})\b',
                r"<mark style='background:#ff6b6b33; color:#ff9a9a; border-radius:3px; padding:1px 4px;'>\1</mark>",
                highlighted, flags=re.IGNORECASE)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div style='color:#00d2ff; font-weight:700; font-size:1rem; margin-bottom:8px;'>📝 Message Highlights</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='background:#1e2a3a; border:1px solid #2d4a6e; border-radius:10px; padding:16px; color:#e2e8f0; line-height:1.8;'>{highlighted}</div>", unsafe_allow_html=True)

    elif predict_btn:
        st.warning("⚠️ Please enter a message before predicting.")

# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 6 — UPLOAD & PREDICT CSV
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📂  Upload & Predict CSV":
    st.markdown("<div class='sec-header'>📂 Upload & Predict CSV</div>", unsafe_allow_html=True)
    st.markdown("<div class='grad-divider'></div>", unsafe_allow_html=True)

    st.markdown("""
    <div style='background:#1e2a3a; border:1px solid #2d4a6e; border-radius:12px;
                padding:16px 20px; margin-bottom:20px; color:#b0bec5; font-size:0.9rem; line-height:1.8;'>
    📋 <b style='color:#00d2ff;'>How to use:</b> Upload a <b>.csv</b> file with a column containing
    SMS messages. Select the message column, choose a model, and get instant spam/ham predictions
    for every row. Results are downloadable as a new CSV.
    </div>
    """, unsafe_allow_html=True)

    up_col, cfg_col = st.columns([1.4, 1])
    with up_col:
        uploaded_file = st.file_uploader("📤 Upload your CSV file", type=["csv"],
            help="CSV must have at least one text column with SMS messages")
    with cfg_col:
        st.markdown("<div style='color:#00d2ff; font-weight:700; font-size:0.95rem; margin-bottom:10px;'>⚙️ Configuration</div>", unsafe_allow_html=True)
        up_model      = st.selectbox("🤖 Model", list(trained.keys()),
                                     index=list(trained.keys()).index(results_df.index[0]), key="up_model")
        show_features = st.checkbox("Show extracted features", value=True)
        show_keywords = st.checkbox("Show spam keywords", value=True)

    if uploaded_file is not None:
        try:
            try:
                up_df = pd.read_csv(uploaded_file, encoding='utf-8')
            except UnicodeDecodeError:
                uploaded_file.seek(0)
                up_df = pd.read_csv(uploaded_file, encoding='latin-1')

            st.markdown("<div class='grad-divider'></div>", unsafe_allow_html=True)

            text_cols = [c for c in up_df.columns if up_df[c].dtype == object]
            if not text_cols:
                st.error("⚠️ No text columns found in the uploaded CSV.")
                st.stop()

            col_sel, prev_col = st.columns([1,2])
            with col_sel:
                msg_col = st.selectbox("📌 Select message column", text_cols)
                st.markdown(f"<div style='color:#8892a4; font-size:0.82rem; margin-top:6px;'>{len(up_df):,} rows · {len(up_df.columns)} columns</div>", unsafe_allow_html=True)
            with prev_col:
                st.markdown("<div style='color:#00d2ff; font-weight:600; font-size:0.85rem; margin-bottom:6px;'>Preview</div>", unsafe_allow_html=True)
                st.dataframe(up_df[[msg_col]].head(3), use_container_width=True)

            st.markdown("<br>", unsafe_allow_html=True)
            run_btn = st.button("🚀 Run Predictions", use_container_width=True)

            if run_btn:
                messages = up_df[msg_col].fillna('').astype(str).tolist()
                with st.spinner(f"🔍 Predicting {len(messages):,} messages with {up_model}…"):
                    preds, confs = predict_batch(messages, up_model, trained, tfidf, scaler)

                result_df = up_df.copy()
                result_df['Prediction']  = ['SPAM' if p==1 else 'HAM' for p in preds]
                result_df['Confidence%'] = confs

                if show_features:
                    feat_rows = [[len(m), len(m.split()), m.count('!'),
                                  m.count('$')+m.count('£'),
                                  sum(c.isdigit() for c in m), sum(c.isupper() for c in m)]
                                 for m in messages]
                    feat_arr = np.array(feat_rows)
                    for j,fname in enumerate(['Chars','Words','Exclaim','Currency','Digits','Uppercase']):
                        result_df[fname] = feat_arr[:,j].astype(int)

                if show_keywords:
                    result_df['SpamKeywords'] = [
                        ', '.join(set(w for w in re.findall(r'\b\w+\b', m.lower()) if w in SPAM_KEYWORDS)) or '—'
                        for m in messages
                    ]

                spam_count = (result_df['Prediction']=='SPAM').sum()
                ham_count  = (result_df['Prediction']=='HAM').sum()
                avg_conf   = result_df['Confidence%'].mean()

                st.markdown("<br>", unsafe_allow_html=True)
                s1,s2,s3,s4 = st.columns(4)
                for col,val,lbl in zip([s1,s2,s3,s4],
                    [f"{len(result_df):,}", f"{spam_count:,}", f"{ham_count:,}", f"{avg_conf:.1f}%"],
                    ["Total Msgs","Spam Detected","Ham Detected","Avg Confidence"]):
                    col.markdown(f"<div class='metric-card'><div class='val'>{val}</div><div class='lbl'>{lbl}</div></div>", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                ch1,ch2 = st.columns(2)
                with ch1:
                    fig,ax = dark_fig((5,3))
                    ax.bar(['Ham','Spam'],[ham_count,spam_count], color=['#6bcb77','#ff6b6b'],
                           width=0.45, edgecolor='none', zorder=3)
                    for i,(v,_) in enumerate(zip([ham_count,spam_count],['Ham','Spam'])):
                        ax.text(i, v+0.3, str(v), ha='center', color='#e2e8f0', fontweight='bold', fontsize=12)
                    ax.set_title('Prediction Distribution', color='#e2e8f0', fontsize=11)
                    ax.grid(axis='y', alpha=0.2, color='#2d4a6e')
                    ax.set_ylim(0, max(ham_count,spam_count)*1.15)
                    st.pyplot(fig); plt.close()

                with ch2:
                    fig,ax = dark_fig((5,3))
                    ax.hist(result_df['Confidence%'], bins=20, color='#7b2ff7', alpha=0.85, edgecolor='none')
                    ax.set_title('Confidence Distribution', color='#e2e8f0', fontsize=11)
                    ax.set_xlabel('Confidence (%)', color='#8892a4')
                    ax.set_ylabel('Count', color='#8892a4')
                    ax.grid(alpha=0.15, color='#2d4a6e')
                    st.pyplot(fig); plt.close()

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("<div style='color:#00d2ff; font-weight:700; font-size:1rem; margin-bottom:8px;'>📋 Prediction Results</div>", unsafe_allow_html=True)

                def color_pred(val):
                    if val == 'SPAM': return 'background-color: rgba(255,107,107,0.2); color:#ff9a9a; font-weight:bold;'
                    if val == 'HAM':  return 'background-color: rgba(107,203,119,0.2); color:#6bcb77; font-weight:bold;'
                    return ''

                st.dataframe(result_df.style.applymap(color_pred, subset=['Prediction']),
                             use_container_width=True, height=400)

                st.markdown("<br>", unsafe_allow_html=True)
                csv_bytes = result_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="⬇️  Download Results as CSV",
                    data=csv_bytes,
                    file_name="spam_predictions.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        except Exception as e:
            st.error(f"❌ Error reading file: {e}")

    else:
        st.markdown("""
        <div style='background:#1e2a3a; border:2px dashed #2d4a6e; border-radius:14px;
                    padding:50px; text-align:center;'>
            <div style='font-size:3.5rem; margin-bottom:12px;'>📤</div>
            <div style='color:#00d2ff; font-weight:700; font-size:1.1rem; margin-bottom:8px;'>No file uploaded yet</div>
            <div style='color:#8892a4; font-size:0.88rem; line-height:1.7;'>
                Upload a CSV file above to get started.<br>
                Your file needs a text column with SMS messages.
            </div>
        </div>
        <br>
        <div style='color:#00d2ff; font-weight:700; font-size:0.95rem; margin-bottom:10px;'>📝 Expected CSV format:</div>
        <div style='background:#111827; border:1px solid #2d4a6e; border-radius:10px;
                    padding:14px; font-family:monospace; font-size:0.82rem; color:#8892a4; line-height:1.8;'>
            message<br>
            "Hello, how are you?"<br>
            "WINNER!! Call 09061701461 to claim your prize!"<br>
            "Are you free for lunch tomorrow?"<br>
            "Free entry! Text WIN to 80082 now."
        </div>
        """, unsafe_allow_html=True)