"""
DIGIT CLASSIFIER
- Dibuja un dígito → predicción automática
- Corrige si se equivoca → el modelo aprende
- Guarda todo en corrections.csv
"""
import warnings; warnings.filterwarnings("ignore")

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image, ImageOps, ImageFilter
import os, json, pickle
from io import BytesIO

from streamlit_drawable_canvas import st_canvas
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# ─────────────────────────────────────────────
# PAGE
# ─────────────────────────────────────────────
st.set_page_config("✏️ Digit Classifier", "✏️", layout="wide")

# ─────────────────────────────────────────────
# COLORES
# ─────────────────────────────────────────────
C = {
    "bg":      "#0a0e1a",
    "card":    "#111827",
    "border":  "#1f2937",
    "green":   "#22c55e",
    "red":     "#ef4444",
    "blue":    "#3b82f6",
    "yellow":  "#f59e0b",
    "text":    "#f1f5f9",
    "muted":   "#6b7280",
}

DIGIT_CLR = ["#f87171","#fb923c","#fbbf24","#a3e635","#34d399",
             "#22d3ee","#818cf8","#c084fc","#f472b6","#94a3b8"]

plt.rcParams.update({
    "figure.facecolor": C["bg"], "axes.facecolor": C["card"],
    "axes.edgecolor": C["border"], "text.color": C["text"],
    "axes.labelcolor": C["text"], "xtick.color": C["text"],
    "ytick.color": C["text"], "grid.color": C["border"],
})

st.markdown(f"""
<style>
html,body,[class*="css"]{{background:{C['bg']};color:{C['text']};font-family:'Segoe UI',sans-serif;}}
.big-digit{{
    font-size:7rem;font-weight:900;text-align:center;line-height:1;
    text-shadow:0 0 40px currentColor;font-family:monospace;
}}
.conf-text{{font-size:1.4rem;font-weight:600;text-align:center;font-family:monospace;}}
.card{{
    background:{C['card']};border:1px solid {C['border']};
    border-radius:14px;padding:1.2rem 1.4rem;margin:.4rem 0;
}}
.label{{color:{C['muted']};font-size:.72rem;letter-spacing:2px;text-transform:uppercase;}}
.value{{font-size:1.2rem;font-weight:700;font-family:monospace;}}
.badge{{
    display:inline-block;padding:3px 12px;border-radius:20px;
    font-size:.78rem;font-weight:600;font-family:monospace;
}}
.step{{
    background:{C['card']};border-left:3px solid {C['blue']};
    border-radius:0 8px 8px 0;padding:.6rem .9rem;margin:.35rem 0;
    font-size:.88rem;
}}
.step-n{{color:{C['blue']};font-size:.7rem;font-weight:700;letter-spacing:1px;}}
h3{{color:{C['text']};border-bottom:1px solid {C['border']};padding-bottom:.4rem;}}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# ARCHIVOS PERSISTENTES
# ─────────────────────────────────────────────
CORRECTIONS_FILE = "corrections.csv"
MODEL_FILE       = "model_state.pkl"

def load_corrections():
    if os.path.exists(CORRECTIONS_FILE):
        try:
            return pd.read_csv(CORRECTIONS_FILE)
        except:
            pass
    return pd.DataFrame(columns=["pixels","true_label"])

def save_correction(vec64: np.ndarray, label: int):
    df = load_corrections()
    row = pd.DataFrame([{"pixels": json.dumps(vec64.tolist()), "true_label": int(label)}])
    df  = pd.concat([df, row], ignore_index=True)
    df.to_csv(CORRECTIONS_FILE, index=False)
    return len(df)

def corrections_to_arrays(df):
    if len(df) == 0:
        return np.empty((0, 64)), np.empty(0, dtype=int)
    X = np.array([json.loads(r) for r in df["pixels"]])
    y = df["true_label"].values.astype(int)
    return X, y

# ─────────────────────────────────────────────
# PREPROCESAMIENTO ← CLAVE PARA PRECISIÓN
# ─────────────────────────────────────────────
def canvas_to_vec64(img_data: np.ndarray) -> np.ndarray | None:
    """
    Convierte imagen RGBA del canvas (280×280) → vector 64 float
    normalizado igual que sklearn digits (0-16 / 16 = 0-1).
    
    Pasos críticos:
      1. Tomar canal de alpha/blanco como máscara del trazo
      2. Recortar bounding box del dígito (crop tight)
      3. Añadir padding proporcional (20% cada lado)
      4. Redimensionar a 8×8 con antialiasing
      5. Normalizar 0-1
    """
    if img_data is None:
        return None

    # Canal RGB promedio (el canvas dibuja blanco sobre negro)
    gray = img_data[:, :, :3].mean(axis=2).astype(np.float32)

    if gray.max() < 15:          # canvas vacío
        return None

    # ── Bounding box del trazo ───────────────────────────────
    mask = gray > 20
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return None

    r0, r1 = rows[0], rows[-1]
    c0, c1 = cols[0], cols[-1]

    # Recorte justo
    digit_crop = gray[r0:r1+1, c0:c1+1]
    h, w = digit_crop.shape

    # ── Padding proporcional (para que no esté pegado al borde) ─
    pad = max(h, w) // 4
    canvas_size = max(h, w) + 2 * pad
    padded = np.zeros((canvas_size, canvas_size), dtype=np.float32)
    off_y  = (canvas_size - h) // 2
    off_x  = (canvas_size - w) // 2
    padded[off_y:off_y+h, off_x:off_x+w] = digit_crop

    # ── Resize a 8×8 ────────────────────────────────────────
    img_pil = Image.fromarray(padded.astype(np.uint8))
    # Suavizar antes de reducir (reduce aliasing)
    img_pil = img_pil.filter(ImageFilter.GaussianBlur(radius=0.5))
    img_pil = img_pil.resize((8, 8), Image.LANCZOS)

    arr = np.array(img_pil, dtype=np.float32)
    arr = arr / arr.max() if arr.max() > 0 else arr   # normalizar 0-1
    return arr.flatten()

# ─────────────────────────────────────────────
# ENTRENAMIENTO
# ─────────────────────────────────────────────
MODELS_DEF = {
    "SVM (RBF)":           SVC(probability=True, C=10, gamma="scale"),
    "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=2000, C=1.0, solver="saga"),
    "KNN (k=5)":           KNeighborsClassifier(n_neighbors=5),
}

@st.cache_resource(show_spinner="⚙️ Entrenando modelos…")
def train_base_models():
    digits  = load_digits()
    X, y    = digits.data / 16.0, digits.target   # → rango 0-1
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    trained = {}
    for name, clf in MODELS_DEF.items():
        pipe = Pipeline([("sc", StandardScaler()),
                         ("clf", clf.__class__(**clf.get_params()))])
        pipe.fit(Xtr, ytr)
        acc  = accuracy_score(yte, pipe.predict(Xte))
        trained[name] = {"pipe": pipe, "acc": acc,
                         "Xtr": Xtr, "ytr": ytr,
                         "Xte": Xte, "yte": yte,
                         "X_all": X, "y_all": y}
    return trained, digits

trained_models, digits_ds = train_base_models()

def retrain_with_corrections(model_name: str):
    """Re-entrena el pipeline base + correcciones del usuario."""
    df_corr = load_corrections()
    if len(df_corr) == 0:
        return trained_models[model_name]["pipe"]

    Xc, yc    = corrections_to_arrays(df_corr)
    base_info = trained_models[model_name]
    X_aug = np.vstack([base_info["X_all"], Xc])
    y_aug = np.hstack([base_info["y_all"],  yc])

    clf   = MODELS_DEF[model_name].__class__(**MODELS_DEF[model_name].get_params())
    pipe  = Pipeline([("sc", StandardScaler()), ("clf", clf)])
    pipe.fit(X_aug, y_aug)
    return pipe

# ─────────────────────────────────────────────
# ESTADO DE SESIÓN
# ─────────────────────────────────────────────
if "last_vec"       not in st.session_state: st.session_state.last_vec       = None
if "last_pred"      not in st.session_state: st.session_state.last_pred      = None
if "last_probs"     not in st.session_state: st.session_state.last_probs     = None
if "correction_done" not in st.session_state: st.session_state.correction_done = False
if "retrained_pipe" not in st.session_state: st.session_state.retrained_pipe = None
if "n_corrections"  not in st.session_state:
    df = load_corrections()
    st.session_state.n_corrections = len(df)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"<div style='font-size:1.3rem;font-weight:700;color:{C['green']};margin-bottom:.5rem;'>✏️ Digit Classifier</div>", unsafe_allow_html=True)

    model_name = st.selectbox(
        "🤖 Modelo",
        list(MODELS_DEF.keys()),
        index=0,
        help="Elige el algoritmo de clasificación"
    )

    pipe = (st.session_state.retrained_pipe
            if st.session_state.retrained_pipe is not None
            else trained_models[model_name]["pipe"])

    acc_base = trained_models[model_name]["acc"]
    st.markdown(f"""
    <div class="card">
        <div class="label">Precisión base</div>
        <div class="value" style="color:{C['green']};">{acc_base*100:.1f}%</div>
    </div>
    <div class="card">
        <div class="label">Correcciones guardadas</div>
        <div class="value" style="color:{C['yellow']};">{st.session_state.n_corrections}</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    stroke_w = st.slider("✏️ Grosor del trazo", 12, 50, 25)

    st.divider()
    st.markdown("**💡 Cómo usar:**")
    for paso in [
        "1️⃣  Dibuja un dígito",
        "2️⃣  La predicción aparece sola",
        "3️⃣  Si falla, indica cuál era",
        "4️⃣  El modelo aprende y mejora",
    ]:
        st.markdown(f"<div style='font-size:.85rem;padding:.2rem 0;'>{paso}</div>", unsafe_allow_html=True)

    st.divider()
    if st.button("🔄 Re-entrenar con mis correcciones", use_container_width=True):
        with st.spinner("Re-entrenando…"):
            new_pipe = retrain_with_corrections(model_name)
            st.session_state.retrained_pipe = new_pipe
            pipe = new_pipe
        st.success("✅ Modelo actualizado con tus correcciones")

    if st.session_state.n_corrections > 0:
        df_corr = load_corrections()
        csv_bytes = df_corr.to_csv(index=False).encode()
        st.download_button(
            "💾 Descargar correcciones (.csv)",
            data=csv_bytes,
            file_name="corrections.csv",
            mime="text/csv",
            use_container_width=True,
        )

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown(f"<h1 style='font-size:2rem;margin-bottom:.1rem;'>✏️ Clasificador de Dígitos</h1>", unsafe_allow_html=True)
st.markdown(f"<p style='color:{C['muted']};margin-bottom:1.2rem;'>Dibuja un número del 0 al 9 y el modelo te dirá qué es — y aprenderá si se equivoca</p>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LAYOUT PRINCIPAL
# ─────────────────────────────────────────────
col_canvas, col_result = st.columns([1, 1], gap="large")

# ── CANVAS ────────────────────────────────────
with col_canvas:
    st.markdown("### 🖊️ Dibuja aquí")
    canvas_result = st_canvas(
        fill_color="rgba(0,0,0,0)",
        stroke_width=stroke_w,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=300,
        width=300,
        drawing_mode="freedraw",
        key="main_canvas",
        display_toolbar=True,
    )
    col_b1, col_b2 = st.columns(2)
    with col_b2:
        use_random = st.button("🎲 Ejemplo aleatorio", use_container_width=True)

# ── PREDICCIÓN ────────────────────────────────
with col_result:
    st.markdown("### 🎯 Predicción")

    # Obtener vector
    vec64 = None
    is_random = False

    if use_random:
        idx  = np.random.randint(0, len(digits_ds.data))
        vec64 = digits_ds.data[idx] / 16.0
        is_random = True
        true_random_label = int(digits_ds.target[idx])
    elif canvas_result.image_data is not None:
        vec64 = canvas_to_vec64(canvas_result.image_data)

    if vec64 is not None:
        st.session_state.last_vec = vec64
        probs = pipe.predict_proba(vec64.reshape(1, -1))[0]
        pred  = int(np.argmax(probs))
        conf  = float(probs[pred])
        st.session_state.last_pred  = pred
        st.session_state.last_probs = probs
        st.session_state.correction_done = False

        # Color según confianza
        clr = C["green"] if conf > 0.75 else C["yellow"] if conf > 0.45 else C["red"]

        # Mostrar dígito predicho grande
        st.markdown(f"""
        <div class="card" style="border-color:{clr}44;text-align:center;padding:1.5rem;">
            <div class="label" style="margin-bottom:.5rem;">EL MODELO DICE</div>
            <div class="big-digit" style="color:{DIGIT_CLR[pred]};">{pred}</div>
            <div class="conf-text" style="color:{clr};margin-top:.5rem;">{conf*100:.0f}% confianza</div>
        </div>
        """, unsafe_allow_html=True)

        if is_random:
            match = pred == true_random_label
            badge_c = C["green"] if match else C["red"]
            badge_t = f"✅ Correcto (real: {true_random_label})" if match else f"❌ Falló (real: {true_random_label})"
            st.markdown(f"<div style='text-align:center;margin-top:.4rem;'><span class='badge' style='background:{badge_c}22;color:{badge_c};border:1px solid {badge_c}66;'>{badge_t}</span></div>", unsafe_allow_html=True)

        # ── Barra de probabilidades ────────────────────────
        st.markdown(f"<div style='margin-top:1rem;margin-bottom:.3rem;color:{C['muted']};font-size:.75rem;letter-spacing:1px;'>PROBABILIDAD POR DÍGITO</div>", unsafe_allow_html=True)
        for d in range(10):
            p   = float(probs[d])
            w   = int(p * 100)
            col = DIGIT_CLR[d]
            bold = "font-weight:700;" if d == pred else ""
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:8px;margin:2px 0;">
                <div style="width:14px;font-size:.9rem;font-family:monospace;
                            color:{col};text-align:center;{bold}">{d}</div>
                <div style="flex:1;background:{C['border']};border-radius:3px;height:10px;overflow:hidden;">
                    <div style="width:{w}%;height:100%;background:{col};border-radius:3px;opacity:.85;"></div>
                </div>
                <div style="width:38px;font-size:.75rem;font-family:monospace;
                            color:{C['muted']};text-align:right;">{p*100:.0f}%</div>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.markdown(f"""
        <div class="card" style="text-align:center;padding:3rem 1rem;">
            <div style="font-size:3.5rem;margin-bottom:.8rem;">✏️</div>
            <div style="color:{C['muted']};font-size:.9rem;">
                Dibuja un número<br>para ver la predicción
            </div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SECCIÓN DE CORRECCIÓN
# ─────────────────────────────────────────────
if st.session_state.last_pred is not None and not st.session_state.correction_done:
    st.divider()
    st.markdown("### 🔧 ¿Se equivocó el modelo?")
    st.markdown(f"<p style='color:{C['muted']};font-size:.88rem;'>Si la predicción es incorrecta, selecciona cuál era el dígito real. El modelo guardará este ejemplo para aprender.</p>", unsafe_allow_html=True)

    cols_digits = st.columns(10)
    for d in range(10):
        with cols_digits[d]:
            is_pred = d == st.session_state.last_pred
            btn_style = f"background:{DIGIT_CLR[d]}33;border:2px solid {DIGIT_CLR[d]};" if is_pred else ""
            if st.button(
                str(d),
                key=f"corr_{d}",
                use_container_width=True,
                type="primary" if is_pred else "secondary",
            ):
                if st.session_state.last_vec is not None:
                    n = save_correction(st.session_state.last_vec, d)
                    st.session_state.n_corrections = n
                    st.session_state.correction_done = True
                    if d == st.session_state.last_pred:
                        st.success(f"✅ Confirmado: es un {d}. ¡Guardado!")
                    else:
                        st.success(f"📝 Corregido: era un {d}, no un {st.session_state.last_pred}. ¡Aprendido!")
                    st.rerun()

# ─────────────────────────────────────────────
# TABS DE ANÁLISIS (debajo, opcionales)
# ─────────────────────────────────────────────
st.divider()
tab_viz, tab_how, tab_stats, tab_learn = st.tabs([
    "🖼️ Ver imagen procesada",
    "📖 ¿Cómo funciona?",
    "📊 Estadísticas del modelo",
    "🧠 Mis correcciones",
])

# ── VER IMAGEN PROCESADA ─────────────────────
with tab_viz:
    if st.session_state.last_vec is not None:
        vec = st.session_state.last_vec
        pred = st.session_state.last_pred

        fig, axes = plt.subplots(1, 4, figsize=(13, 3.5))
        fig.patch.set_facecolor(C["bg"])

        # 1. Imagen 8x8
        ax = axes[0]
        ax.set_facecolor(C["card"])
        im = ax.imshow(vec.reshape(8,8), cmap="plasma", interpolation="nearest", vmin=0, vmax=1)
        ax.set_title("Tu dígito (8×8)", fontsize=9)
        ax.axis("off")
        # grilla
        for i in range(9):
            ax.axhline(i-.5, color="#000", lw=.5)
            ax.axvline(i-.5, color="#000", lw=.5)

        # 2. Valores numéricos
        ax2 = axes[1]
        ax2.set_facecolor(C["card"])
        mat = vec.reshape(8, 8)
        ax2.imshow(mat, cmap="plasma", interpolation="nearest", vmin=0, vmax=1)
        for r in range(8):
            for c in range(8):
                v = mat[r, c]
                tc = "white" if v < 0.5 else "black"
                ax2.text(c, r, f"{v:.1f}", ha="center", va="center", fontsize=5, color=tc)
        ax2.set_title("Intensidades", fontsize=9)
        ax2.axis("off")

        # 3. Promedio de clase predicha
        X_all = trained_models[model_name]["X_all"]
        y_all = trained_models[model_name]["y_all"]
        avg   = X_all[y_all == pred].mean(axis=0).reshape(8, 8)
        ax3   = axes[2]
        ax3.set_facecolor(C["card"])
        ax3.imshow(avg, cmap="plasma", interpolation="bilinear", vmin=0, vmax=1)
        ax3.set_title(f"Promedio clase {pred}", fontsize=9)
        ax3.axis("off")

        # 4. Diferencia
        ax4 = axes[3]
        ax4.set_facecolor(C["card"])
        diff = np.abs(vec.reshape(8,8) - avg)
        ax4.imshow(diff, cmap="hot", interpolation="bilinear")
        ax4.set_title("Diferencia |tu - promedio|", fontsize=9)
        ax4.axis("off")

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.markdown(f"""
        <div style="color:{C['muted']};font-size:.82rem;margin-top:.5rem;">
        <b>¿Por qué 8×8?</b> El dataset Digits de sklearn usa imágenes 8×8. 
        Tu dibujo (300×300 px) se recorta al área del trazo, se centra con padding, 
        y se reduce a 8×8 con suavizado para que los píxeles sean comparables 
        con los de entrenamiento.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Dibuja un dígito primero para ver cómo lo procesa el modelo.")

# ── CÓMO FUNCIONA ────────────────────────────
with tab_how:
    st.markdown("### Proceso completo de clasificación")

    steps = [
        ("🖊️ Dibujas", "Tu trazo se captura como imagen PNG de 300×300 píxeles en blanco sobre negro."),
        ("✂️ Recorte", "Se detecta el bounding box del dígito (área con píxeles blancos) y se recorta."),
        ("📐 Centrado", "Se añade padding (20% del tamaño) para que el dígito no quede pegado al borde."),
        ("🔬 Reducción", "La imagen se reduce a 8×8 = 64 píxeles usando antialiasing (LANCZOS) para preservar la forma."),
        ("📏 Normalización", "Los valores de pixel (0–255) se normalizan a rango 0–1. El StandardScaler del pipeline aplica media/varianza del entrenamiento."),
        ("🧠 Clasificación", f"El modelo {model_name} recibe el vector de 64 valores y calcula la probabilidad de cada dígito (0–9)."),
        ("🏆 Decisión", "Se elige el dígito con mayor probabilidad como predicción final."),
    ]

    for title, desc in steps:
        st.markdown(f"""
        <div class="step">
            <div class="step-n">{title}</div>
            <div>{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ¿Por qué a veces falla?")
    reasons = [
        ("Estilo diferente", "El modelo fue entrenado con dígitos escritos de una forma particular. Si escribes muy diferente (ej. el 7 con barra, el 1 muy inclinado), puede confundirse."),
        ("Resolución 8×8", "Al reducir tanto la imagen, detalles importantes se pierden. Un 3 y un 8 pueden verse muy parecidos a 8×8."),
        ("Posición/tamaño", "Intenta dibujar el dígito grande y centrado en el canvas para mejores resultados."),
    ]
    for title, desc in reasons:
        st.markdown(f"**{title}:** {desc}")

    st.markdown("---")
    st.markdown("### Modelos disponibles")
    for nm, info in trained_models.items():
        descs = {
            "SVM (RBF)":           "Encuentra hiperplanos de separación en espacio de alta dimensión. Muy preciso.",
            "Random Forest":       "Voto de 200 árboles de decisión. Robusto y rápido.",
            "Logistic Regression": "Modelo lineal con softmax. Simple e interpretable.",
            "KNN (k=5)":           "Busca los 5 ejemplos más similares en el dataset y vota.",
        }
        st.markdown(f"""
        <div class="card" style="margin:.3rem 0;">
            <b style="color:{C['green']};">{nm}</b>
            <span style="color:{C['muted']};font-size:.85rem;"> · {info['acc']*100:.1f}% accuracy</span><br>
            <span style="font-size:.85rem;">{descs.get(nm,'')}</span>
        </div>
        """, unsafe_allow_html=True)

# ── ESTADÍSTICAS ─────────────────────────────
with tab_stats:
    st.markdown("### Rendimiento del modelo en datos de test")

    Xte   = trained_models[model_name]["Xte"]
    yte   = trained_models[model_name]["yte"]
    ypred = trained_models[model_name]["pipe"].predict(Xte)

    # Métricas rápidas
    from sklearn.metrics import f1_score, precision_score, recall_score
    c1, c2, c3, c4 = st.columns(4)
    for col, lbl, val in zip(
        [c1, c2, c3, c4],
        ["Accuracy","F1 Macro","Precision","Recall"],
        [accuracy_score(yte,ypred),
         f1_score(yte,ypred,average="macro"),
         precision_score(yte,ypred,average="macro",zero_division=0),
         recall_score(yte,ypred,average="macro",zero_division=0)],
    ):
        col.markdown(f"""
        <div class="card" style="text-align:center;">
            <div class="label">{lbl}</div>
            <div class="value" style="color:{C['green']};">{val*100:.1f}%</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("")

    col_cm, col_f1 = st.columns([1, 1])

    with col_cm:
        st.markdown("**Matriz de confusión**")
        cm = confusion_matrix(yte, ypred)
        cm_n = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        import seaborn as sns
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.set_facecolor(C["card"])
        sns.heatmap(cm_n, annot=cm, fmt="d", ax=ax,
                    cmap="YlOrRd", linewidths=.3, linecolor=C["bg"],
                    xticklabels=range(10), yticklabels=range(10),
                    cbar_kws={"shrink":.7})
        ax.set_xlabel("Predicho"); ax.set_ylabel("Real")
        ax.set_title(model_name, fontsize=9)
        ax.tick_params(colors=C["text"], labelsize=7)
        fig.tight_layout()
        st.pyplot(fig); plt.close(fig)

    with col_f1:
        st.markdown("**F1 por dígito**")
        f1_per = f1_score(yte, ypred, average=None, zero_division=0)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.set_facecolor(C["card"])
        bars = ax.bar(range(10), f1_per, color=DIGIT_CLR, alpha=.85, edgecolor=C["bg"])
        for bar, v in zip(bars, f1_per):
            ax.text(bar.get_x()+bar.get_width()/2, v+.005,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(range(10)); ax.set_ylim(0, 1.1)
        ax.set_xlabel("Dígito"); ax.set_ylabel("F1")
        ax.set_title("F1-Score por clase", fontsize=9)
        ax.grid(axis="y", alpha=.4)
        fig.tight_layout()
        st.pyplot(fig); plt.close(fig)

# ── MIS CORRECCIONES ─────────────────────────
with tab_learn:
    df_corr = load_corrections()
    st.markdown(f"### 📚 Correcciones guardadas: **{len(df_corr)}**")

    if len(df_corr) == 0:
        st.info("Aún no has hecho ninguna corrección. Cuando el modelo falle, usa los botones de corrección y aquí aparecerán.")
    else:
        # Distribución de clases corregidas
        st.markdown("**Distribución de tus correcciones por dígito:**")
        counts = df_corr["true_label"].value_counts().sort_index()

        fig, ax = plt.subplots(figsize=(8, 3))
        ax.set_facecolor(C["card"])
        bars = ax.bar(counts.index, counts.values,
                      color=[DIGIT_CLR[i] for i in counts.index],
                      alpha=.85, edgecolor=C["bg"])
        for bar, v in zip(bars, counts.values):
            ax.text(bar.get_x()+bar.get_width()/2, v+.1,
                    str(int(v)), ha="center", va="bottom", fontsize=9)
        ax.set_xticks(range(10)); ax.set_xlabel("Dígito"); ax.set_ylabel("Ejemplos")
        ax.set_title("Mis correcciones por clase")
        fig.tight_layout()
        st.pyplot(fig); plt.close(fig)

        # Galería de las últimas correcciones
        st.markdown("**Últimas 20 correcciones:**")
        recent = df_corr.tail(20).iloc[::-1]
        cols_g = st.columns(10)
        for i, (_, row) in enumerate(recent.iterrows()):
            vec = np.array(json.loads(row["pixels"]))
            lbl = int(row["true_label"])
            col = cols_g[i % 10]
            with col:
                fig_s, ax_s = plt.subplots(figsize=(1.2, 1.2))
                fig_s.patch.set_facecolor(C["bg"])
                ax_s.set_facecolor(C["bg"])
                ax_s.imshow(vec.reshape(8, 8), cmap="plasma",
                            interpolation="nearest", vmin=0, vmax=1)
                ax_s.set_title(str(lbl), color=DIGIT_CLR[lbl],
                               fontsize=9, pad=2, fontweight="bold")
                ax_s.axis("off")
                fig_s.tight_layout(pad=0)
                st.pyplot(fig_s); plt.close(fig_s)

        st.divider()
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            if st.button("🔄 Re-entrenar ahora con mis correcciones", use_container_width=True, type="primary"):
                with st.spinner("Re-entrenando…"):
                    new_pipe = retrain_with_corrections(model_name)
                    st.session_state.retrained_pipe = new_pipe
                st.success(f"✅ Modelo actualizado con {len(df_corr)} correcciones!")
        with col_r2:
            if st.button("🗑️ Borrar todas las correcciones", use_container_width=True):
                if os.path.exists(CORRECTIONS_FILE):
                    os.remove(CORRECTIONS_FILE)
                st.session_state.n_corrections = 0
                st.session_state.retrained_pipe = None
                st.success("Correcciones borradas.")
                st.rerun()
