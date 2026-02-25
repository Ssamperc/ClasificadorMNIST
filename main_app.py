"""
╔══════════════════════════════════════════════════════════════╗
║        MNIST DIGIT CLASSIFIER — Clasificador Interactivo     ║
║  Dibuja un dígito → el modelo predice y explica cómo lo hizo ║
╚══════════════════════════════════════════════════════════════╝
"""
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from PIL import Image, ImageOps, ImageFilter
import io

from streamlit_drawable_canvas import st_canvas

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score, confusion_matrix

# ──────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MNIST · Clasificador de Dígitos",
    page_icon="✏️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────
# TEMA VISUAL — estética "terminal retro-futurista"
# ──────────────────────────────────────────────────────────────
BG      = "#050810"
SURF    = "#0d1117"
CARD    = "#111827"
BORDER  = "#1f2937"
ACCENT  = "#00FF88"       # verde neón
ACCENT2 = "#FF6B35"       # naranja
ACCENT3 = "#00BFFF"       # azul hielo
WARN    = "#FFD700"
TEXT    = "#e2e8f0"
MUTED   = "#4b5563"
GRID    = "#1f2937"

DIGIT_PALETTE = [
    "#FF6B6B","#FF9F43","#FECA57","#48CAE4","#00BFFF",
    "#6C63FF","#A29BFE","#FD79A8","#00FF88","#74B9FF",
]

plt.rcParams.update({
    "text.color": TEXT, "axes.labelcolor": TEXT,
    "xtick.color": TEXT, "ytick.color": TEXT,
    "figure.facecolor": BG, "axes.facecolor": SURF,
    "axes.edgecolor": BORDER, "grid.color": GRID,
    "legend.facecolor": CARD, "legend.edgecolor": BORDER,
    "legend.labelcolor": TEXT, "font.family": "monospace",
})

def dax(ax):
    ax.set_facecolor(SURF)
    ax.tick_params(colors=TEXT, labelsize=8)
    ax.xaxis.label.set_color(TEXT); ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    for sp in ax.spines.values(): sp.set_edgecolor(BORDER)
    ax.grid(color=GRID, linewidth=0.4, alpha=0.6)
    return ax

# ──────────────────────────────────────────────────────────────
# CSS
# ──────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Exo+2:wght@300;600;800&display=swap');

html,body,[class*="css"]{{
    background-color:{BG};
    font-family:'Exo 2',sans-serif;
}}
.mono{{font-family:'Share Tech Mono',monospace;}}

/* Título */
.hero-title{{
    font-family:'Share Tech Mono',monospace;
    font-size:2.4rem; font-weight:700; letter-spacing:3px;
    color:{ACCENT}; text-shadow: 0 0 30px {ACCENT}88;
    margin-bottom:.2rem;
}}
.hero-sub{{color:{MUTED};font-size:.9rem;letter-spacing:2px;margin-bottom:1.5rem;}}

/* Sección */
.sec{{
    font-family:'Share Tech Mono',monospace;
    font-size:.72rem;letter-spacing:3px;text-transform:uppercase;
    color:{ACCENT3};border-left:3px solid {ACCENT3};
    padding-left:.6rem;margin:1.4rem 0 .7rem;
}}

/* Tarjeta de predicción */
.pred-card{{
    background:linear-gradient(135deg,{CARD},{SURF});
    border:1px solid {ACCENT}44;
    border-radius:16px;padding:1.5rem;text-align:center;
    box-shadow: 0 0 40px {ACCENT}22;
}}
.pred-digit{{
    font-family:'Share Tech Mono',monospace;
    font-size:6rem;font-weight:700;
    color:{ACCENT};text-shadow:0 0 60px {ACCENT};
    line-height:1;
}}
.pred-label{{color:{MUTED};font-size:.8rem;letter-spacing:2px;margin-top:.4rem;}}
.pred-conf{{
    font-family:'Share Tech Mono',monospace;
    font-size:1.5rem;color:{ACCENT2};margin-top:.5rem;
}}

/* Info boxes */
.info-box{{
    background:{CARD};border:1px solid {BORDER};
    border-radius:10px;padding:.8rem 1rem;
    margin:.3rem 0;
}}
.ib-label{{color:{MUTED};font-size:.7rem;letter-spacing:2px;text-transform:uppercase;}}
.ib-val{{color:{TEXT};font-size:1.1rem;font-weight:600;font-family:'Share Tech Mono',monospace;}}

/* Barra de confianza */
.conf-bar-wrap{{background:{BORDER};border-radius:4px;height:10px;margin:4px 0;}}
.conf-bar{{height:10px;border-radius:4px;transition:width .4s ease;}}

/* Pasos del proceso */
.step{{
    background:{CARD};border-left:3px solid {ACCENT3};
    border-radius:0 10px 10px 0;padding:.7rem 1rem;margin:.5rem 0;
}}
.step-num{{color:{ACCENT3};font-family:'Share Tech Mono';font-size:.75rem;letter-spacing:2px;}}
.step-text{{color:{TEXT};font-size:.9rem;margin-top:.15rem;}}

/* Modelo tag */
.model-tag{{
    display:inline-block;background:{ACCENT}22;border:1px solid {ACCENT}66;
    color:{ACCENT};border-radius:6px;padding:2px 10px;
    font-family:'Share Tech Mono';font-size:.75rem;letter-spacing:1px;
}}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# MODELOS DISPONIBLES
# ──────────────────────────────────────────────────────────────
AVAILABLE_MODELS = {
    "Random Forest": {
        "clf": RandomForestClassifier(n_estimators=200, random_state=42),
        "desc": "Ensemble de 200 árboles de decisión. Robusto, sin necesidad de escalar.",
        "icon": "🌲",
    },
    "SVM (RBF)": {
        "clf": SVC(probability=True, kernel="rbf", C=10, gamma="scale"),
        "desc": "Support Vector Machine con kernel RBF. Excelente para espacios de alta dimensión.",
        "icon": "⚡",
    },
    "Logistic Regression": {
        "clf": LogisticRegression(max_iter=2000, C=0.1, solver="saga"),
        "desc": "Modelo lineal con regularización L2. Rápido e interpretable.",
        "icon": "📈",
    },
    "K-Nearest Neighbors": {
        "clf": KNeighborsClassifier(n_neighbors=5, metric="euclidean"),
        "desc": "Clasifica por los 5 vecinos más cercanos en el espacio de píxeles.",
        "icon": "🎯",
    },
    "Gradient Boosting": {
        "clf": GradientBoostingClassifier(n_estimators=150, learning_rate=0.1),
        "desc": "Boosting iterativo que corrige errores del modelo anterior.",
        "icon": "🚀",
    },
    "Naive Bayes": {
        "clf": GaussianNB(),
        "desc": "Probabilístico. Asume independencia entre píxeles.",
        "icon": "🎲",
    },
}

# ──────────────────────────────────────────────────────────────
# ENTRENAMIENTO (cacheado)
# ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🔧 Entrenando modelos sobre Digits dataset…")
def train_models():
    digits = load_digits()
    X, y   = digits.data / 16.0, digits.target   # normalizar 0-1

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    trained = {}
    for name, info in AVAILABLE_MODELS.items():
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    info["clf"].__class__(**info["clf"].get_params())),
        ])
        pipe.fit(X_tr, y_tr)
        acc  = accuracy_score(y_te, pipe.predict(X_te))
        trained[name] = {
            "pipe":  pipe,
            "acc":   acc,
            "desc":  info["desc"],
            "icon":  info["icon"],
            "X_te":  X_te,
            "y_te":  y_te,
            "X_tr":  X_tr,
            "y_tr":  y_tr,
            "X_all": X,
            "y_all": y,
        }
    return trained, digits

trained_models, digits_ds = train_models()

# ──────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"<div style='color:{ACCENT};font-family:Share Tech Mono;font-size:1.1rem;letter-spacing:2px;'>⚙ CONTROL PANEL</div>", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("**🤖 Modelo de clasificación**")
    model_name = st.selectbox(
        "Selecciona modelo",
        list(AVAILABLE_MODELS.keys()),
        index=0,
        label_visibility="collapsed",
    )

    info = trained_models[model_name]
    st.markdown(f"""
    <div class="info-box">
        <div class="ib-label">Accuracy en test</div>
        <div class="ib-val">{info['acc']*100:.2f}%</div>
    </div>
    <div class="info-box">
        <div class="ib-label">Descripción</div>
        <div style="color:{TEXT};font-size:.82rem;margin-top:.2rem;">{info['desc']}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**✏️ Herramienta de dibujo**")
    stroke_width = st.slider("Grosor del trazo", 10, 40, 22)
    canvas_bg    = "#000000"

    st.markdown("---")
    st.markdown("**🔬 Explicabilidad**")
    show_neighbors    = st.checkbox("Mostrar vecinos similares", True)
    show_pixel_imp    = st.checkbox("Mapa de importancia (pixels)", True)
    show_topk         = st.number_input("Top-K probabilidades a mostrar", 3, 10, 5)
    show_all_probs    = st.checkbox("Barra completa de probabilidades", True)

    st.markdown("---")
    st.markdown("**📦 Dataset**")
    st.markdown(f"""
    <div class="info-box">
        <div class="ib-label">Muestras totales</div>
        <div class="ib-val">1 797</div>
    </div>
    <div class="info-box">
        <div class="ib-label">Resolución imagen</div>
        <div class="ib-val">8 × 8 px</div>
    </div>
    <div class="info-box">
        <div class="ib-label">Clases</div>
        <div class="ib-val">10 (0 – 9)</div>
    </div>
    """, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────────────────────
st.markdown(f'<div class="hero-title">✏ DIGIT.AI</div>', unsafe_allow_html=True)
st.markdown(f'<div class="hero-sub">CLASIFICADOR INTERACTIVO DE DÍGITOS MANUSCRITOS · MNIST/DIGITS</div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# LAYOUT PRINCIPAL: Canvas ← | → Predicción
# ──────────────────────────────────────────────────────────────
col_draw, col_pred = st.columns([1, 1], gap="large")

# ── CANVAS ────────────────────────────────────────────────────
with col_draw:
    st.markdown('<div class="sec">// DIBUJA UN DÍGITO AQUÍ</div>', unsafe_allow_html=True)
    st.markdown(f"<div style='color:{MUTED};font-size:.8rem;margin-bottom:.5rem;'>Dibuja un número del 0 al 9 con el mouse o dedo</div>", unsafe_allow_html=True)

    canvas_result = st_canvas(
        fill_color="rgba(0,0,0,0)",
        stroke_width=stroke_width,
        stroke_color="#FFFFFF",
        background_color=canvas_bg,
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
        display_toolbar=True,
    )

    col_b1, col_b2 = st.columns(2)
    with col_b1:
        if st.button("🗑️ Limpiar canvas", use_container_width=True):
            st.rerun()
    with col_b2:
        use_sample = st.button("🎲 Dígito aleatorio", use_container_width=True)

# ──────────────────────────────────────────────────────────────
# PROCESAR IMAGEN
# ──────────────────────────────────────────────────────────────
def preprocess_canvas(img_data: np.ndarray) -> np.ndarray | None:
    """Convierte la imagen del canvas (280×280 RGBA) → vector 64 normalizado."""
    if img_data is None:
        return None
    # Canal alfa como máscara de trazos
    img_gray = img_data[:, :, 3].astype(np.float32)
    if img_gray.max() < 10:
        return None                          # canvas vacío
    img_pil = Image.fromarray(img_gray.astype(np.uint8))
    img_pil = img_pil.resize((8, 8), Image.LANCZOS)
    img_arr = np.array(img_pil, dtype=np.float32)
    img_arr = img_arr / img_arr.max() * 16.0 if img_arr.max() > 0 else img_arr
    return img_arr.flatten() / 16.0          # normalizado 0-1, 64 features

def preprocess_sample(x_raw: np.ndarray) -> np.ndarray:
    """Dígito del dataset ya normalizado (0-1), lo devuelve tal cual."""
    return x_raw / 16.0 if x_raw.max() > 1 else x_raw

# ──────────────────────────────────────────────────────────────
# OBTENER VECTOR DE ENTRADA
# ──────────────────────────────────────────────────────────────
sample_digit   = None
sample_label   = None
input_vector   = None
input_source   = None

if use_sample:
    idx = np.random.randint(0, len(digits_ds.data))
    sample_digit = digits_ds.data[idx] / 16.0
    sample_label = int(digits_ds.target[idx])
    input_vector = sample_digit
    input_source = "sample"
elif canvas_result.image_data is not None:
    vec = preprocess_canvas(canvas_result.image_data)
    if vec is not None:
        input_vector = vec
        input_source = "canvas"

# ──────────────────────────────────────────────────────────────
# PREDICCIÓN Y VISUALIZACIÓN
# ──────────────────────────────────────────────────────────────
with col_pred:
    st.markdown('<div class="sec">// RESULTADO DE CLASIFICACIÓN</div>', unsafe_allow_html=True)

    if input_vector is None:
        st.markdown(f"""
        <div style="background:{CARD};border:1px dashed {BORDER};border-radius:16px;
                    padding:3rem;text-align:center;margin-top:.5rem;">
            <div style="font-size:3rem;margin-bottom:1rem;">✏️</div>
            <div style="color:{MUTED};font-family:'Share Tech Mono';font-size:.85rem;letter-spacing:2px;">
                DIBUJA UN DÍGITO<br>PARA COMENZAR
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        pipe  = trained_models[model_name]["pipe"]
        probs = pipe.predict_proba(input_vector.reshape(1, -1))[0]
        pred  = int(np.argmax(probs))
        conf  = float(probs[pred])
        top_k_idx = np.argsort(probs)[::-1][:int(show_topk)]

        # Colorear confianza
        if conf >= 0.85:   conf_color = ACCENT
        elif conf >= 0.55: conf_color = WARN
        else:              conf_color = ACCENT2

        # Tarjeta principal
        st.markdown(f"""
        <div class="pred-card">
            <div class="pred-label">EL MODELO PREDICE</div>
            <div class="pred-digit" style="color:{DIGIT_PALETTE[pred]};
                 text-shadow:0 0 60px {DIGIT_PALETTE[pred]};">{pred}</div>
            <div class="pred-conf" style="color:{conf_color};">
                {conf*100:.1f}% confianza
            </div>
            <div style="margin-top:.6rem;">
                <span class="model-tag">{AVAILABLE_MODELS[model_name]['icon']} {model_name}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if input_source == "sample" and sample_label is not None:
            correct = pred == sample_label
            badge_color = ACCENT if correct else ACCENT2
            badge_text  = f"✅ CORRECTO (real: {sample_label})" if correct else f"❌ ERROR (real: {sample_label})"
            st.markdown(f"""
            <div style="text-align:center;margin-top:.5rem;font-family:'Share Tech Mono';
                        font-size:.8rem;color:{badge_color};">{badge_text}</div>
            """, unsafe_allow_html=True)

        # Top-K probabilidades
        if show_all_probs:
            st.markdown(f'<div class="sec" style="margin-top:.8rem;">// PROBABILIDADES POR CLASE</div>', unsafe_allow_html=True)

            for i in top_k_idx:
                p     = float(probs[i])
                bar_w = int(p * 100)
                col_b = DIGIT_PALETTE[i]
                is_top = "font-weight:700;" if i == pred else ""
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:8px;margin:3px 0;">
                    <div style="font-family:'Share Tech Mono';font-size:.9rem;
                                color:{col_b};width:18px;text-align:center;{is_top}">{i}</div>
                    <div style="flex:1;background:{BORDER};border-radius:4px;height:12px;overflow:hidden;">
                        <div style="width:{bar_w}%;height:100%;background:{col_b};
                                    border-radius:4px;opacity:0.85;"></div>
                    </div>
                    <div style="font-family:'Share Tech Mono';font-size:.8rem;color:{TEXT};
                                width:48px;text-align:right;">{p*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# FILA INFERIOR — EXPLICABILIDAD
# ──────────────────────────────────────────────────────────────
if input_vector is not None:
    st.markdown("---")

    pipe  = trained_models[model_name]["pipe"]
    probs = pipe.predict_proba(input_vector.reshape(1, -1))[0]
    pred  = int(np.argmax(probs))

    tab_process, tab_pixels, tab_neighbors, tab_cm, tab_how = st.tabs([
        "🔄 Proceso paso a paso",
        "🖼️ Visualización del input",
        "👥 Vecinos similares",
        "📊 Matriz de confusión",
        "📖 ¿Cómo funciona?",
    ])

    # ── PROCESO PASO A PASO ────────────────────────────────────
    with tab_process:
        st.markdown('<div class="sec">// PIPELINE DE CLASIFICACIÓN</div>', unsafe_allow_html=True)

        steps_info = [
            ("1. CAPTURA", f"Se captura la imagen dibujada ({280}×{280} px) desde el canvas."),
            ("2. REDIMENSIÓN", "La imagen se escala a 8×8 píxeles (64 valores) para coincidir con el formato del dataset Digits."),
            ("3. NORMALIZACIÓN", f"Los valores de píxel se normalizan al rango [0, 1]. Antes del escalado interno, el StandardScaler del pipeline resta la media y divide por la desviación estándar aprendida en el entrenamiento."),
            ("4. EXTRACCIÓN DE FEATURES", f"El vector resultante tiene {64} features. Cada feature = intensidad de un pixel de la cuadrícula 8×8."),
            ("5. CLASIFICACIÓN", f"El modelo {model_name} calcula la probabilidad para cada uno de los 10 dígitos."),
            ("6. DECISIÓN", f"Se selecciona el dígito con mayor probabilidad: → {pred} ({probs[pred]*100:.1f}%)"),
        ]

        for title, text in steps_info:
            st.markdown(f"""
            <div class="step">
                <div class="step-num">{title}</div>
                <div class="step-text">{text}</div>
            </div>
            """, unsafe_allow_html=True)

        # Diagrama visual del pipeline
        st.markdown('<div class="sec">// REPRESENTACIÓN DEL VECTOR DE ENTRADA</div>', unsafe_allow_html=True)
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
        fig.patch.set_facecolor(BG)

        # a) Imagen 8x8
        ax = axes[0]
        ax.set_facecolor(SURF)
        ax.imshow(input_vector.reshape(8, 8), cmap="plasma", interpolation="nearest",
                  vmin=0, vmax=1)
        ax.set_title("Input (8×8)", color=TEXT, fontsize=9)
        ax.axis("off")
        for i in range(9):
            ax.axhline(i - 0.5, color=BG, linewidth=0.5)
            ax.axvline(i - 0.5, color=BG, linewidth=0.5)

        # b) Heatmap con valores
        ax2 = axes[1]
        ax2.set_facecolor(SURF)
        mat = input_vector.reshape(8, 8)
        im  = ax2.imshow(mat, cmap="plasma", interpolation="nearest", vmin=0, vmax=1)
        for r in range(8):
            for c in range(8):
                v = mat[r, c]
                color = "white" if v < 0.5 else "black"
                ax2.text(c, r, f"{v:.1f}", ha="center", va="center",
                         fontsize=5, color=color)
        ax2.set_title("Valores de intensidad", color=TEXT, fontsize=9)
        ax2.axis("off")
        plt.colorbar(im, ax=ax2, shrink=0.8, label="Intensidad")

        # c) Vector 1D
        ax3 = axes[2]
        dax(ax3)
        ax3.bar(range(64), input_vector, color=ACCENT3, alpha=0.7, width=1.0)
        ax3.set_xlim(-1, 64)
        ax3.set_title("Vector 64D (input al modelo)", color=TEXT, fontsize=9)
        ax3.set_xlabel("Índice de pixel")
        ax3.set_ylabel("Intensidad")

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ── VISUALIZACIÓN DEL INPUT ────────────────────────────────
    with tab_pixels:
        st.markdown('<div class="sec">// ANÁLISIS VISUAL DEL DÍGITO INGRESADO</div>', unsafe_allow_html=True)

        c1, c2 = st.columns([1, 2])
        with c1:
            fig, ax = plt.subplots(figsize=(4, 4))
            fig.patch.set_facecolor(BG)
            ax.set_facecolor(BG)
            im = ax.imshow(input_vector.reshape(8, 8), cmap="plasma",
                           interpolation="nearest", vmin=0, vmax=1)
            ax.set_title(f"Tu dígito → {pred}", color=DIGIT_PALETTE[pred], fontsize=12)
            ax.axis("off")
            plt.colorbar(im, ax=ax, shrink=0.8)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with c2:
            if show_pixel_imp:
                st.markdown('<div class="sec">// MAPA DE IMPORTANCIA DE PIXELS</div>', unsafe_allow_html=True)
                clf_step = trained_models[model_name]["pipe"]["clf"]

                if hasattr(clf_step, "feature_importances_"):
                    imp = clf_step.feature_importances_
                    method_text = "Importancia de features (Random Forest / Extra Trees)"
                elif hasattr(clf_step, "coef_"):
                    # Para clasificación multiclase: importancia relativa al dígito predicho
                    coef = clf_step.coef_
                    imp  = np.abs(coef[pred]) / np.abs(coef[pred]).max()
                    method_text = f"Magnitud de coeficientes para dígito {pred}"
                else:
                    imp = None
                    method_text = "Importancia no disponible para este modelo"

                if imp is not None:
                    fig, axes2 = plt.subplots(1, 2, figsize=(9, 4))
                    fig.patch.set_facecolor(BG)

                    ax_a = axes2[0]
                    ax_a.set_facecolor(BG)
                    ax_a.imshow(imp.reshape(8, 8), cmap="hot", interpolation="bilinear",
                                vmin=0)
                    ax_a.set_title("Pixels más importantes", color=TEXT, fontsize=9)
                    ax_a.axis("off")

                    # Superposición: input × importancia
                    ax_b = axes2[1]
                    ax_b.set_facecolor(BG)
                    overlay = input_vector * imp
                    overlay = overlay / overlay.max() if overlay.max() > 0 else overlay
                    ax_b.imshow(overlay.reshape(8, 8), cmap="inferno",
                                interpolation="bilinear", vmin=0, vmax=1)
                    ax_b.set_title("Input × Importancia (solapado)", color=TEXT, fontsize=9)
                    ax_b.axis("off")

                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                    st.markdown(f"<div style='color:{MUTED};font-size:.78rem;margin-top:.3rem;'>{method_text}</div>",
                                unsafe_allow_html=True)
                else:
                    st.info(method_text)

        # Comparar con promedio de clase
        st.markdown('<div class="sec">// COMPARACIÓN CON PROMEDIO DE CLASE PREDICHA</div>', unsafe_allow_html=True)
        X_all = trained_models[model_name]["X_all"]
        y_all = trained_models[model_name]["y_all"]

        fig, axes3 = plt.subplots(1, 4, figsize=(14, 3.5))
        fig.patch.set_facecolor(BG)
        imgs_to_show = [
            (input_vector.reshape(8, 8), f"Tu input → pred: {pred}", DIGIT_PALETTE[pred]),
        ]
        for digit in [pred] + [d for d in range(10) if d != pred][:2]:
            mask = y_all == digit
            avg  = X_all[mask].mean(axis=0).reshape(8, 8)
            imgs_to_show.append((avg, f"Promedio clase {digit}", DIGIT_PALETTE[digit]))

        for ax, (img, title, color) in zip(axes3, imgs_to_show):
            ax.set_facecolor(BG)
            ax.imshow(img, cmap="plasma", interpolation="bilinear", vmin=0, vmax=1)
            ax.set_title(title, color=color, fontsize=8, pad=4)
            ax.axis("off")

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ── VECINOS MÁS SIMILARES ─────────────────────────────────
    with tab_neighbors:
        st.markdown('<div class="sec">// K VECINOS MÁS SIMILARES EN EL DATASET</div>', unsafe_allow_html=True)

        if show_neighbors:
            X_all = trained_models[model_name]["X_all"]
            y_all = trained_models[model_name]["y_all"]

            # Distancia euclidiana
            dists = np.linalg.norm(X_all - input_vector, axis=1)
            top_n = 10
            nn_idx = np.argsort(dists)[:top_n]

            fig, axes4 = plt.subplots(2, 5, figsize=(12, 5))
            fig.patch.set_facecolor(BG)

            for i, idx in enumerate(nn_idx):
                ax = axes4[i // 5, i % 5]
                ax.set_facecolor(BG)
                label  = int(y_all[idx])
                dist   = dists[idx]
                correct = label == pred
                color  = DIGIT_PALETTE[label]

                ax.imshow(X_all[idx].reshape(8, 8), cmap="plasma",
                          interpolation="nearest", vmin=0, vmax=1)
                mark = "✓" if correct else "✗"
                ax.set_title(f"{mark} Clase:{label}  d={dist:.2f}",
                             color=color, fontsize=7, pad=3)
                ax.axis("off")

                # Borde de color
                for spine in ax.spines.values():
                    spine.set_edgecolor(ACCENT if correct else ACCENT2)
                    spine.set_linewidth(2)

            fig.suptitle(f"Los 10 dígitos más parecidos a tu input (predicción: {pred})",
                         color=TEXT, fontsize=10)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            # Distribución de clases entre vecinos
            nn_labels = y_all[nn_idx]
            st.markdown(f'<div class="sec">// DISTRIBUCIÓN DE CLASES ENTRE VECINOS</div>', unsafe_allow_html=True)

            counts = {d: int(np.sum(nn_labels == d)) for d in range(10) if np.sum(nn_labels == d) > 0}
            fig, ax = plt.subplots(figsize=(8, 2.5))
            dax(ax)
            if counts:
                ax.bar(list(counts.keys()), list(counts.values()),
                       color=[DIGIT_PALETTE[k] for k in counts.keys()], alpha=0.85)
            ax.set_xticks(range(10))
            ax.set_xlabel("Dígito")
            ax.set_ylabel("# vecinos")
            ax.set_title(f"Clases de los {top_n} vecinos más cercanos")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    # ── MATRIZ DE CONFUSIÓN DEL MODELO ────────────────────────
    with tab_cm:
        st.markdown('<div class="sec">// RENDIMIENTO GENERAL DEL MODELO</div>', unsafe_allow_html=True)

        X_te = trained_models[model_name]["X_te"]
        y_te = trained_models[model_name]["y_te"]
        y_pred_all = trained_models[model_name]["pipe"].predict(X_te)

        col_m1, col_m2, col_m3 = st.columns(3)
        acc = accuracy_score(y_te, y_pred_all)
        for col, label, val in zip(
            [col_m1, col_m2, col_m3],
            ["Accuracy global", "Muestras test", "Errores"],
            [f"{acc*100:.2f}%", str(len(y_te)), str(int((1-acc)*len(y_te)))],
        ):
            col.markdown(f"""
            <div class="info-box" style="text-align:center">
                <div class="ib-label">{label}</div>
                <div class="ib-val" style="color:{ACCENT};">{val}</div>
            </div>""", unsafe_allow_html=True)

        # Matriz
        cm      = confusion_matrix(y_te, y_pred_all)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        fig, axes5 = plt.subplots(1, 2, figsize=(14, 6))
        fig.patch.set_facecolor(BG)

        for ax, data, fmt, title in [
            (axes5[0], cm,      "d",    "Conteos absolutos"),
            (axes5[1], cm_norm, ".2f",  "Normalizada (por fila)"),
        ]:
            ax.set_facecolor(SURF)
            sns.heatmap(data, annot=True, fmt=fmt, ax=ax,
                        cmap="YlOrRd", linewidths=0.3, linecolor=BG,
                        xticklabels=range(10), yticklabels=range(10),
                        cbar_kws={"shrink": 0.75})
            ax.set_title(f"{model_name} — {title}", color=TEXT, fontsize=10, pad=8)
            ax.set_xlabel("Predicho", color=TEXT, fontsize=9)
            ax.set_ylabel("Real",     color=TEXT, fontsize=9)
            ax.tick_params(colors=TEXT, labelsize=8)

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # F1 por dígito
        from sklearn.metrics import f1_score as f1s
        f1_per = f1s(y_te, y_pred_all, average=None, zero_division=0)

        fig, ax = plt.subplots(figsize=(9, 3.5))
        dax(ax)
        bars = ax.bar(range(10), f1_per, color=DIGIT_PALETTE, alpha=0.85, edgecolor=BG)
        for bar, v in zip(bars, f1_per):
            ax.text(bar.get_x()+bar.get_width()/2, v+0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8, color=TEXT)
        ax.set_xticks(range(10))
        ax.set_xlabel("Dígito")
        ax.set_ylabel("F1-Score")
        ax.set_ylim(0, 1.1)
        ax.set_title(f"F1-Score por dígito — {model_name}")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ── ¿CÓMO FUNCIONA? ──────────────────────────────────────
    with tab_how:
        st.markdown('<div class="sec">// ¿CÓMO FUNCIONA EL CLASIFICADOR?</div>', unsafe_allow_html=True)

        explanations = {
            "Random Forest": {
                "pasos": [
                    ("Entrada", "El dígito 8×8 (64 píxeles) entra al bosque."),
                    ("Árboles de decisión", "200 árboles se entrenaron sobre muestras aleatorias del dataset. Cada árbol aprendió reglas como: 'Si pixel[20] > 0.5 Y pixel[35] < 0.3 → probablemente es 3'."),
                    ("Votación", "Cada árbol vota por una clase. La clase con más votos gana."),
                    ("Probabilidad", "La probabilidad es la fracción de árboles que votaron por cada clase."),
                    ("Fortaleza", "Robusto al ruido. No necesita que el dígito esté perfectamente centrado."),
                ],
                "formula": "P(clase k) = (# árboles que votan k) / (# total de árboles)",
            },
            "SVM (RBF)": {
                "pasos": [
                    ("Espacio de features", "Los 64 píxeles definen un punto en un espacio 64-dimensional."),
                    ("Kernel RBF", "Transforma el espacio de entrada a uno de mayor dimensión donde las clases son linealmente separables."),
                    ("Hiperplanos", "Durante el entrenamiento, se calcularon hiperplanos óptimos que separan cada par de clases."),
                    ("Clasificación", "El punto se asigna a la región correspondiente al dígito más cercano."),
                    ("Fortaleza", "Excelente en espacios de alta dimensión. Alto accuracy con imágenes bien dibujadas."),
                ],
                "formula": "K(x, xᵢ) = exp(-γ ||x - xᵢ||²) → decisión por margen máximo",
            },
            "Logistic Regression": {
                "pasos": [
                    ("Modelo lineal", "Aprende un vector de pesos w para cada clase (10 vectores de 64 dimensiones)."),
                    ("Score por clase", "Para cada dígito k calcula: score(k) = w_k · x + b_k (producto punto con el vector de entrada)."),
                    ("Softmax", "Convierte los 10 scores en probabilidades que suman 1."),
                    ("Decisión", "Se elige la clase con mayor probabilidad."),
                    ("Fortaleza", "Rápido y transparente. Los pesos muestran qué píxeles importan más."),
                ],
                "formula": "P(k|x) = softmax(Wx + b) = exp(wₖ·x) / Σⱼ exp(wⱼ·x)",
            },
            "K-Nearest Neighbors": {
                "pasos": [
                    ("Memoria", "KNN almacena todos los 1437 ejemplos de entrenamiento."),
                    ("Distancia", "Calcula la distancia euclidiana entre tu dígito y todos los de entrenamiento."),
                    ("Selección", "Selecciona los 5 ejemplos más cercanos (vecinos)."),
                    ("Votación", "La clase más frecuente entre los 5 vecinos es la predicción."),
                    ("Fortaleza", "Intuitivo. Literalmente compara tu dígito con los que ya conoce."),
                ],
                "formula": "pred = moda({y_i : i ∈ k vecinos más cercanos a x})",
            },
            "Gradient Boosting": {
                "pasos": [
                    ("Árbol inicial", "Entrena un árbol de decisión simple sobre los datos."),
                    ("Corrección iterativa", "En cada iteración, un nuevo árbol aprende a corregir los errores del anterior."),
                    ("150 iteraciones", "Después de 150 correcciones, el modelo es muy preciso."),
                    ("Suma de árboles", "La predicción final es la suma ponderada de todos los árboles."),
                    ("Fortaleza", "Muy preciso. Captura patrones complejos que otros modelos pierden."),
                ],
                "formula": "F(x) = Σₘ γₘ · hₘ(x) donde cada hₘ corrige errores residuales",
            },
            "Naive Bayes": {
                "pasos": [
                    ("Prior", "Aprende qué tan frecuente es cada dígito (P(clase))."),
                    ("Likelihood", "Para cada pixel aprende: dado que es el dígito k, ¿qué tan probable es este valor de intensidad?"),
                    ("Independencia", "Asume que cada pixel es independiente de los demás (por eso 'naive')."),
                    ("Bayes", "Combina prior y likelihood para calcular la probabilidad posterior de cada clase."),
                    ("Limitación", "Los píxeles de un dígito no son independientes, pero funciona sorprendentemente bien."),
                ],
                "formula": "P(k|x) ∝ P(k) × Πᵢ P(xᵢ|k)  [Teorema de Bayes]",
            },
        }

        info_exp = explanations.get(model_name, {})
        if info_exp:
            st.markdown(f"""
            <div style="background:{CARD};border:1px solid {BORDER};border-radius:12px;
                        padding:1.2rem;margin-bottom:1rem;">
                <div style="font-family:'Share Tech Mono';font-size:.7rem;color:{MUTED};
                            letter-spacing:2px;margin-bottom:.5rem;">FÓRMULA CLAVE</div>
                <div style="font-family:'Share Tech Mono';font-size:.9rem;color:{ACCENT};
                            background:{BG};padding:.7rem;border-radius:8px;">
                    {info_exp['formula']}
                </div>
            </div>
            """, unsafe_allow_html=True)

            for title, text in info_exp["pasos"]:
                st.markdown(f"""
                <div class="step">
                    <div class="step-num">→ {title.upper()}</div>
                    <div class="step-text">{text}</div>
                </div>
                """, unsafe_allow_html=True)

        # Comparativa de modelos
        st.markdown('<div class="sec">// COMPARATIVA DE TODOS LOS MODELOS</div>', unsafe_allow_html=True)

        rows_cmp = []
        for nm, info_m in trained_models.items():
            rows_cmp.append({
                "Modelo": f"{AVAILABLE_MODELS[nm]['icon']} {nm}",
                "Accuracy": f"{info_m['acc']*100:.2f}%",
                "Tipo":     "Ensemble" if "Forest" in nm or "Boosting" in nm
                            else "Kernel" if "SVM" in nm
                            else "Lineal" if "Logistic" in nm
                            else "Instancia" if "Nearest" in nm
                            else "Probabilístico",
                "Velocidad": "⚡⚡⚡" if "Naive" in nm or "Logistic" in nm
                             else "⚡⚡" if "Forest" in nm or "KNN" in nm
                             else "⚡",
            })
        df_cmp = pd.DataFrame(rows_cmp).set_index("Modelo")
        st.dataframe(df_cmp, use_container_width=True)

# ──────────────────────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(f"""
<div style='text-align:center;color:{MUTED};font-family:Share Tech Mono;
            font-size:.72rem;letter-spacing:2px;'>
    DIGIT.AI · SCIKIT-LEARN + STREAMLIT · DIGITS DATASET (sklearn)
</div>
""", unsafe_allow_html=True)
