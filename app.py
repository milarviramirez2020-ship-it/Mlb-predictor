"""
MLB Motor vFinal - App Web de Predicciones
Motor de análisis de valor para apuestas MLB con simulación Monte Carlo.
"""

import numpy as np
import statsapi 
import requests
import warnings
from datetime import datetime

import streamlit as st

warnings.filterwarnings("ignore")

# Configuración global
N_SIMULACIONES = 10000
ERA_LIGA = 4.20
INNINGS_STARTER = 5.2
INNINGS_BULLPEN = 9 - INNINGS_STARTER
FACTOR_AJUSTE = 0.20
PHI = 1.35

PALETA = {
    "fondo": "#0b1220",
    "card": "#121c2e",
    "card_alt": "#162238",
    "verde": "#22c55e",
    "rojo": "#ef4444",
    "amarillo": "#eab308",
    "azul": "#3b82f6",
    "texto": "#e5e7eb",
    "texto_dim": "#94a3b8",
    "borde": "#1f2d45",
}

st.set_page_config(
    page_title="MLB Predictor",
    page_icon="⚾",
    layout="wide",
)

# CSS para diseño bonito
st.markdown(
    f"""
    <style>
        .stApp {{ background-color: {PALETA["fondo"]}; }}
        section[data-testid="stSidebar"] {{
            background-color: {PALETA["card"]};
            border-right: 1px solid {PALETA["borde"]};
        }}
        h1, h2, h3, h4, p, span, div, label {{ color: {PALETA["texto"]}; }}
        .card {{
            background: linear-gradient(145deg, {PALETA["card"]}, {PALETA["card_alt"]});
            border: 1px solid {PALETA["borde"]};
            border-radius: 16px;
            padding: 1.25rem 1.5rem;
            margin-bottom: 1rem;
        }}
        .card-verde {{ border-left: 5px solid {PALETA["verde"]}; }}
        .card-rojo {{ border-left: 5px solid {PALETA["rojo"]}; }}
        .card-amarillo {{ border-left: 5px solid {PALETA["amarillo"]}; }}
        .card-azul {{ border-left: 5px solid {PALETA["azul"]}; }}
        .titulo-card {{ font-size: 0.9rem; color: {PALETA["texto_dim"]}; margin-bottom: 0.25rem; }}
        .valor-grande {{ font-size: 2rem; font-weight: 800; }}
        .badge {{
            display: inline-block; padding: 0.3rem 0.8rem; border-radius: 999px;
            font-size: 0.8rem; font-weight: 700;
        }}
        .badge-verde {{ background: rgba(34,197,94,0.15); color: {PALETA["verde"]}; }}
        .badge-rojo {{ background: rgba(239,68,68,0.15); color: {PALETA["rojo"]}; }}
        .badge-amarillo {{ background: rgba(234,179,8,0.15); color: {PALETA["amarillo"]}; }}
        .score-pred {{
            font-size: 2.5rem; font-weight: 900; text-align: center;
            background: linear-gradient(135deg, {PALETA["azul"]}, {PALETA["verde"]});
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        }}
        .divider {{ height: 1px; background: {PALETA["borde"]}; margin: 1rem 0; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Funciones de simulación
def calcular_ev(prob, cuota):
    return (prob * cuota) - 1

def simular_binomial_negativa(media, phi=PHI, n_sims=N_SIMULACIONES):
    if media <= 0:
        media = 0.5
    p = 1.0 / phi
    r = media / (phi - 1.0)
    r_int = max(1, int(round(r)))
    p_clip = max(0.01, min(0.99, p))
    return np.random.negative_binomial(n=r_int, p=p_clip, size=n_sims)

def run_simulacion(media_loc, media_vis, linea_over, cuota_local, cuota_visit, cuota_over):
    np.random.seed(None)
    runs_loc = simular_binomial_negativa(media_loc)
    runs_vis = simular_binomial_negativa(media_vis)

    win_local = np.mean(runs_loc > runs_vis)
    win_visit = np.mean(runs_vis > runs_loc)
    empate = np.mean(runs_loc == runs_vis)

    if empate > 0:
        total = win_local + win_visit
        if total > 0:
            win_local += empate * (win_local / total)
            win_visit += empate * (win_visit / total)

    total_runs = runs_loc + runs_vis
    over_prob = np.mean(total_runs > linea_over) if linea_over else 0.0
    under_prob = np.mean(total_runs < linea_over) if linea_over else 0.0

    ev_local = calcular_ev(win_local, cuota_local)
    ev_visit = calcular_ev(win_visit, cuota_visit)
    ev_over = calcular_ev(over_prob, cuota_over)

    return {
        "win_local": win_local,
        "win_visit": win_visit,
        "over_prob": over_prob,
        "under_prob": under_prob,
        "ev_local": ev_local,
        "ev_visit": ev_visit,
        "ev_over": ev_over,
        "runs_loc_mean": float(np.mean(runs_loc)),
        "runs_vis_mean": float(np.mean(runs_vis)),
        "total_mean": float(np.mean(total_runs)),
        "linea_over": linea_over,
    }

def calcular_medias_ajustadas(
    media_anot_local, media_perm_local, media_anot_visit, media_perm_visit,
    era_local_starter, era_visit_starter, era_local_bullpen, era_visit_bullpen,
    factor_estadio,
):
    era_loc = (era_local_starter * INNINGS_STARTER + era_local_bullpen * INNINGS_BULLPEN) / 9
    era_vis = (era_visit_starter * INNINGS_STARTER + era_visit_bullpen * INNINGS_BULLPEN) / 9

    ajuste_loc = (era_vis - ERA_LIGA) * FACTOR_AJUSTE
    ajuste_vis = (era_loc - ERA_LIGA) * FACTOR_AJUSTE

    media_loc = (media_anot_local + media_perm_visit) / 2 + ajuste_loc + factor_estadio
    media_vis = (media_anot_visit + media_perm_local) / 2 + ajuste_vis

    media_loc = max(0.5, media_loc)
    media_vis = max(0.5, media_vis)
    return media_loc, media_vis

# Buscar partido con statsapi
@st.cache_data(show_spinner=False, ttl=300)
def buscar_partido(local, visitante, fecha):
    if fecha is None:
        fecha = datetime.now().strftime("%Y-%m-%d")
    try:
        juegos = statsapi.schedule(date=fecha)
    except Exception as e:
        st.error(f"Error conectando con MLB API: {e}")
        return None, None

    partido = None
    for j in juegos:
        home_name = j.get("home_name", "").lower()
        away_name = j.get("away_name", "").lower()
        if local.lower() in home_name and visitante.lower() in away_name:
            partido = j
            break
    return partido, juegos

@st.cache_data(show_spinner=False, ttl=600)
def obtener_ultimos_juegos(team_id, n=60):
    year = datetime.now().year
    try:
        schedule = statsapi.schedule(
            start_date=f"{year}-01-01", end_date=f"{year}-12-31", team=team_id
        )
    except Exception:
        return [], []

    finalizados = [g for g in schedule if g.get("status") in ["Final", "Game Over"]]
    ultimos = finalizados[-n:] if len(finalizados) >= n else finalizados

    anotadas, permitidas = [], []
    for g in ultimos:
        if g.get("home_id") == team_id:
            anotadas.append(g.get("home_score", 0) or 0)
            permitidas.append(g.get("away_score", 0) or 0)
        else:
            anotadas.append(g.get("away_score", 0) or 0)
            permitidas.append(g.get("home_score", 0) or 0)
    return anotadas, permitidas

# ===== INTERFAZ DE USUARIO =====

st.sidebar.markdown(
    f'<div style="text-align:center;margin-bottom:1rem;">'
    f'<h2 style="margin:0;">⚾ MLB Predictor</h2>'
    f'<p style="color:{PALETA["texto_dim"]};font-size:0.85rem;">Monte Carlo · 10,000 iteraciones</p>'
    f'</div>',
    unsafe_allow_html=True,
)

# Inputs del usuario
equipo_local = st.sidebar.text_input("🏠 Equipo Local", value="yankees", help="Ej: yankees, boston, texas")
equipo_visitante = st.sidebar.text_input("✈️ Equipo Visitante", value="red sox", help="Ej: red sox, dodgers, rays")
fecha_sel = st.sidebar.date_input("📅 Fecha", value=datetime.now())
fecha_str = fecha_sel.strftime("%Y-%m-%d")

# Inputs de cuotas y pitchers
st.sidebar.markdown("### 💰 Cuotas (decimal)")
cuota_local = st.sidebar.number_input("Cuota Local", min_value=1.01, value=1.86, step=0.01, format="%.2f")
cuota_visitante = st.sidebar.number_input("Cuota Visitante", min_value=1.01, value=1.95, step=0.01, format="%.2f")
cuota_over = st.sidebar.number_input("Cuota Over", min_value=1.01, value=1.83, step=0.01, format="%.2f")
linea_over = st.sidebar.number_input("Línea Over/Under", min_value=1.0, value=8.5, step=0.5, format="%.1f")

st.sidebar.markdown("### 🎯 Pitchers (ERA)")
era_local_starter = st.sidebar.number_input("ERA Abridor Local", min_value=0.0, value=4.40, step=0.01, format="%.2f")
era_visitante_starter = st.sidebar.number_input("ERA Abridor Visitante", min_value=0.0, value=5.60, step=0.01, format="%.2f")

st.sidebar.markdown("### 🧢 Bullpen (ERA)")
era_local_bullpen = st.sidebar.number_input("ERA Bullpen Local", min_value=0.0, value=4.37, step=0.01, format="%.2f")
era_visitante_bullpen = st.sidebar.number_input("ERA Bullpen Visitante", min_value=0.0, value=4.03, step=0.01, format="%.2f")

factor_estadio = st.sidebar.slider("🏟️ Factor Estadio", min_value=0.0, max_value=1.5, value=0.5, step=0.1)

boton_analizar = st.sidebar.button("🚀 Analizar Partido", use_container_width=True)

# Cuerpo principal
st.markdown(
    f'<div style="text-align:center;margin-bottom:2rem;">'
    f'<h1>⚾ MLB Predictor</h1>'
    f'<p style="color:{PALETA["texto_dim"]};">Predicciones con simulación Monte Carlo · 10,000 iteraciones</p>'
    f'</div>',
    unsafe_allow_html=True,
)

if boton_analizar:
    with st.spinner("🔍 Buscando partido..."):
        partido, juegos = buscar_partido(equipo_local, equipo_visitante, fecha_str)
        
        if partido is None:
            st.error(f"❌ Partido no encontrado: {equipo_visitante} @ {equipo_local} el {fecha_str}")
            st.info("💡 Intenta con nombres más cortos: 'yankees' en vez de 'new york yankees'")
            st.stop()
        
        home_name = partido.get("home_name", equipo_local.title())
        away_name = partido.get("away_name", equipo_visitante.title())
        st.success(f"✅ Partido encontrado: {away_name} @ {home_name}")
        st.caption(f"🏟️ {partido.get('venue_name', 'N/A')} · {fecha_str}")
    
    with st.spinner("📊 Cargando datos históricos..."):
        loc_anot, loc_perm = obtener_ultimos_juegos(partido["home_id"], n=60)
        vis_anot, vis_perm = obtener_ultimos_juegos(partido["away_id"], n=60)
        
        if len(loc_anot) < 10 or len(vis_anot) < 10:
            st.warning("⚠️ Pocos datos históricos. Usando medias de liga (4.5).")
            media_anot_local = media_perm_local = media_anot_visit = media_perm_visit = 4.5
        else:
            media_anot_local = np.mean(loc_anot)
            media_perm_local = np.mean(loc_perm)
            media_anot_visit = np.mean(vis_anot)
            media_perm_visit = np.mean(vis_perm)
    
    with st.spinner("🧮 Calculando medias ajustadas..."):
        media_loc, media_vis = calcular_medias_ajustadas(
            media_anot_local, media_perm_local, media_anot_visit, media_perm_visit,
            era_local_starter, era_visitante_starter, era_local_bullpen, era_visitante_bullpen,
            factor_estadio
        )
    
    with st.spinner(f"🎲 Simulando {N_SIMULACIONES:,} partidos..."):
        resultado = run_simulacion(
            media_loc, media_vis, linea_over,
            cuota_local, cuota_visitante, cuota_over
        )
    
    # Mostrar resultados
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        ev_local = resultado["ev_local"] * 100
        color = "verde" if resultado["ev_local"] > 0.03 else ("amarillo" if resultado["ev_local"] > -0.02 else "rojo")
        badge = "✅ CON VALOR" if resultado["ev_local"] > 0.03 else ("⚠️ MARGINAL" if resultado["ev_local"] > -0.02 else "❌ NO APOSTAR")
        st.markdown(
            f"""
            <div class="card card-{color}">
                <div class="titulo-card">🏠 {home_name}</div>
                <div class="valor-grande">{resultado["win_local"]*100:.1f}%</div>
                <div style="color:{PALETA["texto_dim"]};">EV: {ev_local:.1f}% · Cuota: {cuota_local:.2f}</div>
                <div class="badge badge-{color}">{badge}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    with col2:
        ev_visit = resultado["ev_visit"] * 100
        color = "verde" if resultado["ev_visit"] > 0.03 else ("amarillo" if resultado["ev_visit"] > -0.02 else "rojo")
        badge = "✅ CON VALOR" if resultado["ev_visit"] > 0.03 else ("⚠️ MARGINAL" if resultado["ev_visit"] > -0.02 else "❌ NO APOSTAR")
        st.markdown(
            f"""
            <div class="card card-{color}">
                <div class="titulo-card">✈️ {away_name}</div>
                <div class="valor-grande">{resultado["win_visit"]*100:.1f}%</div>
                <div style="color:{PALETA["texto_dim"]};">EV: {ev_visit:.1f}% · Cuota: {cuota_visitante:.2f}</div>
                <div class="badge badge-{color}">{badge}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    with col3:
        ev_over = resultado["ev_over"] * 100
        color = "verde" if resultado["ev_over"] > 0.03 else ("amarillo" if resultado["ev_over"] > -0.02 else "rojo")
        badge = "✅ CON VALOR" if resultado["ev_over"] > 0.03 else ("⚠️ MARGINAL" if resultado["ev_over"] > -0.02 else "❌ NO APOSTAR")
        st.markdown(
            f"""
            <div class="card card-{color}">
                <div class="titulo-card">Over {linea_over}</div>
                <div class="valor-grande">{resultado["over_prob"]*100:.1f}%</div>
                <div style="color:{PALETA["texto_dim"]};">EV: {ev_over:.1f}% · Cuota: {cuota_over:.2f}</div>
                <div class="badge badge-{color}">{badge}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    # Score predicho
    st.markdown(
        f"""
        <div class="card card-azul" style="text-align:center;">
            <div class="titulo-card">📊 SCORE PREDICHO</div>
            <div class="score-pred">{home_name} {resultado["runs_loc_mean"]:.1f} - {resultado["runs_vis_mean"]:.1f} {away_name}</div>
            <div style="color:{PALETA["texto_dim"]};font-size:0.9rem;">
                Total promedio: {resultado["total_mean"]:.1f} carreras
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Recomendación final
    picks = []
    if resultado["ev_local"] > 0.03:
        picks.append(f"✅ {home_name} ML @ {cuota_local:.2f} (+{resultado['ev_local']*100:.1f}% EV)")
    if resultado["ev_visit"] > 0.03:
        picks.append(f"✅ {away_name} ML @ {cuota_visitante:.2f} (+{resultado['ev_visit']*100:.1f}% EV)")
    if resultado["ev_over"] > 0.03:
        picks.append(f"✅ Over {linea_over} @ {cuota_over:.2f} (+{resultado['ev_over']*100:.1f}% EV)")
    
    if picks:
        st.markdown(
            f"""
            <div class="card card-verde">
                <div class="titulo-card">🎯 RECOMENDACIÓN FINAL</div>
                <div style="font-size:1.1rem;font-weight:600;color:{PALETA["verde"]};">
                    APUESTAS CON VALOR ENCONTRADAS
                </div>
                <div style="margin-top:0.5rem;">
                    {"<br>".join(picks)}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="card card-rojo">
                <div class="titulo-card">⚠️ RECOMENDACIÓN FINAL</div>
                <div style="font-size:1.1rem;font-weight:600;color:{PALETA["rojo"]};">
                    NO HAY APUESTAS CON VALOR
                </div>
                <div style="color:{PALETA["texto_dim"]};margin-top:0.5rem;">
                    Guarda tu dinero. Ningún pick supera el +3% de EV.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
