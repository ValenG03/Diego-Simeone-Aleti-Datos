"""EL EFECTO CHOLO — Atlético de Madrid 2005/06–2025/26
Rendimiento deportivo vs. recursos económicos. Stack: Streamlit · Pandas · Plotly · Seaborn · K-means
"""
import base64
from pathlib import Path

from PIL import Image

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="El Efecto Cholo", page_icon="🔴", layout="wide")
RED, DARK, BLACK, WHITE, GREY = "#E0402F", "#7A2018", "#14161C", "#F5F5F5", "#9BA0AB"
CARD, TEXT = "#20232C", "#E6E8EC"
ROOT = Path(__file__).parent

@st.cache_data
def b64(name):  # busca la foto en la raíz del repo o en assets/
    p = next((d / name for d in (ROOT, ROOT / "assets") if (d / name).exists()), None)
    if p is None:
        return ""
    mime = "png" if p.suffix.lower() == ".png" else "jpeg"
    return f"data:image/{mime};base64,{base64.b64encode(p.read_bytes()).decode()}"

@st.cache_data
def ratio(name):  # ancho / alto de la foto, para que el marco la abrace con márgenes iguales
    p = next((d / name for d in (ROOT, ROOT / "assets") if (d / name).exists()), None)
    if p is None:
        return 1.0
    w, h = Image.open(p).size
    return round(w / h, 4)

# ─────────────────────────── ESTILO ───────────────────────────
st.markdown(f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@400;600;800&family=Space+Grotesk:wght@400;500;700&display=swap');
.stApp {{background:
  radial-gradient(circle at 85% -5%,{RED}40 0%,transparent 40%),
  radial-gradient(circle at 0% 100%,{DARK}55 0%,transparent 45%),
  linear-gradient(rgba(255,255,255,.035) 1px,transparent 1px) 0 0/44px 44px,
  linear-gradient(90deg,rgba(255,255,255,.035) 1px,transparent 1px) 0 0/44px 44px,
  {BLACK};color:{TEXT};font-family:Inter,sans-serif}}
[data-testid="stHeader"] {{background:transparent}}
[data-testid="stSidebar"] {{background:#1A1D24;border-right:1px solid {RED}55}}
.stApp p, .stApp label, .stApp span, [data-testid="stCaptionContainer"] {{color:{TEXT}}}
[data-testid="stCaptionContainer"] p {{color:{GREY}!important}}
h1,h2,h3 {{font-family:'Bebas Neue',Impact,sans-serif;letter-spacing:2px;color:{WHITE}!important}}
.top {{position:relative;display:grid;grid-template-columns:minmax(70px,1fr) minmax(0,auto) minmax(70px,1fr);
  align-items:start;gap:12px;margin-top:-1rem;padding:1.4rem 0 .6rem}}
.top::before {{content:"";position:absolute;inset:-10% 15% 0;z-index:0;pointer-events:none;
  background:radial-gradient(ellipse at 50% 55%,{RED}38 0%,transparent 62%);animation:breathe 6s ease-in-out infinite}}
.titles, .crest {{position:relative;z-index:1}}
.crest {{justify-self:end;margin-top:14px;
  animation:drop .9s cubic-bezier(.2,.8,.2,1) both .9s, float 5s ease-in-out infinite 1.8s}}
.titles {{text-align:center;display:flex;flex-direction:column;align-items:center}}
.hero {{margin:0;line-height:.86;font-family:'Bebas Neue',Oswald,Impact,sans-serif!important;white-space:nowrap}}
.hero .l1 {{display:block;font-size:clamp(2.4rem,6.2vw,6.2rem)!important;letter-spacing:.32em;margin-right:-.32em;
  background:linear-gradient(100deg,{WHITE} 0%,{WHITE} 40%,#ffd9d4 48%,{RED} 50%,#ffd9d4 52%,{WHITE} 60%,{WHITE} 100%);
  background-size:250% 100%;-webkit-background-clip:text;background-clip:text;color:transparent!important;
  animation:rise .9s cubic-bezier(.2,.8,.2,1) both .1s, sheen 5.5s linear infinite 1.2s}}
.hero .l2 {{position:relative;display:inline-block;font-size:clamp(4.6rem,15vw,13.5rem)!important;letter-spacing:.06em;
  color:{RED}!important;text-shadow:0 0 8px {RED},0 0 28px {RED}cc,0 0 70px {RED}77;
  animation:rise 1s cubic-bezier(.2,.8,.2,1) both .35s, flicker 2.2s linear both 1.1s, pulse 4s ease-in-out infinite 3.4s}}
.hero .l2::before, .hero .l2::after {{content:attr(data-text);position:absolute;inset:0;pointer-events:none;opacity:0}}
.hero .l2::before {{color:{WHITE};animation:glitch 7s steps(1) infinite 4s}}
.hero .l2::after {{color:#2F5BD8;animation:glitch 7s steps(1) infinite 4.08s reverse}}
.rule {{position:relative;height:2px;width:clamp(120px,22vw,320px);margin:1.1rem auto .9rem;overflow:hidden;
  background:linear-gradient(90deg,transparent,{RED},transparent);transform-origin:center;animation:grow .9s ease-out both 1s}}
.rule::after {{content:"";position:absolute;top:-2px;left:-30%;width:30%;height:6px;
  background:radial-gradient(ellipse,{WHITE} 0%,transparent 70%);animation:scan 2.8s ease-in-out infinite 2s}}
.sub {{display:flex;flex-wrap:wrap;justify-content:center;align-items:center;gap:.4rem 1rem;margin:0 0 1.8rem;
  font-family:'Space Grotesk',Inter,sans-serif!important;font-weight:500;text-transform:uppercase;
  letter-spacing:.22em;font-size:clamp(.72rem,1vw,.95rem);color:{TEXT}}}
.sub span {{white-space:nowrap}}
.sub span {{opacity:0;animation:rise .7s ease-out forwards}}
.sub span:nth-of-type(1) {{animation-delay:1.25s}} .sub span:nth-of-type(2) {{animation-delay:1.4s;color:{WHITE};font-weight:700}}
.sub span:nth-of-type(3) {{animation-delay:1.55s}}
.sub i {{width:6px;height:6px;background:{RED};transform:rotate(45deg);box-shadow:0 0 10px {RED};opacity:0;animation:rise .5s ease-out forwards 1.35s}}
@keyframes rise {{from {{opacity:0;transform:translateY(34px);filter:blur(10px)}} to {{opacity:1;transform:none;filter:blur(0)}}}}
@keyframes sheen {{from {{background-position:120% 0}} to {{background-position:-130% 0}}}}
@keyframes flicker {{0%,18%,22%,25%,53%,57%,100% {{opacity:1}} 20%,24%,55% {{opacity:.35}}}}
@keyframes pulse {{0%,100% {{text-shadow:0 0 8px {RED},0 0 28px {RED}cc,0 0 70px {RED}77}}
  50% {{text-shadow:0 0 12px {RED},0 0 42px {RED},0 0 110px {RED}99}}}}
@keyframes glitch {{0%,94%,100% {{opacity:0;transform:none}} 95% {{opacity:.55;transform:translate(-4px,1px);clip-path:inset(18% 0 52% 0)}}
  97% {{opacity:.55;transform:translate(4px,-1px);clip-path:inset(60% 0 12% 0)}}}}
@keyframes grow {{from {{transform:scaleX(0)}} to {{transform:scaleX(1)}}}}
@keyframes scan {{0% {{left:-30%}} 100% {{left:100%}}}}
@keyframes breathe {{0%,100% {{opacity:.75}} 50% {{opacity:1}}}}
@keyframes drop {{from {{opacity:0;transform:translateY(-30px) rotate(-8deg)}} to {{opacity:1;transform:none}}}}
@keyframes float {{0%,100% {{transform:translateY(0)}} 50% {{transform:translateY(-6px)}}}}
@keyframes fadeup {{from {{opacity:0;transform:translateY(24px)}} to {{opacity:1;transform:none}}}}
.banner {{display:block;margin:0 auto 2rem!important;width:100%!important;max-width:1100px;height:clamp(220px,32vw,460px)!important;object-fit:cover!important;object-position:center 40%;border-radius:14px;border:1px solid {RED}88;
  box-shadow:0 0 0 1px #000,0 18px 50px {RED}33;animation:fadeup 1s ease-out both 1.7s}}
[data-testid="stMetric"] {{background:{CARD};border-left:4px solid {RED};border-radius:8px;padding:14px 18px;
  box-shadow:0 0 0 1px #2E323D;transition:.25s}}
[data-testid="stMetric"]:hover {{box-shadow:0 0 26px {RED}66;transform:translateY(-3px)}}
[data-testid="stMetricLabel"] p {{color:{TEXT}!important;font-weight:600}}
[data-testid="stMetricValue"] {{font-family:'Bebas Neue',Impact,sans-serif;font-size:2.8rem;color:{WHITE}}}
.stTabs [data-baseweb="tab"] {{font-family:'Bebas Neue',Impact,sans-serif;font-size:1.3rem;letter-spacing:1px;color:{GREY}}}
.stTabs [aria-selected="true"] {{color:{RED}!important}}
.stTabs [data-baseweb="tab-highlight"] {{background:{RED}}}
.stTabs [role="tablist"] {{justify-content:center!important;gap:.6rem;flex-wrap:wrap}}
.insight {{border:1px solid {RED}66;background:linear-gradient(90deg,{RED}2e,{CARD}cc);padding:12px 18px;
  border-radius:8px;margin-top:.5rem;font-size:.95rem;color:{TEXT}}}
.insight b {{color:{WHITE}}}
.gallery {{display:flex;gap:20px;align-items:flex-start;margin:2.5rem 0 1.5rem}}
.gallery figure {{margin:0;padding:16px;box-sizing:border-box;border-radius:14px;border:1px solid {RED}88;
  background:radial-gradient(circle at 50% 35%,{RED}88 0%,{DARK} 45%,{BLACK} 80%);box-shadow:0 18px 50px {RED}26}}
.gallery img {{display:block;width:100%!important;height:auto!important;max-width:none!important;border-radius:6px}}
.sources {{text-align:center;margin:1rem 0 0;padding:22px 26px;border:1px solid {RED}77;border-radius:14px;
  background:linear-gradient(135deg,{CARD} 0%,#1A1D24 100%);box-shadow:inset 0 0 0 1px #2E323D,0 12px 34px #00000055}}
.sources h4 {{margin:0 0 .5rem;font-family:'Bebas Neue',Impact,sans-serif;letter-spacing:.2em;font-size:1.35rem;color:{RED}!important}}
.sources p {{margin:0;font-size:1.06rem;line-height:1.75;color:{TEXT}!important}}
.sources a {{color:{WHITE}!important;text-decoration:none;border-bottom:1px solid {RED};padding-bottom:1px;transition:.2s}}
.sources a:hover {{color:{RED}!important;border-bottom-color:{WHITE};text-shadow:0 0 12px {RED}88}}
.note {{color:{WHITE}!important;font-size:1.02rem;margin:.6rem 0 1.2rem}}
.sources em {{display:block;margin-top:.5rem;font-style:normal;font-size:.92rem;color:{GREY}}}
.signature {{text-align:center;margin:3rem 0 2.5rem}}
.crest-neon {{position:relative;width:64px;margin:0 auto 4.2rem}}
.crest-neon::before {{content:"";position:absolute;inset:-14px;border-radius:50%;z-index:0;filter:blur(16px);opacity:.75;
  background:conic-gradient({RED},{WHITE},#2F5BD8,{RED});animation:spin 6s linear infinite}}
.top-neon {{margin:0;width:clamp(58px,7vw,96px)}}
.top-neon img {{width:100%!important}}
.crest-neon img {{position:relative;z-index:1;width:64px;display:block;animation:neon 3s ease-in-out infinite}}
@keyframes spin {{to {{transform:rotate(360deg)}}}}
@keyframes neon {{0%,100% {{filter:drop-shadow(0 0 4px {RED}) drop-shadow(0 0 12px {RED}aa)}}
  33% {{filter:drop-shadow(0 0 4px {WHITE}) drop-shadow(0 0 12px #ffffffaa)}}
  66% {{filter:drop-shadow(0 0 4px #2F5BD8) drop-shadow(0 0 12px #2F5BD8aa)}}}}
.signature p {{margin:0;font-size:.95rem;letter-spacing:1px;color:{TEXT}!important}}
.signature b {{color:{RED}}}
@media (max-width:640px) {{.gallery {{flex-direction:column}} .gallery figure {{width:100%}}
  .top {{grid-template-columns:1fr}} .top > div:first-child {{display:none}}
  .crest {{position:absolute;top:0;right:4px;margin:0}} .top-neon {{width:42px}} .titles {{padding-top:52px}}
  .hero .l1 {{font-size:9vw!important;letter-spacing:.22em;margin-right:-.22em}} .hero .l2 {{font-size:22vw!important}}
  .sub {{flex-direction:column;gap:.35rem;letter-spacing:.16em;font-size:.7rem}} .sub i {{display:none}} .sources {{padding:16px 18px}}}}
@media (prefers-reduced-motion:reduce) {{*, *::before, *::after {{animation:none!important}} .sub span, .sub i {{opacity:1}}}}
</style>""", unsafe_allow_html=True)

# ─────────────────────────── DATOS ───────────────────────────
# LaLiga, 38 jornadas por temporada. Validar contra laliga.com antes de publicar.
SEASONS = [f"{y}/{str(y + 1)[2:]}" for y in range(2005, 2026)]
ATM = dict(
    pos=[10, 7, 4, 4, 9, 7, 5, 3, 1, 3, 3, 3, 2, 2, 3, 1, 3, 3, 4, 3, 4],
    pts=[51, 60, 64, 67, 47, 58, 56, 76, 90, 78, 88, 78, 79, 76, 70, 86, 71, 77, 76, 76, 69],
    w=[13, 17, 19, 20, 13, 17, 15, 23, 28, 23, 28, 23, 23, 22, 18, 26, 21, 23, 24, 22, 21],
    d=[12, 9, 7, 7, 8, 7, 11, 7, 6, 9, 4, 9, 10, 10, 16, 8, 8, 8, 4, 10, 6],
    gf=[45, 46, 66, 80, 57, 62, 53, 65, 77, 67, 63, 70, 58, 55, 51, 67, 65, 70, 70, 68, 62],
    ga=[37, 39, 47, 57, 61, 53, 46, 31, 26, 29, 18, 27, 22, 29, 27, 25, 43, 33, 43, 30, 44],
)
RIVALS = {  # (puntos, victorias, goles recibidos)
    "Real Madrid": ([70, 76, 85, 78, 96, 92, 100, 85, 87, 92, 90, 93, 76, 68, 87, 84, 86, 78, 95, 84, 86],
                    [20, 23, 27, 25, 31, 29, 32, 26, 27, 30, 28, 29, 22, 21, 26, 25, 26, 24, 29, 26, 27],
                    [40, 40, 36, 52, 35, 33, 32, 42, 38, 38, 34, 41, 44, 46, 25, 28, 31, 36, 26, 38, 35]),
    "Barcelona": ([82, 76, 67, 87, 99, 96, 91, 100, 87, 94, 91, 90, 93, 87, 82, 79, 73, 88, 85, 88, 94],
                  [25, 22, 19, 27, 31, 30, 28, 32, 27, 30, 29, 28, 28, 26, 25, 24, 21, 28, 26, 28, 31],
                  [33, 33, 35, 35, 24, 21, 29, 40, 33, 21, 29, 37, 29, 36, 38, 38, 38, 20, 44, 39, 36]),
}
# Mejor fase alcanzada: 0 sin participación · 1 grupos/fase liga · 2 dieciseisavos · 3 octavos · 4 cuartos · 5 semis · 6 final · 7 campeón
EURO = {"Champions": [0, 0, 0, 3, 1, 0, 0, 0, 6, 4, 6, 5, 1, 3, 4, 3, 4, 1, 4, 3, 5],
        "Europa League": [0, 0, 2, 0, 7, 1, 7, 2, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0]}
PHASE = {0: "—", 1: "Grupos", 2: "1/16", 3: "Octavos", 4: "Cuartos", 5: "Semis", 6: "Final", 7: "CAMPEÓN"}
# Límite de Coste de Plantilla Deportiva (LaLiga, M€). Completar el resto desde la fuente oficial.
LCPD = {"2017/18": 237.8, "2018/19": 293.0, "2019/20": 348.0, "2020/21": 252.7, "2025/26": 327.0}

df = pd.DataFrame(ATM).assign(yr=range(2005, 2026), season=SEASONS)
df = df.assign(l=38 - df.w - df.d, gd=df.gf - df.ga, ppg=df.pts / 38, win=df.w / 38 * 100, ga_pg=df.ga / 38,
               stage=np.select([df.yr < 2011, df.yr == 2011], ["Antes de Simeone", "Transición"], "Era Simeone"))
pre, cho = df[df.stage == "Antes de Simeone"], df[df.stage == "Era Simeone"]
STAGE_C = {"Antes de Simeone": GREY, "Transición": WHITE, "Era Simeone": RED}

rv = pd.concat([pd.DataFrame({"yr": df.yr, "season": SEASONS, "Equipo": t, "Puntos": p,
                              "% victorias": np.array(w) / 38 * 100, "Goles recibidos": g})
                for t, (p, w, g) in {"Atlético": (df.pts, df.w, df.ga), **RIVALS}.items()])
rv["Distancia al campeón"] = rv.groupby("yr").Puntos.transform("max") - rv.Puntos

# ─────────────────────────── HELPERS ───────────────────────────
def style(fig, h=430):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Inter", color=TEXT, size=13), height=h, margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h", y=1.12, x=0), hoverlabel=dict(bgcolor=CARD, font_color=WHITE))
    fig.update_xaxes(showgrid=False, tickvals=list(range(2005, 2026)), ticktext=SEASONS, tickangle=-45)
    fig.update_yaxes(gridcolor="#2E323D", zeroline=False)
    return fig

def cholo(fig):  # franja de la era Simeone + marca de llegada
    fig.add_vrect(x0=2011.5, x1=2025.5, fillcolor=RED, opacity=.06, line_width=0)
    fig.add_vline(x=2011, line=dict(color=RED, dash="dot", width=2),
                  annotation_text="⚡ Llega el Cholo · dic 2011", annotation_font_color=RED)
    return fig

def insight(txt):
    st.markdown(f"<div class='insight'>{txt}</div>", unsafe_allow_html=True)

# ─────────────────────────── SIDEBAR ───────────────────────────
with st.sidebar:
    st.markdown("## 🎛️ Control")
    a, b = st.select_slider("Rango de temporadas", SEASONS, value=(SEASONS[0], SEASONS[-1]))
    f = df[df.yr.between(int(a[:4]), int(b[:4]))]
    st.markdown("## 💶 Coste de plantilla (M€)")
    st.caption("Precargado: LCPD publicado por LaLiga. Editá o completá las celdas vacías.")
    cost = st.data_editor(pd.DataFrame({"Temporada": SEASONS, "Coste": [LCPD.get(s, np.nan) for s in SEASONS]}),
                          hide_index=True, disabled=["Temporada"], height=280,
                          column_config={"Coste": st.column_config.NumberColumn(format="%.1f", min_value=0)})

# ─────────────────────────── HERO + KPIs ───────────────────────────
st.markdown(f"""<div class='top'><div></div>
<div class='titles'><h1 class='hero'><span class='l1'>EL EFECTO</span><span class='l2' data-text='CHOLO'>CHOLO</span></h1>
<div class='rule'></div>
<div class='sub'><span>Atlético de Madrid</span><i></i><span>2005/06 → 2025/26</span><i></i><span>rendimiento vs. dinero</span></div></div>
<div class='crest crest-neon top-neon'><img src='{b64("Atletico_Madrid.png")}' alt='Escudo del Atlético de Madrid'></div></div>
<img class='banner' src='{b64("Simeone-Copas.jpg")}' alt='Simeone y sus títulos con el Atlético'>""",
            unsafe_allow_html=True)

def stat(s, k):
    n = 38 * len(s)
    return {"ppg": s.pts.sum() / n, "win": s.w.sum() / n * 100, "ga": s.ga.sum() / n, "top3": (s.pos <= 3).mean() * 100}[k]

c = st.columns(4)
for col, (k, lab, fmt, unit, inv) in zip(c, [("ppg", "Puntos por partido", "{:.2f}", "", False),
                                            ("win", "% de victorias", "{:.1f}%", " pp", False),
                                            ("ga", "Goles recibidos / partido", "{:.2f}", "", True),
                                            ("top3", "Temporadas en Top 3", "{:.0f}%", " pp", False)]):
    x0, x1 = stat(pre, k), stat(cho, k)
    col.metric(lab, fmt.format(x1), (f"{x1 - x0:+.1f}" if unit else f"{x1 - x0:+.2f}") + f"{unit} vs. antes ({fmt.format(x0)})",
               delta_color="inverse" if inv else "normal")
st.markdown("<p class='note'>Era Simeone (2012/13–2025/26) frente a Antes de Simeone (2005/06–2010/11). 2011/12 queda como transición.</p>", unsafe_allow_html=True)

# ─────────────────────────── TABS ───────────────────────────
t = st.tabs(["⚡ Puntos", "📊 Posiciones", "🛡️ Defensa", "🌍 Europa", "💶 Dinero", "⚔️ Rivales", "🧠 Clustering"])

with t[0]:
    fig = go.Figure(go.Scatter(x=f.yr, y=f.pts, mode="lines+markers", line=dict(color=WHITE, width=2),
                               marker=dict(size=12, color=f.stage.map(STAGE_C), line=dict(color=BLACK, width=2)),
                               customdata=f[["season", "stage"]], hovertemplate="%{customdata[0]}<br>%{y} pts · %{customdata[1]}<extra></extra>"))
    for s, x0, x1 in [(pre, 2005, 2010), (cho, 2012, 2025)]:
        fig.add_shape(type="line", x0=x0, x1=x1, y0=s.pts.mean(), y1=s.pts.mean(), line=dict(color=RED, dash="dash"))
    st.plotly_chart(cholo(style(fig)), width="stretch")
    insight(f"Promedio <b>{pre.pts.mean():.1f} pts</b> antes → <b>{cho.pts.mean():.1f} pts</b> con Simeone "
            f"(<b>+{cho.pts.mean() - pre.pts.mean():.1f}</b> por temporada). Pico: <b>{df.pts.max()} pts</b> en {df.loc[df.pts.idxmax(), 'season']}.")

with t[1]:
    fig = go.Figure(go.Scatter(x=f.yr, y=f.pos, mode="lines+markers+text", text=f.pos.astype(str) + "º",
                               textposition="top center", line=dict(color=RED, width=3, shape="hv"),
                               marker=dict(size=10, color=WHITE)))
    fig.add_hrect(y0=.5, y1=3.5, fillcolor=WHITE, opacity=.05, line_width=0,
                  annotation_text="ZONA TOP 3", annotation_position="top left")
    st.plotly_chart(cholo(style(fig)).update_yaxes(range=[20.5, .5], dtick=2, title="Puesto"), width="stretch")
    insight(f"Antes: <b>{(pre.pos <= 3).sum()}/{len(pre)}</b> temporadas en el podio. Con Simeone: "
            f"<b>{(cho.pos <= 3).sum()}/{len(cho)}</b>, incluidos <b>{(cho.pos == 1).sum()} títulos</b> (2013/14 y 2020/21).")

with t[2]:
    fig = go.Figure([go.Bar(x=f.yr, y=f.ga_pg, name="Goles recibidos / partido", marker_color=f.stage.map(STAGE_C)),
                     go.Scatter(x=f.yr, y=f.gd, name="Diferencia de gol", yaxis="y2", mode="lines+markers",
                                line=dict(color=WHITE, width=2))])
    fig.update_layout(yaxis2=dict(overlaying="y", side="right", showgrid=False, title="DG"), yaxis_title="GC / partido")
    st.plotly_chart(cholo(style(fig)), width="stretch")
    insight(f"Goles recibidos por partido: <b>{stat(pre, 'ga'):.2f}</b> → <b>{stat(cho, 'ga'):.2f}</b>. "
            f"Récord defensivo: <b>{df.ga.min()} goles</b> en {df.loc[df.ga.idxmin(), 'season']}. "
            f"Ojo: las últimas temporadas ({', '.join(cho[cho.ga >= 40].season)}) superan los 40 recibidos.")

with t[3]:
    eu = pd.DataFrame(EURO, index=SEASONS).T.loc[:, a:b]
    fig, ax = plt.subplots(figsize=(14, 2.4)); fig.patch.set_alpha(0)
    sns.heatmap(eu, cmap=LinearSegmentedColormap.from_list("atm", ["#2A2E38", DARK, RED]), vmin=0, vmax=7, cbar=False,
                annot=eu.apply(lambda c: c.map(PHASE)), fmt="", linewidths=2, linecolor=BLACK, ax=ax,
                annot_kws=dict(color=WHITE, fontsize=8, weight="bold"))
    ax.tick_params(colors=TEXT, labelsize=8, length=0); plt.xticks(rotation=45); plt.yticks(rotation=0)
    st.pyplot(fig, width="stretch")
    insight("Antes de Simeone: una Europa League (2009/10) y presencia intermitente. Con él: 14 temporadas seguidas en "
            "Champions, <b>2 finales</b> (2014, 2016), semis en 2016/17 y 2025/26, y la Europa League 2017/18.")

with t[4]:
    m = df.merge(cost.rename(columns={"Temporada": "season"}), on="season").dropna(subset=["Coste"])
    if len(m) < 3:
        st.warning("Cargá al menos 3 temporadas con coste en la barra lateral para trazar la relación.")
    else:
        sl, ic = np.polyfit(m.Coste, m.pts, 1)
        m = m.assign(esperado=sl * m.Coste + ic, ptsxM=m.pts / m.Coste)
        m["Rendimiento"] = np.where(m.pts >= m.esperado, "Por encima de lo esperado", "Por debajo de lo esperado")
        fig = px.scatter(m, x="Coste", y="pts", color="Rendimiento", text="season", size="ptsxM", size_max=28,
                         color_discrete_map={"Por encima de lo esperado": RED, "Por debajo de lo esperado": GREY},
                         hover_data={"esperado": ":.1f", "ptsxM": ":.3f"})
        xs = np.linspace(m.Coste.min(), m.Coste.max(), 50)
        fig.add_scatter(x=xs, y=sl * xs + ic, mode="lines", line=dict(color=WHITE, dash="dot"), name="Tendencia")
        fig = style(fig).update_traces(textposition="top center")
        fig.update_xaxes(tickvals=None, ticktext=None, tickangle=0, title="Coste de plantilla (M€)")
        st.plotly_chart(fig.update_yaxes(title="Puntos en LaLiga"), width="stretch")
        best = m.loc[(m.pts - m.esperado).idxmax()]
        insight(f"Cada 10 M€ extra se asocian a <b>{sl * 10:+.2f} pts</b> (n={len(m)}, correlación r = "
                f"<b>{m.Coste.corr(m.pts):.2f}</b>). Mayor sobre-rendimiento: <b>{best.season}</b> "
                f"({best.pts} pts vs. {best.esperado:.0f} esperados). El LCPD es un tope autorizado, no el gasto real.")

with t[5]:
    met = st.radio("Métrica", ["Puntos", "% victorias", "Goles recibidos", "Distancia al campeón"], horizontal=True)
    r = rv[rv.yr.between(int(a[:4]), int(b[:4]))]
    fig = px.line(r, x="yr", y=met, color="Equipo", markers=True, hover_data={"season": True, "yr": False},
                  color_discrete_map={"Atlético": RED, "Real Madrid": WHITE, "Barcelona": GREY})
    fig.update_traces(line=dict(width=2)).update_traces(selector=dict(name="Atlético"), line=dict(width=4))
    st.plotly_chart(cholo(style(fig)), width="stretch")
    g = rv[rv.Equipo == "Atlético"].set_index("yr")["Distancia al campeón"]
    insight(f"Distancia media del Atlético al campeón: <b>{g.loc[2005:2010].mean():.1f} pts</b> antes → "
            f"<b>{g.loc[2012:2025].mean():.1f} pts</b> con Simeone. Rompió el duopolio 2 veces en 14 temporadas.")

with t[6]:
    k = st.slider("Número de perfiles (K)", 2, 5, 3)
    feats = ["ppg", "win", "ga_pg", "gd", "pos"]
    df["cl"] = KMeans(k, n_init=10, random_state=42).fit_predict(StandardScaler().fit_transform(df[feats]))
    order = df.groupby("cl").ppg.mean().rank(ascending=False).astype(int)
    names = {1: "Élite", 2: "Aspirante", 3: "Irregular", 4: "Crisis", 5: "Fondo"}
    df["Perfil"] = df.cl.map(order).map(lambda i: f"{i}. {names[i]}")
    fig = px.scatter(df, x="ga_pg", y="ppg", color="Perfil", symbol="stage", text="season",
                     category_orders={"Perfil": sorted(df.Perfil.unique())},
                     color_discrete_sequence=[RED, WHITE, GREY, DARK, "#3a3a3a"],
                     labels={"ga_pg": "Goles recibidos / partido", "ppg": "Puntos / partido", "stage": "Etapa"})
    fig = style(fig, 480).update_traces(textposition="top center", marker=dict(size=13, line=dict(color=BLACK, width=1)))
    fig.update_xaxes(tickvals=None, ticktext=None, tickangle=0, autorange="reversed")
    st.plotly_chart(fig, width="stretch")
    st.dataframe(df.groupby("Perfil").agg(Temporadas=("season", ", ".join), Pts=("pts", "mean"), GC=("ga", "mean"))
                 .round(1), width="stretch")
    insight("K-means sobre puntos/partido, % victorias, goles recibidos/partido, diferencia de gol y puesto "
            "(estandarizados). Eje X invertido: cuanto más a la derecha, mejor defensa.")

st.markdown(f"""<div class='gallery'>
<figure style='flex:{ratio("Diego-Simeone-1.png")} 1 34px'><img src='{b64("Diego-Simeone-1.png")}' alt='Diego Simeone celebrando'></figure>
<figure style='flex:{ratio("Foto-Aura-Cholo.jpg")} 1 34px'><img src='{b64("Foto-Aura-Cholo.jpg")}' alt='El aura del Cholo'></figure></div>""",
            unsafe_allow_html=True)
L = {"cla": "https://www.laliga.com/laliga-easports/clasificacion",
     "lcpd": "https://www.laliga.com/transparencia/gestion-economica/limite-coste-plantilla",
     "uefa": "https://www.uefa.com/uefachampionsleague/news/0250-0c50fb933c1b-ee0d8fabd000-1000--club-facts-atletico",
     "atm": "https://www.atleticodemadrid.com/atm/informacion-economica-financiera"}
a_ = lambda k, t: f"<a href='{L[k]}' target='_blank' rel='noopener'>{t}</a>"
st.markdown(f"""<div class='sources'><h4>Fuentes</h4>
<p>{a_("cla", "LaLiga")} (clasificaciones y {a_("lcpd", "límites de coste de plantilla")}) · {a_("uefa", "UEFA")} (historial europeo) ·
{a_("atm", "Atlético de Madrid")} (cuentas anuales).</p></div>
<div class='signature'><div class='crest-neon'><img src='{b64("Atletico_Madrid.png")}' alt='Escudo del Atlético de Madrid'></div>
<p>Análisis de datos hecho por <b>Valentín Gerold</b> en colaboración con la IA</p></div>""", unsafe_allow_html=True)
