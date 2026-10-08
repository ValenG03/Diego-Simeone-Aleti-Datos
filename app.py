"""EL EFECTO CHOLO — Atlético de Madrid 2005/06–2025/26
Rendimiento deportivo vs. recursos económicos. Stack: Streamlit · Pandas · Plotly · Seaborn · K-means
"""
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
RED, DARK, BLACK, WHITE, GREY = "#CB3524", "#5C1610", "#0A0A0A", "#F5F5F5", "#7A7A7A"

# ─────────────────────────── ESTILO ───────────────────────────
st.markdown(f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@400;600;800&display=swap');
.stApp {{background:radial-gradient(circle at 85% -10%,#3a0b07 0%,{BLACK} 42%);color:#cfcfcf;font-family:Inter,sans-serif}}
[data-testid="stSidebar"] {{background:#070707;border-right:1px solid {DARK}}}
h1,h2,h3 {{font-family:'Bebas Neue';letter-spacing:2px;color:{WHITE}!important}}
.hero {{font-family:'Bebas Neue';font-size:clamp(3rem,8vw,6rem);line-height:.85;color:{WHITE};margin:0}}
.hero span {{color:{RED};text-shadow:0 0 22px {RED}aa}}
.sub {{color:{GREY};letter-spacing:3px;text-transform:uppercase;font-size:.8rem;margin-bottom:1.2rem}}
[data-testid="stMetric"] {{background:#0f0f0f;border-left:4px solid {RED};border-radius:6px;padding:14px 18px;
  box-shadow:0 0 0 1px #1c1c1c;transition:.25s}}
[data-testid="stMetric"]:hover {{box-shadow:0 0 24px {RED}55;transform:translateY(-3px)}}
[data-testid="stMetricValue"] {{font-family:'Bebas Neue';font-size:2.6rem;color:{WHITE}}}
.stTabs [data-baseweb="tab"] {{font-family:'Bebas Neue';font-size:1.25rem;letter-spacing:1px;color:{GREY}}}
.stTabs [aria-selected="true"] {{color:{RED}!important}}
.stTabs [data-baseweb="tab-highlight"] {{background:{RED}}}
.insight {{border:1px solid {DARK};background:linear-gradient(90deg,{RED}22,transparent);padding:10px 16px;
  border-radius:6px;margin-top:.5rem;font-size:.92rem}}
.insight b {{color:{WHITE}}}
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
                      font=dict(family="Inter", color="#cfcfcf"), height=h, margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h", y=1.12, x=0), hoverlabel=dict(bgcolor="#111", font_color=WHITE))
    fig.update_xaxes(showgrid=False, tickvals=list(range(2005, 2026)), ticktext=SEASONS, tickangle=-45)
    fig.update_yaxes(gridcolor="#1f1f1f", zeroline=False)
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
st.markdown("<p class='hero'>EL EFECTO <span>CHOLO</span></p>"
            "<p class='sub'>Atlético de Madrid · 2005/06 → 2025/26 · rendimiento vs. dinero</p>", unsafe_allow_html=True)

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
st.caption("Era Simeone (2012/13–2025/26) frente a Antes de Simeone (2005/06–2010/11). 2011/12 queda como transición.")

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
    fig, ax = plt.subplots(figsize=(14, 2.4), facecolor=BLACK)
    sns.heatmap(eu, cmap=LinearSegmentedColormap.from_list("atm", ["#141414", DARK, RED]), vmin=0, vmax=7, cbar=False,
                annot=eu.apply(lambda c: c.map(PHASE)), fmt="", linewidths=2, linecolor=BLACK, ax=ax,
                annot_kws=dict(color=WHITE, fontsize=8, weight="bold"))
    ax.tick_params(colors="#cfcfcf", labelsize=8, length=0); plt.xticks(rotation=45); plt.yticks(rotation=0)
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

st.markdown("---")
st.caption("Fuentes: LaLiga (clasificaciones y límites de coste de plantilla) · UEFA (historial europeo) · "
           "Atlético de Madrid (cuentas anuales). Datos precargados: validar contra las fuentes oficiales antes de publicar.")
