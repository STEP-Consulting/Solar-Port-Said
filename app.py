import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- SEITENKONFIGURATION ---
st.set_page_config(page_title="E-Methanol Simulation Port Said", layout="wide")

st.title("☀️ E-Methanol & PV-Energiekopplung: Standort Port Said")
st.markdown(
    "Interaktive Simulation der Tages- und Jahresdynamik zwischen **PV-Erzeugung**, "
    "**PEM-Elektrolyse**, **$\text{H}_2$-Nachtpuffer** und **kontinuierlicher Methanolsynthese**."
)

# --- SIDEBAR: PARAMETER & SCHIEBEREGLER ---
st.sidebar.header("⚙️ Anlagenparameter")

target_methanol_year = st.sidebar.number_input(
    "Zielproduktion Methanol (t/Jahr)", min_value=1000, max_value=50000, value=9000, step=500
)

electrolyzer_mw = st.sidebar.slider(
    "PEM-Elektrolyseur Nennleistung (MW)", min_value=10.0, max_value=45.0, value=26.0, step=1.0
)

pv_oversizing = st.sidebar.slider(
    "PV-Oversizing-Faktor", min_value=1.0, max_value=3.0, value=1.7, step=0.1,
    help="Verhältnis von installierter PV-Spitzenleistung (MWp) zu Elektrolyseur-Leistung (MW)."
)

h2_buffer_hours = st.sidebar.slider(
    "H₂-Pufferkapazität (Stunden Synthesebetrieb)", min_value=4, max_value=36, value=16, step=1
)

st.sidebar.header("🌍 Standort & Wetter (Port Said)")
season = st.sidebar.selectbox(
    "Simulierter Referenztag (Port Said, 31.26° N)",
    options=["Sommer-Peak (Juni)", "Frühjahr / Herbst (März/Sept.)", "Winter-Tief (Dezember)"],
    index=0
)

# --- PHYSIKALISCHE & METEOROLOGISCHE BASISDATEN (PORT SAID) ---
# Typische Volllaststunden und Tageslängen für Port Said (31.26° N)
season_params = {
    "Sommer-Peak (Juni)": {
        "peak_factor": 0.90,       # Max. Leistungsfaktor mittags (Temperaturverluste inkl.)
        "sunrise": 5.0,
        "sunset": 19.0,
        "synthesis_load": 1.15     # Synthese läuft im Sommer leicht über Nennlast (115%)
    },
    "Frühjahr / Herbst (März/Sept.)": {
        "peak_factor": 0.85,
        "sunrise": 6.0,
        "sunset": 18.0,
        "synthesis_load": 1.00     # 100% Nennlast
    },
    "Winter-Tief (Dezember)": {
        "peak_factor": 0.68,       # Tieferer Sonnenstand
        "sunrise": 7.0,
        "sunset": 17.0,
        "synthesis_load": 0.65     # Geplanter Winter-Teillastbetrieb (65%)
    }
}

params = season_params[season]
pv_mwp = electrolyzer_mw * pv_oversizing

# Stöchiometrie & Wirkungsgrade
h2_per_t_methanol = 0.195          # t H2 pro t Methanol
kwh_per_kg_h2 = 52.0               # MWh pro t H2 (inkl. Systemperipherie)
annual_h2_needed = target_methanol_year * h2_per_t_methanol
nominal_h2_rate_t_h = annual_h2_needed / 8760.0  # Kontinuierlicher Bedarf in t/h

# Aktueller H2-Bedarf der Methanolsynthese in der gewählten Saison (t/h)
current_synthesis_h2_demand = nominal_h2_rate_t_h * params["synthesis_load"]

# Puffergröße in Tonnen H2
h2_buffer_capacity_t = nominal_h2_rate_t_h * h2_buffer_hours

# --- 24-STUNDEN SIMULATION (STUNDENBASIS) ---
hours = np.arange(0, 24, 1)
pv_generation = np.zeros(24)
electrolyzer_power = np.zeros(24)
curtailment = np.zeros(24)
h2_produced = np.zeros(24)
h2_storage_level = np.zeros(24)

# Startfüllstand des Speichers um 00:00 Uhr (z.B. 55% gefüllt aus dem Vortag)
current_storage = h2_buffer_capacity_t * 0.55
bottleneck_hours = 0

for h in hours:
    # PV-Glockenkurve für Port Said berechnen
    if params["sunrise"] <= h <= params["sunset"]:
        day_length = params["sunset"] - params["sunrise"]
        solar_angle = np.pi * (h - params["sunrise"]) / day_length
        pv_generation[h] = pv_mwp * params["peak_factor"] * (np.sin(solar_angle) ** 1.3)
    else:
        pv_generation[h] = 0.0

    # Elektrolyseur-Last & Abregelung (Curtailment)
    if pv_generation[h] > electrolyzer_mw:
        electrolyzer_power[h] = electrolyzer_mw
        curtailment[h] = pv_generation[h] - electrolyzer_mw
    else:
        electrolyzer_power[h] = pv_generation[h]
        curtailment[h] = 0.0

    # Produzierter Wasserstoff (in Tonnen)
    h2_produced[h] = electrolyzer_power[h] / kwh_per_kg_h2

    # Speicherbilanz (Zugang durch Elektrolyse - Abgang durch Methanolsynthese)
    net_h2 = h2_produced[h] - current_synthesis_h2_demand
    current_storage += net_h2

    # Begrenzung des Speichers (Min 0, Max Kapazität)
    if current_storage > h2_buffer_capacity_t:
        current_storage = h2_buffer_capacity_t
    elif current_storage <= 0.01:
        current_storage = 0.0
        bottleneck_hours += 1

    h2_storage_level[h] = current_storage

# --- KENNZAHLEN (KPIs) ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Installierte PV-Leistung", f"{pv_mwp:.1f} MWp", f"Faktor {pv_oversizing}x")
col2.metric("H₂-Nachtspeicher", f"{h2_buffer_capacity_t:.2f} t H₂", f"{h2_buffer_hours} Std. Puffer")
col3.metric("Abgeregelte PV-Spitze (Tag)", f"{np.sum(curtailment):.1f} MWh", "Potenzial für SCZone-Netz")
col4.metric(
    "Synthese-Versorgung (24h)",
    "Stabil (24/7)" if bottleneck_hours == 0 else f"Engpass ({bottleneck_hours}h)",
    delta="OK" if bottleneck_hours == 0 else "Speicher zu klein!",
    delta_color="normal" if bottleneck_hours == 0 else "inverse"
)

if bottleneck_hours > 0:
    st.error(
        f"⚠️ **Achtung Bottleneck:** In dieser Konfiguration läuft der H₂-Speicher nachts für **{bottleneck_hours} Stunden** leer. "
        "Die kontinuierliche Methanolsynthese würde abreißen! Erhöhe den **H₂-Puffer**, das **PV-Oversizing** oder reduziere die Winter-Last."
    )
else:
    st.success(
        f"✅ **System stabil:** Die Methanolsynthese läuft über alle 24 Stunden kontinuierlich durch (Last: {int(params['synthesis_load']*100)}%)."
    )

# --- PLOT 1: TAGESPROFIL (PV, ELEKTROLYSE & H2-SPEICHER) ---
fig = make_subplots(specs=[[{"secondary_y": True}]])

# PV-Erzeugung
fig.add_trace(
    go.Scatter(
        x=hours, y=pv_generation, name="PV-Erzeugung (MW)",
        line=dict(color="#FDB813", width=3), fill="tozeroy", fillcolor="rgba(253, 184, 19, 0.2)"
    ),
    secondary_y=False
)

# Elektrolyseur-Leistung
fig.add_trace(
    go.Scatter(
        x=hours, y=electrolyzer_power, name="Elektrolyseur-Last (MW)",
        line=dict(color="#00A8E8", width=3, dash="solid")
    ),
    secondary_y=False
)

# Speicherfüllstand
fig.add_trace(
    go.Scatter(
        x=hours, y=h2_storage_level, name="H₂-Speicherfüllstand (t)",
        line=dict(color="#2ECC71", width=3, dash="dot")
    ),
    secondary_y=True
)

fig.update_layout(
    title=f"24-Stunden-Lastprofil & Speicherverlauf – {season}",
    xaxis_title="Uhrzeit (Stunde des Tages)",
    hovermode="x unified",
    height=450,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
fig.update_yaxes(title_text="Elektrische Leistung (MW)", range=[0, pv_mwp * 1.1], secondary_y=False)
fig.update_yaxes(title_text="H₂-Speicherfüllstand (Tonnen)", range=[0, h2_buffer_capacity_t * 1.15], secondary_y=True)

st.plotly_chart(fig, use_container_width=True)

# --- JAHRESBILANZ (12 MONATE PORT SAID) ---
st.subheader("📅 Saisonale Jahresübersicht (Port Said Referenzjahr)")

months = ["Jan", "Feb", "Mär", "Apr", "Mai", "Jun", "Jul", "Aug", "Sep", "Okt", "Nov", "Dez"]
# Spezifischer monatlicher PV-Ertrag in Port Said (kWh/kWp pro Monat, Summe ~ 1.980 kWh/kWp/a)
monthly_spec_yield = np.array([115, 128, 165, 185, 205, 215, 218, 205, 178, 152, 118, 106])
days_in_month = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])

# Effektive Stromnutzung nach ca. 8-12% Abregelung durch Oversizing
effective_factor = np.clip(1.0 - (pv_oversizing - 1.3) * 0.12, 0.65, 1.0)
monthly_pv_mwh = pv_mwp * monthly_spec_yield * effective_factor
monthly_h2_t = monthly_pv_mwh / kwh_per_kg_h2
monthly_methanol_t = monthly_h2_t / h2_per_t_methanol

total_annual_methanol = np.sum(monthly_methanol_t)

df_year = pd.DataFrame({
    "Monat": months,
    "Methanol-Produktion (t)": np.round(monthly_methanol_t, 0),
    "Ziel-Durchschnitt (t)": np.round(target_methanol_year / 12, 0)
})

fig_year = go.Figure()
fig_year.add_trace(go.Bar(
    x=df_year["Monat"], y=df_year["Methanol-Produktion (t)"],
    name="Simulierte Produktion (t/Monat)", marker_color="#0077B6"
))
fig_year.add_trace(go.Scatter(
    x=df_year["Monat"], y=df_year["Ziel-Durchschnitt (t)"],
    name="Monatliches Soll (für 9.000 t/a)", line=dict(color="#E63946", width=2, dash="dash")
))

fig_year.update_layout(
    title=f"Monatliche E-Methanol-Produktion (Jahressumme: {total_annual_methanol:,.0f} t / Ziel: {target_methanol_year:,.0f} t)",
    yaxis_title="Methanol (Tonnen)",
    height=380
)
st.plotly_chart(fig_year, use_container_width=True)
