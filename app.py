import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- SEITENKONFIGURATION ---
st.set_page_config(page_title="E-Methanol Simulation Port Said", layout="wide")

st.title("☀️ E-Methanol & PV-Energiekopplung: Standort Port Said")
st.markdown(
    "Interaktive Simulation auf Basis realer **PVGIS-TMY-Daten** für Port Said (31,26° N, 32,30° O): "
    "Automatische Dimensionierung von **PV-Oversizing** und **PV-Fläche** (inkl. +30 % Zuschlag) aus der Ziel-Methanolproduktion, "
    "**24h-Ansicht**, **8.760h-Jahresverlauf** sowie **stöchiometrischer Stoff- & Energiebilanz**."
)

# --- PVGIS- & FLÄCHEN-KONSTANTEN ---
STC_IRRADIANCE_W_M2 = 1000.0
DEFAULT_PR = 0.82
PVGIS_CSV_PATH = "pvgis_port_said.csv"  # Ggf. anpassen, z. B. "data/pvgis_port_said.csv"

# Standard-PV-Basisfläche (ha / MWp) + pauschal +30 % für Gänge, Abstände, Konstellation & Begleitbauwerke
BASE_PV_AREA_HA_PER_MWP = 0.60
AREA_OVERHEAD_FACTOR = 1.30

# --- PHYSIKALISCHE & STÖCHIOMETRISCHE BASISDATEN ---
STOICH_H2_PER_T_MEOH = 0.195        # t H2 / t CH3OH
STOICH_CO2_PER_T_MEOH = 1.420       # t CO2 / t CH3OH
STOICH_H2O_ELEC_PER_T_H2 = 9.5      # t entmineralisiertes Wasser / t H2
STOICH_H2O_SYNTH_BYPROD = 0.562     # t Reaktionswasser als Nebenprodukt / t CH3OH
STOICH_O2_BYPROD_PER_T_H2 = 7.94    # t O2-Nebenprodukt aus Elektrolyse / t H2

KWH_PER_KG_H2 = 52.0                # MWh pro t H2 (inkl. Systemperipherie)


@st.cache_data
def load_pvgis_tmy_pv_mw(
    csv_path: str,
    pv_mwp_val: float,
    pr: float = DEFAULT_PR,
    clean_daytime_zeros: bool = True,
) -> np.ndarray:
    """
    Liest eine PVGIS-TMY-CSV (17 Kopf-/Monatszeilen) in ihrer TMY-Monatsreihenfolge
    (Jan -> Dez, 8.760 h) ein, bereinigt optional unplausible 0.0-Werte in den
    Kern-Tagesstunden (07–16h UTC) und liefert stündliche PV-Leistung in MW.
    """
    df = pd.read_csv(csv_path, skiprows=17)

    if "G(h)" not in df.columns:
        raise ValueError(f"Spalte 'G(h)' nicht in '{csv_path}' gefunden. Vorhanden: {list(df.columns)}")

    df = df.iloc[:8760].copy()
    dt_series = pd.to_datetime(df["time(UTC)"], format="%Y%m%d:%H%M", errors="coerce")

    irradiance = pd.to_numeric(df["G(h)"], errors="coerce")

    if clean_daytime_zeros:
        hour_utc = dt_series.dt.hour
        daytime_zero_mask = (hour_utc >= 7) & (hour_utc <= 16) & (irradiance.fillna(0.0) <= 0.0)
        irradiance = irradiance.mask(daytime_zero_mask)

        if irradiance.isna().any():
            irradiance = (
                irradiance.interpolate(method="linear", limit_direction="both")
                .fillna(0.0)
            )

    irradiance_arr = np.clip(irradiance.fillna(0.0).to_numpy(dtype=float), 0.0, None)

    if len(irradiance_arr) < 8760:
        raise ValueError(f"Zu wenige Stundenwerte in '{csv_path}' ({len(irradiance_arr)} statt 8.760).")

    return pv_mwp_val * (irradiance_arr[:8760] / STC_IRRADIANCE_W_M2) * pr


# --- SIDEBAR: PARAMETER & SCHIEBEREGLER ---
st.sidebar.header("⚙️ Anlagenparameter")

target_methanol_year = st.sidebar.number_input(
    "Zielproduktion Methanol (t/Jahr)",
    min_value=1000,
    max_value=100000,
    value=9000,
    step=500,
)

electrolyzer_mw = st.sidebar.slider(
    "PEM-Elektrolyseur Nennleistung (MW)",
    min_value=5.0,
    max_value=60.0,
    value=26.0,
    step=1.0,
)

h2_buffer_hours = st.sidebar.slider(
    "H₂-Pufferkapazität (Stunden Nenn-Synthese)",
    min_value=4,
    max_value=72,
    value=16,
    step=1,
)

st.sidebar.header("🌍 Standort & Wetter (Port Said)")
season = st.sidebar.selectbox(
    "Referenztag aus PVGIS (24h-Ansicht)",
    options=["Sommer (Juni – Tag 166)", "Übergang (September – Tag 258)", "Winter (Dezember – Tag 349)"],
    index=0,
)

nominal_h2_rate_t_h = (target_methanol_year * STOICH_H2_PER_T_MEOH) / 8760.0
h2_buffer_capacity_t = nominal_h2_rate_t_h * h2_buffer_hours


# --- JAHRESSIMULATION (8.760 STUNDEN) AUF BASIS VON PVGIS ---
@st.cache_data
def simulate_full_year(pv_mwp_val, el_mw_val, synth_h2_rate_t_h, buffer_cap_t, kwh_per_kg, csv_path):
    hours_year = np.arange(8760)
    day_of_year = hours_year // 24

    pv_mw = load_pvgis_tmy_pv_mw(csv_path, pv_mwp_val, pr=DEFAULT_PR, clean_daytime_zeros=True)

    el_mw = np.minimum(pv_mw, el_mw_val)
    curtail_mw = np.maximum(pv_mw - el_mw_val, 0.0)
    h2_prod_t = el_mw / kwh_per_kg

    h2_used_t = np.full(8760, synth_h2_rate_t_h)

    storage_t = np.zeros(8760)
    cur_buf = buffer_cap_t * 0.6
    shortfall_h2_t = np.zeros(8760)

    for i in range(8760):
        delta = h2_prod_t[i] - h2_used_t[i]
        cur_buf += delta
        if cur_buf > buffer_cap_t:
            cur_buf = buffer_cap_t
        elif cur_buf < 0.0:
            shortfall_h2_t[i] = -cur_buf
            cur_buf = 0.0
        storage_t[i] = cur_buf

    effective_h2_used_t = np.maximum(h2_used_t - shortfall_h2_t, 0.0)
    methanol_prod_t = effective_h2_used_t / STOICH_H2_PER_T_MEOH

    return pd.DataFrame({
        "hour": hours_year,
        "day": day_of_year,
        "pv_mw": pv_mw,
        "el_mw": el_mw,
        "curtail_mw": curtail_mw,
        "h2_prod_t": h2_prod_t,
        "h2_used_t": effective_h2_used_t,
        "storage_t": storage_t,
        "methanol_t": methanol_prod_t,
        "shortfall_h": (shortfall_h2_t > 1e-6).astype(int),
    })


@st.cache_data
def find_required_pv_mwp(
    target_meoh_t: float,
    el_mw_val: float,
    synth_h2_rate_t_h: float,
    buffer_cap_t: float,
    kwh_per_kg: float,
    csv_path: str,
) -> float:
    """Ermittelt die benötigte PV-Leistung (MWp), um target_meoh_t/a mit PVGIS zu erreichen."""
    candidates_mwp = np.linspace(el_mw_val * 0.5, el_mw_val * 4.0, 141)
    best_mwp = float(candidates_mwp[-1])

    for cand_mwp in candidates_mwp:
        df_test = simulate_full_year(
            float(cand_mwp), el_mw_val, synth_h2_rate_t_h, buffer_cap_t, kwh_per_kg, csv_path
        )
        if df_test["methanol_t"].sum() >= target_meoh_t:
            best_mwp = float(cand_mwp)
            break

    return best_mwp


try:
    pv_mwp = find_required_pv_mwp(
        target_methanol_year,
        electrolyzer_mw,
        nominal_h2_rate_t_h,
        h2_buffer_capacity_t,
        KWH_PER_KG_H2,
        PVGIS_CSV_PATH,
    )
    pv_oversizing = pv_mwp / electrolyzer_mw
    pv_area_ha = pv_mwp * BASE_PV_AREA_HA_PER_MWP * AREA_OVERHEAD_FACTOR
    pv_area_km2 = pv_area_ha / 100.0

    df_year_sim = simulate_full_year(
        pv_mwp,
        electrolyzer_mw,
        nominal_h2_rate_t_h,
        h2_buffer_capacity_t,
        KWH_PER_KG_H2,
        PVGIS_CSV_PATH,
    )
except Exception as exc:
    st.error(f"❌ PVGIS-Daten konnten nicht geladen werden ('{PVGIS_CSV_PATH}'): {exc}")
    st.stop()

# --- AUSLEGUNGS-ÜBERSICHT (AUTOMATISCH BERECHNET) ---
k1, k2, k3, k4 = st.columns(4)
k1.metric("Ziel-Methanolproduktion", f"{target_methanol_year:,.0f} t/a", "Feste Auslegungsgröße")
k2.metric("Benötigte PV-Leistung", f"{pv_mwp:.1f} MWp", f"Oversizing: {pv_oversizing:.2f}x")
k3.metric("Benötigte PV-Fläche", f"{pv_area_ha:,.1f} ha", f"{pv_area_km2:,.2f} km² (inkl. +30 %)")
k4.metric("H₂-Pufferkapazität", f"{h2_buffer_capacity_t:.2f} t H₂", f"{h2_buffer_hours} Std. Nenn-Synthese")

# --- TABS FÜR TAG, JAHR UND STOFFBILANZ ---
tab_day, tab_year, tab_stoich = st.tabs([
    "🕒 Tagesansicht (24h)",
    "📆 Ganzes Jahr (8.760h & Monate)",
    "🧪 Stöchiometrische Stoff- & Energiebilanz",
])

# ==============================================================================
# TAB 1: TAGESANSICHT (24h AUS PVGIS-JAHRESDATEN)
# ==============================================================================
with tab_day:
    day_lookup = {
        "Sommer (Juni – Tag 166)": 165,
        "Übergang (September – Tag 258)": 257,
        "Winter (Dezember – Tag 349)": 348,
    }
    selected_day = day_lookup[season]
    df_day = df_year_sim[df_year_sim["day"] == selected_day].reset_index(drop=True)

    hours = np.arange(0, 24, 1)
    pv_generation = df_day["pv_mw"].to_numpy()
    electrolyzer_power = df_day["el_mw"].to_numpy()
    curtailment = df_day["curtail_mw"].to_numpy()
    h2_storage_level = df_day["storage_t"].to_numpy()
    bottleneck_hours = int(df_day["shortfall_h"].sum())

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Ausgelegte PV-Leistung", f"{pv_mwp:.1f} MWp", f"Faktor {pv_oversizing:.2f}x")
    col2.metric("Benötigte PV-Fläche", f"{pv_area_ha:,.1f} ha", f"{pv_area_km2:,.2f} km² (inkl. +30 %)")
    col3.metric("Abgeregelte PV-Spitze (Tag)", f"{np.sum(curtailment):.1f} MWh", "PVGIS-Tageswert")
    col4.metric(
        "Synthese-Versorgung (24h)",
        "Stabil (24/7)" if bottleneck_hours == 0 else f"Engpass ({bottleneck_hours}h)",
        delta="OK" if bottleneck_hours == 0 else "Speicher leer!",
        delta_color="normal" if bottleneck_hours == 0 else "inverse",
    )

    if bottleneck_hours > 0:
        st.error(
            f"⚠️ **Speicher-Engpass am gewählten Tag:** Der H₂-Puffer reicht an diesem PVGIS-Tag für **{bottleneck_hours} Stunden** nicht aus."
        )
    else:
        st.success("✅ **Stabile Versorgung:** Die H₂-Versorgung der Synthese läuft an diesem Tag ohne Engpass durch.")

    fig_day = make_subplots(specs=[[{"secondary_y": True}]])
    fig_day.add_trace(
        go.Scatter(
            x=hours, y=pv_generation, name="PV-Erzeugung (MW)",
            line=dict(color="#FDB813", width=3), fill="tozeroy", fillcolor="rgba(253, 184, 19, 0.2)"
        ),
        secondary_y=False,
    )
    fig_day.add_trace(
        go.Scatter(
            x=hours, y=electrolyzer_power, name="Elektrolyseur-Last (MW)",
            line=dict(color="#00A8E8", width=3)
        ),
        secondary_y=False,
    )
    fig_day.add_trace(
        go.Scatter(
            x=hours, y=h2_storage_level, name="H₂-Speicherfüllstand (t)",
            line=dict(color="#2ECC71", width=3, dash="dot")
        ),
        secondary_y=True,
    )
    fig_day.update_layout(
        title=f"24-Stunden-Lastprofil & Speicherverlauf (PVGIS) – {season}",
        xaxis_title="Uhrzeit (Stunde des Tages)",
        hovermode="x unified",
        height=440,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig_day.update_yaxes(title_text="Elektrische Leistung (MW)", range=[0, max(pv_mwp * 1.1, 1.0)], secondary_y=False)
    fig_day.update_yaxes(title_text="H₂-Speicherfüllstand (Tonnen)", range=[0, max(h2_buffer_capacity_t * 1.15, 0.5)], secondary_y=True)
    st.plotly_chart(fig_day, use_container_width=True)

# ==============================================================================
# TAB 2: GANZES JAHR (8.760h & MONATSÜBERSICHT)
# ==============================================================================
with tab_year:
    total_pv_gwh = df_year_sim["pv_mw"].sum() / 1000.0
    total_curtail_gwh = df_year_sim["curtail_mw"].sum() / 1000.0
    total_meoh_year = df_year_sim["methanol_t"].sum()
    total_shortfall_hours = int(df_year_sim["shortfall_h"].sum())

    y1, y2, y3, y4 = st.columns(4)
    y1.metric("Jahres-PV-Ertrag", f"{total_pv_gwh:.1f} GWh", f"davon {total_curtail_gwh:.1f} GWh abgeregelt")
    y2.metric("Simulierte Methanol-Produktion", f"{total_meoh_year:,.0f} t/a", f"Ziel: {target_methanol_year:,.0f} t/a")
    y3.metric("Zielerreichung", f"{(total_meoh_year / target_methanol_year) * 100:.1f} %")
    y4.metric("Speicher-Engpassstunden / Jahr", f"{total_shortfall_hours} h", "von 8.760 h/a")

    df_daily = df_year_sim.groupby("day", as_index=False).agg({
        "pv_mw": "sum",
        "el_mw": "sum",
        "curtail_mw": "sum",
        "storage_t": "mean",
        "methanol_t": "sum",
    })
    df_daily["day_label"] = df_daily["day"] + 1

    month_start_days = [1, 60, 121, 182, 244, 305]
    month_names = ["Jan", "Mär", "Mai", "Jul", "Sep", "Nov"]

    fig_365 = make_subplots(specs=[[{"secondary_y": True}]])
    fig_365.add_trace(
        go.Scatter(
            x=df_daily["day_label"], y=df_daily["pv_mw"],
            name="Täglicher PV-Ertrag (MWh/Tag)",
            line=dict(color="#FDB813", width=2), fill="tozeroy", fillcolor="rgba(253, 184, 19, 0.18)"
        ),
        secondary_y=False,
    )
    fig_365.add_trace(
        go.Scatter(
            x=df_daily["day_label"], y=df_daily["el_mw"],
            name="Genutzter Elektrolyseur-Strom (MWh/Tag)",
            line=dict(color="#00A8E8", width=2)
        ),
        secondary_y=False,
    )
    fig_365.add_trace(
        go.Scatter(
            x=df_daily["day_label"], y=df_daily["methanol_t"],
            name="Tägliche Methanol-Produktion (t/Tag)",
            line=dict(color="#2ECC71", width=2.5)
        ),
        secondary_y=True,
    )
    fig_365.update_layout(
        title="Ganzjahresverlauf (365 Tage, PVGIS): Tages-Energieertrag & tägliche E-Methanol-Produktion",
        xaxis_title="Monat",
        hovermode="x unified",
        height=440,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig_365.update_xaxes(
        tickmode="array",
        tickvals=month_start_days,
        ticktext=month_names,
    )
    fig_365.update_yaxes(title_text="Tagesenergie (MWh/Tag)", secondary_y=False)
    fig_365.update_yaxes(title_text="Methanol-Produktion (t/Tag)", secondary_y=True)
    st.plotly_chart(fig_365, use_container_width=True)

    months = ["Jan", "Feb", "Mär", "Apr", "Mai", "Jun", "Jul", "Aug", "Sep", "Okt", "Nov", "Dez"]
    month_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    month_idx = np.repeat(np.arange(12), month_days)
    df_year_sim["month"] = month_idx[df_year_sim["day"]]

    df_month = df_year_sim.groupby("month", as_index=False).agg({
        "methanol_t": "sum",
        "curtail_mw": lambda s: s.sum() / 1000.0,
    })
    df_month["Monat"] = months

    fig_m = go.Figure()
    fig_m.add_trace(go.Bar(
        x=df_month["Monat"], y=df_month["methanol_t"],
        name="Simulierte Produktion (t/Monat)", marker_color="#0077B6"
    ))
    fig_m.add_trace(go.Scatter(
        x=df_month["Monat"], y=[target_methanol_year / 12.0] * 12,
        name="Monatliches Soll (Ziel / 12)", line=dict(color="#E63946", width=2, dash="dash")
    ))
    fig_m.update_layout(
        title="Monatliche E-Methanol-Produktion über das gesamte Jahr",
        yaxis_title="E-Methanol (Tonnen)",
        height=370,
    )
    st.plotly_chart(fig_m, use_container_width=True)

# ==============================================================================
# TAB 3: STÖCHIOMETRISCHE STOFF- & ENERGIEBILANZ
# ==============================================================================
with tab_stoich:
    st.subheader("🧪 Stöchiometrische & prozesstechnische Stoff- und Energiebilanz")
    st.markdown(
        "Berechnungsgrundlage für die katalytische E-Methanol-Synthese ($\\text{CO}_2 + 3\\,\\text{H}_2 \\rightarrow \\text{CH}_3\\text{OH} + \\text{H}_2\\text{O}$) "
        "und PEM-Wasserelektrolyse ($3\\,\\text{H}_2\\text{O} \\rightarrow 3\\,\\text{H}_2 + 1{,}5\\,\\text{O}_2$). "
        "Die Spalten **Tag** und **Jahr** beziehen sich auf die **simulierte Jahresproduktion** auf Basis der PVGIS-Daten."
    )

    sim_meoh_year = float(df_year_sim["methanol_t"].sum())
    sim_meoh_day = sim_meoh_year / 365.0
    sim_h2_year = float(df_year_sim["h2_used_t"].sum())
    sim_h2_day = sim_h2_year / 365.0
    sim_el_mwh_year = float(df_year_sim["el_mw"].sum())
    sim_el_mwh_day = sim_el_mwh_year / 365.0

    h2_per_t_meoh = STOICH_H2_PER_T_MEOH
    co2_per_t_meoh = STOICH_CO2_PER_T_MEOH
    h2o_elec_per_t_meoh = h2_per_t_meoh * STOICH_H2O_ELEC_PER_T_H2
    el_mwh_per_t_meoh = h2_per_t_meoh * KWH_PER_KG_H2
    o2_byprod_per_t_meoh = h2_per_t_meoh * STOICH_O2_BYPROD_PER_T_H2
    h2o_byprod_per_t_meoh = STOICH_H2O_SYNTH_BYPROD

    rows = [
        {
            "Stoff / Energieform": "E-Methanol (CH₃OH) – Produkt",
            "Rolle im Prozess": "Hauptprodukt",
            "Je t Methanol": "1,000 t",
            "Ø pro Tag (simuliert)": f"{sim_meoh_day:,.2f} t/d",
            "Pro Jahr (simuliert)": f"{sim_meoh_year:,.0f} t/a",
        },
        {
            "Stoff / Energieform": "Wasserstoff (H₂)",
            "Rolle im Prozess": "Edukt (Elektrolyse → Synthese)",
            "Je t Methanol": f"{h2_per_t_meoh:.3f} t ({h2_per_t_meoh*1000:.1f} kg)",
            "Ø pro Tag (simuliert)": f"{sim_h2_day:,.2f} t/d",
            "Pro Jahr (simuliert)": f"{sim_h2_year:,.0f} t/a",
        },
        {
            "Stoff / Energieform": "Kohlenstoffdioxid (CO₂)",
            "Rolle im Prozess": "Edukt (Meerwasser-Abscheidung)",
            "Je t Methanol": f"{co2_per_t_meoh:.3f} t",
            "Ø pro Tag (simuliert)": f"{sim_meoh_day * co2_per_t_meoh:,.2f} t/d",
            "Pro Jahr (simuliert)": f"{sim_meoh_year * co2_per_t_meoh:,.0f} t/a",
        },
        {
            "Stoff / Energieform": "Entmineralisiertes Wasser (H₂O)",
            "Rolle im Prozess": "Edukt (Elektrolyse-Speisewasser)",
            "Je t Methanol": f"{h2o_elec_per_t_meoh:.2f} t",
            "Ø pro Tag (simuliert)": f"{sim_meoh_day * h2o_elec_per_t_meoh:,.1f} t/d",
            "Pro Jahr (simuliert)": f"{sim_meoh_year * h2o_elec_per_t_meoh:,.0f} t/a",
        },
        {
            "Stoff / Energieform": "Elektrische Energie (Elektrolyse)",
            "Rolle im Prozess": "Energie-Input (ca. 52 kWh/kg H₂)",
            "Je t Methanol": f"{el_mwh_per_t_meoh:.2f} MWh",
            "Ø pro Tag (simuliert)": f"{sim_el_mwh_day:,.1f} MWh/d",
            "Pro Jahr (simuliert)": f"{sim_el_mwh_year / 1000.0:,.2f} GWh/a",
        },
        {
            "Stoff / Energieform": "Sauerstoff (O₂)",
            "Rolle im Prozess": "Nebenprodukt (aus Elektrolyse)",
            "Je t Methanol": f"{o2_byprod_per_t_meoh:.2f} t",
            "Ø pro Tag (simuliert)": f"{sim_meoh_day * o2_byprod_per_t_meoh:,.1f} t/d",
            "Pro Jahr (simuliert)": f"{sim_meoh_year * o2_byprod_per_t_meoh:,.0f} t/a",
        },
        {
            "Stoff / Energieform": "Reaktionswasser (H₂O)",
            "Rolle im Prozess": "Nebenprodukt (aus Synthese)",
            "Je t Methanol": f"{h2o_byprod_per_t_meoh:.2f} t",
            "Ø pro Tag (simuliert)": f"{sim_meoh_day * h2o_byprod_per_t_meoh:,.1f} t/d",
            "Pro Jahr (simuliert)": f"{sim_meoh_year * h2o_byprod_per_t_meoh:,.0f} t/a",
        },
    ]

    df_stoich = pd.DataFrame(rows)
    st.dataframe(df_stoich, use_container_width=True, hide_index=True)

    st.info(
        "💡 **Hinweis für die Auslegung in Port Said:** Das bei der Methanolsynthese entstehende Reaktionswasser "
        "kann nach Aufbereitung oft anteilig wieder in die Wasseraufbereitung/Elektrolyse zurückgeführt werden; "
        "der abgeschiedene Sauerstoff ($\\text{O}_2$) fällt an der Küste meist als Nebenprodukt an (ggf. industriell nutzbar)."
    )
