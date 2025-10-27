import io
import math
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objs as go
from reportlab.lib.pagesizes import landscape, A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

# Custom CSS for fonts and colors
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Aileron:wght@300&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Museo+Moderno:wght@900&display=swap');
    
    /* Main title and headers */
    h1, h2, h3 {
        font-family: 'Museo Moderno', sans-serif !important;
        font-weight: 900 !important;
        color: #222222 !important;
    }
    
    /* Body text */
    p, div, span, label, .stMarkdown {
        font-family: 'Aileron', sans-serif !important;
        font-weight: 300 !important;
        color: #222222 !important;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-family: 'Museo Moderno', sans-serif !important;
        font-weight: 900 !important;
        font-size: 1.8rem !important;
        color: #222222 !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-family: 'Aileron', sans-serif !important;
        font-weight: 300 !important;
        color: #222222 !important;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #FF8C00 !important;
        color: #f1f0e6 !important;
        font-family: 'Museo Moderno', sans-serif !important;
        font-weight: 900 !important;
        border: 2px solid #222222 !important;
    }
    
    .stButton > button:hover {
        background-color: #222222 !important;
        color: #FF8C00 !important;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #f1f0e6 !important;
    }
    
    /* Main background */
    .main {
        background-color: #f1f0e6 !important;
    }
    
    /* Sliders */
    .stSlider > div > div > div {
        background-color: #FF8C00 !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #222222 !important;
        color: #f1f0e6 !important;
        font-family: 'Museo Moderno', sans-serif !important;
        font-weight: 900 !important;
        border-radius: 8px 8px 0 0;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #FF8C00 !important;
        color: #222222 !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        font-family: 'Museo Moderno', sans-serif !important;
        font-weight: 900 !important;
        color: #222222 !important;
        background-color: #FF8C00 !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------- Helper functions / Physics ----------

G = 9.80665  # m/s^2

def air_density(temperature_c, pressure_kpa):
    """Compute air density (kg/m^3) from T (°C) and p (kPa) using ideal gas law."""
    p_pa = pressure_kpa * 1000.0
    T_k = temperature_c + 273.15
    R = 287.05  # J/(kg·K)
    return p_pa / (R * T_k)

def co2_thrust_adjusted(base_thrust, temperature_c):
    """
    HOTTER temperature = higher CO₂ pressure = MORE thrust
    Reference temperature: 20°C (293.15K)
    """
    T_ref = 293.15  # 20°C in Kelvin
    T_actual = temperature_c + 273.15
    return base_thrust * (T_actual / T_ref)

def co2_thrust_profile(t, peak_thrust, duration, decay_rate=5.0):
    """
    CO₂ cartridge thrust profile with exponential decay
    """
    if t <= duration:
        return peak_thrust * np.exp(-(decay_rate / duration) * t)
    return 0.0

def simulate_run(
    mass_kg,
    Cd,
    area_m2,
    Crr,
    wheel_friction,
    track_length_m,
    time_step,
    enable_drag,
    enable_rolling,
    rho,
    co2_thrust,
    co2_duration,
    temperature_c,
    bearing_quality,
    wheel_diameter_mm,
    launch_technique,
    max_time=30.0
):
    """
    Advanced physics simulation with CO₂ thrust profile and bearing friction
    """
    dt = time_step
    t = 0.0
    v = 0.0
    x = 0.0
    rows = []

    # Calculate temperature-adjusted peak thrust
    peak_thrust = co2_thrust_adjusted(co2_thrust, temperature_c)
    
    # Bearing friction force (1=excellent, 5=poor)
    bearing_friction_force = 0.05 * bearing_quality
    
    # Wheel diameter effect on rolling resistance (smaller wheels = slightly more resistance)
    wheel_factor = 1.0 + (0.025 - wheel_diameter_mm / 1000) * 2

    while (t < max_time) and (x < track_length_m):
        # Calculate forces
        
        # 1. CO₂ thrust with time-varying profile
        if launch_technique == "Standard":
            F_thrust = co2_thrust_profile(t, peak_thrust, co2_duration, decay_rate=5.0)
        elif launch_technique == "Quick Release":
            F_thrust = co2_thrust_profile(t, peak_thrust * 1.2, co2_duration * 0.7, decay_rate=8.0)
        else:  # Gradual Release
            F_thrust = co2_thrust_profile(t, peak_thrust * 0.9, co2_duration * 1.3, decay_rate=3.0)
        
        # 2. Aerodynamic drag (enhanced impact)
        if enable_drag:
            F_drag = 15.0 * 0.5 * rho * Cd * area_m2 * (v ** 2)
        else:
            F_drag = 0.0
        
        # 3. Rolling resistance with wheel effects
        if enable_rolling:
            F_rolling = Crr * mass_kg * G * wheel_friction * wheel_factor
        else:
            F_rolling = 0.0
        
        # 4. Bearing friction
        F_bearing = bearing_friction_force
        
        # Net force
        F_net = F_thrust - (F_drag + F_rolling + F_bearing)
        
        # Acceleration: F = ma → a = F/m
        a = F_net / mass_kg
        
        # Update velocity
        v = v + a * dt
        if v < 0:
            v = 0
        
        # Update position
        x = x + v * dt
        
        # Update time
        t = t + dt
        
        # Record state
        rows.append({
            "t": round(t, 6),
            "x": round(x, 4),
            "v": round(v, 4),
            "speed_kmh": round(v * 3.6, 4),
            "a": round(a, 4),
            "F_thrust": round(F_thrust, 4),
            "F_drag": round(F_drag, 6),
            "F_rolling": round(F_rolling, 6),
            "F_bearing": round(F_bearing, 6),
            "F_net": round(F_net, 4)
        })

    finish_time = rows[-1]["t"] if rows else 0.0
    return rows, finish_time

# ---------- Plotting ----------

def build_plotly_time_series(rows):
    df = pd.DataFrame(rows)
    if df.empty:
        return None, None, None, None, None
    
    # 1. Speed vs Time
    fig_speed = go.Figure()
    fig_speed.add_trace(go.Scatter(
        x=df["t"], 
        y=df["speed_kmh"], 
        mode="lines", 
        name="Speed",
        line=dict(color='#FF8C00', width=4),
        fill='tozeroy',
        fillcolor='rgba(255, 140, 0, 0.2)'
    ))
    fig_speed.update_layout(
        title="<b>Speed Over Time</b>", 
        xaxis_title="Time (s)", 
        yaxis_title="Speed (km/h)", 
        height=400,
        template='plotly_white',
        font=dict(size=14, family='Aileron, sans-serif', color='#222222'),
        title_font=dict(family='Museo Moderno, sans-serif', size=18, color='#222222'),
        paper_bgcolor='#f1f0e6',
        plot_bgcolor='white'
    )
    
    # 2. Force Breakdown - Normalized view
    fig_forces = go.Figure()
    
    # Sample for cleaner visualization
    sample_df = df.iloc[::20].copy()
    
    # Show forces as percentages of total force for better visualization
    total_force = sample_df["F_thrust"] + sample_df["F_drag"] + sample_df["F_rolling"] + sample_df["F_bearing"]
    
    fig_forces.add_trace(go.Scatter(
        x=sample_df["t"],
        y=sample_df["F_thrust"],
        mode="lines",
        name="CO₂ Thrust",
        line=dict(color='#FF8C00', width=3),
        fill='tozeroy',
        fillcolor='rgba(255, 140, 0, 0.2)'
    ))
    fig_forces.add_trace(go.Scatter(
        x=sample_df["t"],
        y=sample_df["F_drag"],
        mode="lines",
        name="Drag Force",
        line=dict(color='#E74C3C', width=2),
        fill='tozeroy',
        fillcolor='rgba(231, 76, 60, 0.2)'
    ))
    fig_forces.add_trace(go.Scatter(
        x=sample_df["t"],
        y=sample_df["F_rolling"] + sample_df["F_bearing"],
        mode="lines",
        name="Rolling + Bearing",
        line=dict(color='#888888', width=2),
        fill='tozeroy',
        fillcolor='rgba(136, 136, 136, 0.2)'
    ))
    
    fig_forces.update_layout(
        title="<b>Force Breakdown Over Time</b>",
        xaxis_title="Time (s)",
        yaxis_title="Force (N)",
        height=400,
        template='plotly_white',
        font=dict(size=14, family='Aileron, sans-serif', color='#222222'),
        title_font=dict(family='Museo Moderno, sans-serif', size=18, color='#222222'),
        hovermode='x unified',
        paper_bgcolor='#f1f0e6',
        plot_bgcolor='white',
        legend=dict(x=0.7, y=0.98)
    )
    
    # 3. Speed vs Distance
    fig_speed_dist = go.Figure()
    fig_speed_dist.add_trace(go.Scatter(
        x=df["x"],
        y=df["speed_kmh"],
        mode="lines",
        name="Speed",
        line=dict(color='#222222', width=4)
    ))
    fig_speed_dist.update_layout(
        title="<b>Speed at Each Track Position</b>",
        xaxis_title="Distance (m)",
        yaxis_title="Speed (km/h)",
        height=400,
        template='plotly_white',
        font=dict(size=14, family='Aileron, sans-serif', color='#222222'),
        title_font=dict(family='Museo Moderno, sans-serif', size=18, color='#222222'),
        paper_bgcolor='#f1f0e6',
        plot_bgcolor='white'
    )
    
    # 4. Acceleration Profile
    fig_accel = go.Figure()
    fig_accel.add_trace(go.Scatter(
        x=df["t"],
        y=df["a"],
        mode="lines",
        name="Acceleration",
        line=dict(color='#FF8C00', width=3),
        fill='tozeroy',
        fillcolor='rgba(255, 140, 0, 0.2)'
    ))
    fig_accel.add_hline(y=0, line_dash="dash", line_color="#222222", 
                        annotation_text="Zero = constant speed")
    fig_accel.update_layout(
        title="<b>Acceleration Profile</b>",
        xaxis_title="Time (s)",
        yaxis_title="Acceleration (m/s²)",
        height=400,
        template='plotly_white',
        font=dict(size=14, family='Aileron, sans-serif', color='#222222'),
        title_font=dict(family='Museo Moderno, sans-serif', size=18, color='#222222'),
        paper_bgcolor='#f1f0e6',
        plot_bgcolor='white'
    )
    
    # 5. CO₂ Thrust Profile
    fig_thrust = go.Figure()
    fig_thrust.add_trace(go.Scatter(
        x=df["t"],
        y=df["F_thrust"],
        mode="lines",
        name="CO₂ Thrust",
        line=dict(color='#FF8C00', width=3),
        fill='tozeroy',
        fillcolor='rgba(255, 140, 0, 0.2)'
    ))
    fig_thrust.update_layout(
        title="<b>CO₂ Thrust Over Time</b>",
        xaxis_title="Time (s)",
        yaxis_title="Thrust (N)",
        height=400,
        template='plotly_white',
        font=dict(size=14, family='Aileron, sans-serif', color='#222222'),
        title_font=dict(family='Museo Moderno, sans-serif', size=18, color='#222222'),
        paper_bgcolor='#f1f0e6',
        plot_bgcolor='white'
    )

    return fig_speed, fig_forces, fig_speed_dist, fig_accel, fig_thrust, df

# ---------- Export helpers ----------

def csv_bytes_from_df(df, params_dict, finish_time):
    """
    Create a comprehensive, professional Excel-style CSV with calculated metrics
    """
    buffer = io.StringIO()
    
    # Header Section
    buffer.write("NEEVO - F1 IN SCHOOLS RACE SIMULATION REPORT\n")
    buffer.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    buffer.write("=" * 80 + "\n\n")
    
    # Performance Summary
    buffer.write("PERFORMANCE SUMMARY\n")
    buffer.write("-" * 80 + "\n")
    buffer.write(f"Finish Time,{finish_time:.4f} s\n")
    buffer.write(f"Top Speed,{df['speed_kmh'].max():.2f} km/h\n")
    buffer.write(f"Average Speed,{df['speed_kmh'].mean():.2f} km/h\n")
    buffer.write(f"Peak Acceleration,{df['a'].max():.2f} m/s² ({df['a'].max()/9.81:.2f} G)\n")
    buffer.write(f"Final Speed,{df['speed_kmh'].iloc[-1]:.2f} km/h\n")
    
    # Split Times
    buffer.write("\nSPLIT TIMES\n")
    buffer.write("-" * 80 + "\n")
    buffer.write("Distance,Time,Speed at Split\n")
    for distance in [5, 10, 15, 20]:
        if distance <= df['x'].max():
            split_data = df[df['x'] >= distance].iloc[0]
            buffer.write(f"{distance}m,{split_data['t']:.3f}s,{split_data['speed_kmh']:.1f} km/h\n")
    
    # Acceleration Zones
    buffer.write("\nACCELERATION ANALYSIS\n")
    buffer.write("-" * 80 + "\n")
    buffer.write("Phase,Distance Range,Time Range,Avg Acceleration\n")
    
    # 0-5m Launch
    launch = df[df['x'] <= 5]
    if not launch.empty:
        buffer.write(f"Launch (0-5m),0-5m,0-{launch['t'].iloc[-1]:.3f}s,{launch['a'].mean():.2f} m/s²\n")
    
    # 5-10m Mid-track
    mid = df[(df['x'] > 5) & (df['x'] <= 10)]
    if not mid.empty:
        buffer.write(f"Mid-Track (5-10m),5-10m,{mid['t'].iloc[0]:.3f}-{mid['t'].iloc[-1]:.3f}s,{mid['a'].mean():.2f} m/s²\n")
    
    # 10-15m
    late = df[(df['x'] > 10) & (df['x'] <= 15)]
    if not late.empty:
        buffer.write(f"Late (10-15m),10-15m,{late['t'].iloc[0]:.3f}-{late['t'].iloc[-1]:.3f}s,{late['a'].mean():.2f} m/s²\n")
    
    # 15-20m Final
    final = df[df['x'] > 15]
    if not final.empty:
        buffer.write(f"Final (15-20m),15-20m,{final['t'].iloc[0]:.3f}-{final['t'].iloc[-1]:.3f}s,{final['a'].mean():.2f} m/s²\n")
    
    # Force Analysis
    buffer.write("\nFORCE ANALYSIS\n")
    buffer.write("-" * 80 + "\n")
    buffer.write("Metric,Value\n")
    buffer.write(f"Peak CO₂ Thrust,{df['F_thrust'].max():.3f} N\n")
    buffer.write(f"Average Drag Force,{df['F_drag'].mean():.3f} N\n")
    buffer.write(f"Peak Drag Force,{df['F_drag'].max():.3f} N\n")
    buffer.write(f"Rolling Resistance,{df['F_rolling'].iloc[0]:.3f} N (constant)\n")
    buffer.write(f"Bearing Friction,{df['F_bearing'].iloc[0]:.3f} N (constant)\n")
    buffer.write(f"Average Net Force,{df['F_net'].mean():.3f} N\n")
    
    # Energy Analysis
    buffer.write("\nENERGY ANALYSIS\n")
    buffer.write("-" * 80 + "\n")
    final_v = df['v'].iloc[-1]
    mass_kg = params_dict.get('mass_g', 50) / 1000
    kinetic_energy = 0.5 * mass_kg * (final_v ** 2)
    
    # Calculate work done by each force
    drag_work = (df['F_drag'] * df['v'] * 0.001).sum()  # Approximate work
    rolling_work = (df['F_rolling'] * df['v'] * 0.001).sum()
    thrust_work = (df['F_thrust'] * df['v'] * 0.001).sum()
    
    buffer.write(f"Final Kinetic Energy,{kinetic_energy:.3f} J\n")
    buffer.write(f"Work by CO₂ Thrust,{thrust_work:.3f} J\n")
    buffer.write(f"Work Lost to Drag,{drag_work:.3f} J\n")
    buffer.write(f"Work Lost to Rolling,{rolling_work:.3f} J\n")
    buffer.write(f"Efficiency,{(kinetic_energy/thrust_work*100):.1f}%\n")
    
    # Vehicle Configuration
    buffer.write("\nVEHICLE CONFIGURATION\n")
    buffer.write("-" * 80 + "\n")
    buffer.write(f"Mass,{params_dict.get('mass_g', 0)} g\n")
    buffer.write(f"Drag Coefficient,{params_dict.get('Cd', 0):.3f}\n")
    buffer.write(f"Frontal Area,{params_dict.get('area_cm2', 0):.2f} cm²\n")
    buffer.write(f"Wheel Diameter,{params_dict.get('wheel_diameter_mm', 0)} mm\n")
    buffer.write(f"Bearing Quality,{params_dict.get('bearing_quality', 0)}/5\n")
    buffer.write(f"Wheel Friction,{params_dict.get('wheel_friction', 0):.2f}×\n")
    
    # Propulsion System
    buffer.write("\nPROPULSION SYSTEM\n")
    buffer.write("-" * 80 + "\n")
    buffer.write(f"Peak CO₂ Thrust,{params_dict.get('co2_thrust', 0):.1f} N @ 20°C\n")
    buffer.write(f"Release Duration,{params_dict.get('co2_duration', 0):.2f} s\n")
    buffer.write(f"Launch Technique,{params_dict.get('launch_technique', 'N/A')}\n")
    buffer.write(f"Actual Thrust (temp adjusted),{params_dict.get('actual_thrust', 0):.2f} N\n")
    
    # Track Conditions
    buffer.write("\nTRACK CONDITIONS\n")
    buffer.write("-" * 80 + "\n")
    buffer.write(f"Track Length,{params_dict.get('track_length_m', 0)} m\n")
    buffer.write(f"Surface Type,{params_dict.get('surface', 'N/A')}\n")
    buffer.write(f"Rolling Coefficient,{params_dict.get('Crr', 0):.4f}\n")
    buffer.write(f"Temperature,{params_dict.get('temperature', 0):.1f} °C\n")
    buffer.write(f"Air Pressure,{params_dict.get('pressure', 0):.2f} kPa\n")
    buffer.write(f"Air Density,{params_dict.get('rho', 0):.4f} kg/m³\n")
    
    # Performance Insights
    buffer.write("\nPERFORMANCE INSIGHTS\n")
    buffer.write("-" * 80 + "\n")
    
    # Time to reach 50 km/h
    time_50kmh = df[df['speed_kmh'] >= 50]['t'].iloc[0] if (df['speed_kmh'] >= 50).any() else None
    if time_50kmh:
        buffer.write(f"Time to 50 km/h,{time_50kmh:.3f} s\n")
    
    # Distance when acceleration drops below 5 m/s²
    low_accel = df[df['a'] < 5]
    if not low_accel.empty:
        buffer.write(f"Distance at low acceleration (<5 m/s²),{low_accel['x'].iloc[0]:.2f} m\n")
    
    # Peak power
    peak_power = (df['F_thrust'] * df['v']).max()
    buffer.write(f"Peak Power Output,{peak_power:.2f} W\n")
    
    # Drag dominance point (where drag = 50% of thrust)
    drag_dominant = df[df['F_drag'] >= df['F_thrust'] * 0.5]
    if not drag_dominant.empty:
        buffer.write(f"Drag Dominance Point,{drag_dominant['x'].iloc[0]:.2f} m at {drag_dominant['t'].iloc[0]:.3f} s\n")
    
    buffer.write("\n" + "=" * 80 + "\n")
    buffer.write("END OF REPORT\n")
    
    return buffer.getvalue().encode("utf-8")

# ---------- Streamlit UI ----------

st.set_page_config(layout="wide", page_title="Neevo - STEM Racing Simulator")

st.title("🏎️ Neevo - STEM Racing Simulator")
st.caption("Professional CO₂ dragster physics simulation with advanced controls")

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []

# Initialize session state for persistent values
if 'params' not in st.session_state:
    st.session_state.params = {
        'mass_g': 50,
        'Cd': 0.5,
        'area_cm2': 0.5,
        'wheel_diameter_mm': 25,
        'bearing_quality': 2,
        'wheel_friction': 1.0,
        'co2_thrust': 10.6,
        'co2_duration': 0.5,
        'launch_technique': "Standard",
        'track_length_m': 20,
        'surface': "Regular",
        'temperature': 20.0,
        'pressure': 101.325,
        'time_step': 0.001,
        'enable_drag': True,
        'enable_rolling': True
    }

# Sidebar - two-level menu system with persistent values
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    # Main category selector
    category = st.selectbox(
        "Select Category",
        ["🚗 Vehicle Design", "🛞 Wheels & Bearings", "⚙️ CO₂ Propulsion", "🏁 Track & Environment", "🔬 Advanced Settings"],
        label_visibility="collapsed"
    )
    
    st.divider()
    
    # Show controls based on selected category
    if category == "🚗 Vehicle Design":
        st.markdown("#### Vehicle Design")
        st.session_state.params['mass_g'] = st.slider("Mass (g)", 10, 100, st.session_state.params['mass_g'], 1)
        st.session_state.params['Cd'] = st.slider("Drag Coefficient", 0.2, 0.8, st.session_state.params['Cd'], 0.01,
                                                   help="0.5 is baseline - lower is better")
        st.session_state.params['area_cm2'] = st.slider("Frontal Area (cm²)", 0.1, 5.0, st.session_state.params['area_cm2'], 0.05)
    
    elif category == "🛞 Wheels & Bearings":
        st.markdown("#### Wheels & Bearings")
        st.session_state.params['wheel_diameter_mm'] = st.slider("Wheel Diameter (mm)", 10, 40, st.session_state.params['wheel_diameter_mm'], 1)
        st.session_state.params['bearing_quality'] = st.slider("Bearing Quality (1-5)", 1, 5, st.session_state.params['bearing_quality'], 1)
        st.session_state.params['wheel_friction'] = st.slider("Wheel Friction Multiplier", 0.3, 3.0, st.session_state.params['wheel_friction'], 0.1)
    
    elif category == "⚙️ CO₂ Propulsion":
        st.markdown("#### CO₂ Propulsion System")
        st.session_state.params['co2_thrust'] = st.slider("Peak Thrust @ 20°C (N)", 2.0, 15.0, st.session_state.params['co2_thrust'], 0.1)
        st.session_state.params['co2_duration'] = st.slider("CO₂ Release Duration (s)", 0.1, 0.8, st.session_state.params['co2_duration'], 0.01)
        st.session_state.params['launch_technique'] = st.radio("Launch Technique", 
                                    ["Standard", "Quick Release", "Gradual Release"],
                                    index=["Standard", "Quick Release", "Gradual Release"].index(st.session_state.params['launch_technique']),
                                    horizontal=True)
    
    elif category == "🏁 Track & Environment":
        st.markdown("#### Track & Environment")
        st.session_state.params['track_length_m'] = st.slider("Track Length (m)", 5, 50, st.session_state.params['track_length_m'], 1)
        st.session_state.params['surface'] = st.selectbox("Track Surface", 
                                                          ["Very Smooth", "Smooth", "Regular", "Bumpy"], 
                                                          index=["Very Smooth", "Smooth", "Regular", "Bumpy"].index(st.session_state.params['surface']))
        
        st.markdown("**Environmental Conditions**")
        st.session_state.params['temperature'] = st.slider("Temperature (°C)", -5.0, 45.0, st.session_state.params['temperature'], 1.0)
        st.session_state.params['pressure'] = st.slider("Air Pressure (kPa)", 95.0, 106.0, st.session_state.params['pressure'], 0.5)
    
    elif category == "🔬 Advanced Settings":
        st.markdown("#### Advanced Settings")
        st.session_state.params['time_step'] = st.select_slider("Simulation Precision", 
                                     [0.0001, 0.0005, 0.001, 0.002], 
                                     value=st.session_state.params['time_step'])
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.params['enable_drag'] = st.checkbox("Enable Drag", value=st.session_state.params['enable_drag'])
        with col2:
            st.session_state.params['enable_rolling'] = st.checkbox("Enable Rolling", value=st.session_state.params['enable_rolling'])
    
    # Extract values from session state
    mass_g = st.session_state.params['mass_g']
    Cd = st.session_state.params['Cd']
    area_cm2 = st.session_state.params['area_cm2']
    wheel_diameter_mm = st.session_state.params['wheel_diameter_mm']
    bearing_quality = st.session_state.params['bearing_quality']
    wheel_friction = st.session_state.params['wheel_friction']
    co2_thrust = st.session_state.params['co2_thrust']
    co2_duration = st.session_state.params['co2_duration']
    launch_technique = st.session_state.params['launch_technique']
    track_length_m = st.session_state.params['track_length_m']
    surface = st.session_state.params['surface']
    temperature = st.session_state.params['temperature']
    pressure = st.session_state.params['pressure']
    time_step = st.session_state.params['time_step']
    enable_drag = st.session_state.params['enable_drag']
    enable_rolling = st.session_state.params['enable_rolling']

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🚀 SIMULATE", use_container_width=True, type="primary"):
            st.session_state.simulate_now = True
    with col2:
        if st.button("💾 SAVE RUN", use_container_width=True):
            st.session_state.save_run = True

# Map surface to Crr
surface_coefficients = {
    "Very Smooth": 0.006,
    "Smooth": 0.010,
    "Regular": 0.018,
    "Bumpy": 0.035
}
Crr = surface_coefficients[surface]

# Calculate air density
rho = air_density(temperature, pressure)

# Convert to SI units
mass_kg = mass_g / 1000.0
area_m2 = area_cm2 / 10000.0

# Calculate temperature-adjusted thrust
actual_thrust = co2_thrust_adjusted(co2_thrust, temperature)
thrust_change_pct = ((actual_thrust - co2_thrust) / co2_thrust * 100) if co2_thrust > 0 else 0

# Run simulation
do_sim = st.session_state.get("simulate_now", True)
if "simulate_now" in st.session_state:
    st.session_state.simulate_now = False

rows, finish_time = simulate_run(
    mass_kg=mass_kg,
    Cd=Cd,
    area_m2=area_m2,
    Crr=Crr,
    wheel_friction=wheel_friction,
    track_length_m=track_length_m,
    time_step=time_step,
    enable_drag=enable_drag,
    enable_rolling=enable_rolling,
    rho=rho,
    co2_thrust=co2_thrust,
    co2_duration=co2_duration,
    temperature_c=temperature,
    bearing_quality=bearing_quality,
    wheel_diameter_mm=wheel_diameter_mm,
    launch_technique=launch_technique
)

fig_speed, fig_forces, fig_speed_dist, fig_accel, fig_thrust, df = build_plotly_time_series(rows)

# Display key metrics
st.markdown("---")
st.markdown("## 📊 Race Performance")

metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)

with metric_col1:
    st.metric("⏱️ FINISH TIME", f"{finish_time:.3f} s" if finish_time else "—")

with metric_col2:
    top_speed = df["speed_kmh"].max() if df is not None and not df.empty else None
    st.metric("🚀 TOP SPEED", f"{top_speed:.1f} km/h" if top_speed else "—")

with metric_col3:
    avg_speed = df["speed_kmh"].mean() if df is not None and not df.empty else None
    st.metric("📊 AVG SPEED", f"{avg_speed:.1f} km/h" if avg_speed else "—")

with metric_col4:
    max_accel = df["a"].max() if df is not None and not df.empty else None
    st.metric("⚡ LAUNCH G's", f"{max_accel/9.81:.1f} G" if max_accel else "—")

with metric_col5:
    st.metric("🌡️ CO₂ THRUST", f"{actual_thrust:.1f} N", 
              delta=f"{thrust_change_pct:+.1f}%" if abs(thrust_change_pct) > 0.1 else "nominal")

# Split times
if df is not None and not df.empty:
    st.markdown("### ⏲️ Split Times")
    split_cols = st.columns(6)
    
    splits = [5, 10, 15, 20]
    for idx, distance in enumerate(splits[:4]):
        if distance <= track_length_m:
            split_time = df[df["x"] >= distance]["t"].iloc[0] if (df["x"] >= distance).any() else None
            with split_cols[idx]:
                st.metric(f"📍 {distance}m", f"{split_time:.2f}s" if split_time else "—")
    
    # Speed at 10m mark
    with split_cols[4]:
        speed_at_10m = df[df["x"] >= 10.0]["speed_kmh"].iloc[0] if (df["x"] >= 10.0).any() else None
        st.metric("🏁 Speed @ 10m", f"{speed_at_10m:.1f} km/h" if speed_at_10m else "—")
    
    # Final speed
    with split_cols[5]:
        final_speed = df["speed_kmh"].iloc[-1]
        st.metric("🎯 Final Speed", f"{final_speed:.1f} km/h")

# Temperature effect notice
if abs(thrust_change_pct) > 1.0:
    if temperature > 20.0:
        st.success(f"🔥 **HOT TRACK ADVANTAGE:** Temperature boost gives +{thrust_change_pct:.1f}% extra thrust!")
    else:
        st.warning(f"❄️ **COLD TRACK PENALTY:** Low temperature reduces thrust by {abs(thrust_change_pct):.1f}%")

# Force breakdown info box
if df is not None and not df.empty:
    st.markdown("### 🔬 Force Analysis at Finish Line")
    force_col1, force_col2, force_col3, force_col4, force_col5 = st.columns(5)
    
    final = df.iloc[-1]
    
    with force_col1:
        st.metric("⚪ CO₂ Thrust", f"{final['F_thrust']:.2f} N")
    with force_col2:
        st.metric("🔴 Drag Force", f"{final['F_drag']:.2f} N")
    with force_col3:
        st.metric("🟠 Rolling Resistance", f"{final['F_rolling']:.2f} N")
    with force_col4:
        total_resistance = final['F_drag'] + final['F_rolling'] + final['F_bearing']
        st.metric("⚫ Total Resistance", f"{total_resistance:.2f} N")
    with force_col5:
        efficiency = (final['F_thrust'] - total_resistance) / final['F_thrust'] * 100 if final['F_thrust'] > 0 else 0
        st.metric("📈 Efficiency", f"{efficiency:.0f}%")

# Display graphs
st.markdown("---")
st.markdown("## 📈 Detailed Analysis")

graph_row1_col1, graph_row1_col2 = st.columns(2)

with graph_row1_col1:
    if fig_speed:
        st.plotly_chart(fig_speed, use_container_width=True)

with graph_row1_col2:
    if fig_speed_dist:
        st.plotly_chart(fig_speed_dist, use_container_width=True)

graph_row2_col1, graph_row2_col2 = st.columns(2)

with graph_row2_col1:
    if fig_forces:
        st.plotly_chart(fig_forces, use_container_width=True)

with graph_row2_col2:
    if fig_accel:
        st.plotly_chart(fig_accel, use_container_width=True)

# CO₂ thrust profile
if fig_thrust:
    st.plotly_chart(fig_thrust, use_container_width=True)

# Physics parameters display
with st.expander("🔍 View Detailed Physics Parameters"):
    phys_col1, phys_col2, phys_col3 = st.columns(3)
    
    with phys_col1:
        st.write("**Aerodynamics:**")
        st.write(f"• Air density: {rho:.4f} kg/m³")
        st.write(f"• Drag coefficient: {Cd:.3f}")
        st.write(f"• Frontal area: {area_cm2:.2f} cm²")
        st.write(f"• Enhanced drag factor: 15×")
    
    with phys_col2:
        st.write("**Rolling & Friction:**")
        st.write(f"• Surface Crr: {Crr:.4f}")
        st.write(f"• Wheel friction: {wheel_friction:.2f}×")
        st.write(f"• Wheel diameter: {wheel_diameter_mm}mm")
        st.write(f"• Bearing quality: {bearing_quality}/5")
        actual_rolling = Crr * mass_kg * G * wheel_friction
        st.write(f"• Total rolling force: {actual_rolling:.4f} N")
    
    with phys_col3:
        st.write("**Propulsion:**")
        st.write(f"• Base thrust: {co2_thrust:.2f} N")
        st.write(f"• CO₂ duration: {co2_duration:.2f}s")
        st.write(f"• Launch technique: {launch_technique}")
        st.write(f"• Temperature: {temperature:.1f}°C")
        st.write(f"• Temp factor: {(temperature+273.15)/293.15:.4f}")
        st.write(f"• Peak thrust: {actual_thrust:.3f} N")

# Comparison tool
with st.expander("📊 Compare Two Configurations"):
    st.info("Save multiple runs and compare them here!")
    if len(st.session_state.history) >= 2:
        comp_col1, comp_col2 = st.columns(2)
        with comp_col1:
            run1 = st.selectbox("Select Run 1", [f"{r['label']}" for r in st.session_state.history], key="comp1")
        with comp_col2:
            run2 = st.selectbox("Select Run 2", [f"{r['label']}" for r in st.session_state.history], key="comp2")
        
        if st.button("Compare Runs"):
            r1_data = next((r for r in st.session_state.history if r['label'] == run1), None)
            r2_data = next((r for r in st.session_state.history if r['label'] == run2), None)
            
            if r1_data and r2_data:
                comp_result_col1, comp_result_col2, comp_result_col3 = st.columns(3)
                with comp_result_col1:
                    time_diff = r2_data['results']['finish_time'] - r1_data['results']['finish_time']
                    st.metric(f"Time Difference", f"{abs(time_diff):.3f}s", 
                             delta=f"{time_diff:.3f}s" if time_diff != 0 else "Same")
                with comp_result_col2:
                    speed_diff = r2_data['results']['top_speed'] - r1_data['results']['top_speed']
                    st.metric(f"Speed Difference", f"{abs(speed_diff):.1f} km/h", 
                             delta=f"{speed_diff:.1f} km/h" if speed_diff != 0 else "Same")
                with comp_result_col3:
                    st.write(f"**Run 1:** {r1_data['results']['finish_time']:.3f}s")
                    st.write(f"**Run 2:** {r2_data['results']['finish_time']:.3f}s")

# Export
st.markdown("---")
st.markdown("### 💾 Export Data")

if df is not None and not df.empty:
    # Prepare params dict for export
    export_params = {
        'mass_g': mass_g,
        'Cd': Cd,
        'area_cm2': area_cm2,
        'wheel_diameter_mm': wheel_diameter_mm,
        'bearing_quality': bearing_quality,
        'wheel_friction': wheel_friction,
        'co2_thrust': co2_thrust,
        'co2_duration': co2_duration,
        'launch_technique': launch_technique,
        'track_length_m': track_length_m,
        'surface': surface,
        'Crr': Crr,
        'temperature': temperature,
        'pressure': pressure,
        'rho': rho,
        'actual_thrust': actual_thrust
    }
    
    csv_bytes = csv_bytes_from_df(df, export_params, finish_time)
    st.download_button(
        "📥 Download Comprehensive Race Report (CSV)", 
        data=csv_bytes, 
        file_name=f"neevo_race_report_{int(datetime.now().timestamp())}.csv", 
        mime="text/csv"
    )
    st.caption("Report includes: Performance metrics, split times, force analysis, energy calculations, and configuration details")

# Save to history
if st.session_state.get("save_run", False):
    st.session_state.save_run = False
    run_snapshot = {
        "id": int(datetime.now().timestamp() * 1000),
        "label": f"Run {len(st.session_state.history)+1}",
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "params": {
            "mass_g": mass_g,
            "Cd": Cd,
            "area_cm2": area_cm2,
            "wheel_friction": wheel_friction,
            "wheel_diameter_mm": wheel_diameter_mm,
            "bearing_quality": bearing_quality,
            "temperature": temperature,
            "co2_thrust": co2_thrust,
            "co2_duration": co2_duration,
            "launch_technique": launch_technique,
            "actual_thrust": actual_thrust,
            "surface": surface,
            "Crr": Crr
        },
        "results": {
            "finish_time": finish_time,
            "top_speed": float(top_speed) if top_speed else None,
            "avg_speed": float(avg_speed) if avg_speed else None
        }
    }
    st.session_state.history.insert(0, run_snapshot)
    st.success("✅ Run saved to history!")

# History display
if st.session_state.history:
    st.markdown("---")
    st.markdown("### 📜 Run History")
    
    for run in st.session_state.history[:5]:
        with st.expander(f"**{run['label']}** — {run['date']} — ⏱️ {run['results']['finish_time']:.3f}s — 🚀 {run['results']['top_speed']:.1f} km/h"):
            h_col1, h_col2, h_col3 = st.columns(3)
            
            with h_col1:
                st.write("**Vehicle:**")
                st.write(f"Mass: {run['params']['mass_g']}g")
                st.write(f"Cd: {run['params']['Cd']:.3f}")
                st.write(f"Area: {run['params']['area_cm2']}cm²")
                st.write(f"Wheel Ø: {run['params']['wheel_diameter_mm']}mm")
            
            with h_col2:
                st.write("**Propulsion:**")
                st.write(f"CO₂ Thrust: {run['params']['co2_thrust']:.1f}N")
                st.write(f"Duration: {run['params']['co2_duration']:.2f}s")
                st.write(f"Launch: {run['params']['launch_technique']}")
                st.write(f"Temp: {run['params']['temperature']}°C")
            
            with h_col3:
                st.write("**Results:**")
                st.write(f"Finish: {run['results']['finish_time']:.3f}s")
                st.write(f"Top: {run['results']['top_speed']:.1f} km/h")
                st.write(f"Avg: {run['results']['avg_speed']:.1f} km/h")
            
            if st.button(f"🗑️ Delete Run", key=f"del_{run['id']}"):
                st.session_state.history.remove(run)
                st.rerun()

st.markdown("---")
st.caption("© Neevo STEM Racing Simulator | Advanced F1 in Schools Physics Simulation")
