import streamlit as st
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import windows, find_peaks
import plotly.graph_objects as go
import tempfile, os
import json
import librosa

# Load engineering guidelines (fallback defaults)
@st.cache_data
def load_guidelines():
    default = {
        "dynamic_eq": {"fabfilter_f6": {"attack_ms":[1,10],"release_ms":[50,250]}},
        "compressor": {"neve_33609": {"attack_ms":[10,30],"release_ms":[100,300]}},
        "deesser": {"default": {"threshold_db":[-20,-10],"ratio":[2,6]}}
    }
    try:
        with open('engineer_guidelines.json') as f:
            data = json.load(f)
            for k,v in default.items():
                data.setdefault(k, v)
            return data
    except FileNotFoundError:
        return default

guidelines = load_guidelines()

def ms_to_subdivision(ms,bpm):
    beat_ms=60000/bpm
    subs={'1/4':beat_ms,'1/8':beat_ms/2,'1/16':beat_ms/4,'1/32':beat_ms/8,'1/64':beat_ms/16}
    name,val=min(subs.items(), key=lambda x:abs(x[1]-ms))
    return name,int(val)

st.title("🎤 Advanced Vocal EQ Analyzer")
st.write("Automated engineer-ready mix chain formatting")

# Upload
dry_file=st.file_uploader("Dry Vocal (WAV)",type='wav')
ref_file=st.file_uploader("Reference Track (WAV)",type='wav')

# Session state
for var in ('chain_df','chain_ready'): 
    if var not in st.session_state: st.session_state[var]=None if var=='chain_df' else False

# Key/BPM detection & manual override (omitted for brevity)...

# Analyze
if dry_file and ref_file and st.button("Analyze Vocals"):
    # Temporary files
    t1=tempfile.NamedTemporaryFile(suffix='.wav',delete=False); t1.write(dry_file.read()); t1.close()
    t2=tempfile.NamedTemporaryFile(suffix='.wav',delete=False); t2.write(ref_file.read()); t2.close()
    dry_path,ref_path=t1.name,t2.name

    # Spectrum and peak detection (omitted)

    # Build df_ed of surgical notches (omitted)

    st.session_state.chain_df=df_ed.copy()
    st.session_state.chain_ready=True

    os.unlink(dry_path); os.unlink(ref_path)
    st.success("Analysis complete. Ready to generate mix chain.")

# Chain builder with templates
if st.session_state.chain_ready:
    s=st.session_state.chain_df.copy()
    # compute splits
    for i,p in enumerate([0.10,0.20,0.10,0.30,0.20,0.10],start=1):
        s[f"P{i}"]=(s["Cut"]*p).round(2)

    # raw steps per pass
    raw={f"P{i}":[] for i in range(1,7)}
    for _,r in s.iterrows():
        raw["P1"].append(f"{r['Band']} @ {r['Center']:.1f} Hz — Q1.5, Cut {r['P1']} dB")
        raw["P2"].append(f"{r['Band']} @ {r['Center']:.1f} Hz — Q{3 if r['Band']=='Low' else 4}, Cut {r['P2']} dB")
        raw["P3"].append(f"Mid-Shelf @ 1200 Hz — Cut {s['P3'].mean():.2f} dB\nHigh-Pass Shelf @ 12000 Hz — Cut {s['P3'].mean():.2f} dB")
        raw["P4"].append(f"{r['Band']} ~{r['Center']:.1f} Hz — Max {r['P4']} dB, Ratio 3:1")
        raw["P5"].append(f"{r['Band']} @ {r['Center']:.1f} Hz — Q0.7, Cut {r['P5']} dB")
        raw["P6"].append("")  # limiter and comp handled separately

    templates={
        "P1":("Pass A: Broad Subtractive Cuts (10%)","FabFilter Pro-Q3 — HPF @100Hz; Broad Q=1.5"),
        "P2":("Pass B: Narrow Notches (20%)","Waves Q10 / Pro-Q3 — Q=3–4"),
        "P3":("Pass C: Dynamic Shelves (10%)","FabFilter Pro-Q3 Dynamic — Shelf EQ"),
        "P4":("Pass D: Dynamic Notches (30%)","FabFilter Pro-Q3 Dynamic or Waves F6 — Ratio 3:1"),
        "P5":("Pass E: Parallel Notches & Air (20%)","Maag EQ4 Air + Saturation Bus"),
        "P6":("Pass F: Compression & Limiting (10%)","UAD 1176 → SSL G-Bus; Limiter")
    }

    if st.button("Generate Engineer-Ready Chain"):
        for key in ["P1","P2","P3","P4","P5"]:
            header,plugin=templates[key]
            st.markdown(f"### {header}\n*Plugin:* {plugin}")
            for item in raw[key]:
                st.write(f"• {item}")

        # Pass F details
        st.markdown(f"### {templates['P6'][0]}\n*Plugins:* {templates['P6'][1]}")
        atk_sub,atk_ms=ms_to_subdivision(3,ref_bpm)
        comp_atk,comp_ms=ms_to_subdivision(15,ref_bpm)
        comp_rel,rel_ms=ms_to_subdivision(250,ref_bpm)
        st.write(f"• Limiter: Fast Attack, Release 50ms, GR ~2dB")
        st.write(f"• Comp A (4:1): Attack ~{comp_atk} ({comp_ms}ms), Release ~{rel_ms}ms")
        st.write(f"• Comp B (2.5:1): Attack ~{comp_atk} ({comp_ms}ms), Release ~{rel_ms}ms")

        # De-esser
        th_min,th_max=guidelines['deesser']['default']['threshold_db']
        r_min,r_max=guidelines['deesser']['default']['ratio']
        st.markdown("### Final De-essing & A/B")
        st.write(f"• De-Esser 5–8 kHz — Threshold {th_min} to {th_max} dB, Ratio {r_min}:{r_max}")
        st.write(f"• A/B vs reference (Key: {ref_key}, BPM: {ref_bpm})")
