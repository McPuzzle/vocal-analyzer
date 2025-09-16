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

# Session state defaults
if 'chain_ready' not in st.session_state: st.session_state.chain_ready=False
if 'chain_df' not in st.session_state: st.session_state.chain_df=pd.DataFrame()

# (Key/BPM auto+manual override code goes here…)

# Analyze block
if dry_file and ref_file and st.button("Analyze Vocals"):
    # Save temp files
    t1=tempfile.NamedTemporaryFile(suffix='.wav',delete=False)
    t1.write(dry_file.read()); t1.close()
    t2=tempfile.NamedTemporaryFile(suffix='.wav',delete=False)
    t2.write(ref_file.read()); t2.close()
    dry_path,ref_path=t1.name,t2.name

    # Spectrum
    seg_start,seg_end=0.33,0.66
    def spectrum(path):
        sr,data=wavfile.read(path)
        if data.ndim>1: data=data.mean(1)
        seg=data[int(len(data)*seg_start):int(len(data)*seg_end)]
        w=seg*windows.hann(len(seg)); fft=np.fft.rfft(w); freqs=np.fft.rfftfreq(len(seg),1/sr)
        return freqs,np.abs(fft)
    freqs,mag_dry=spectrum(dry_path); _,mag_ref=spectrum(ref_path)

    # Plot
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=freqs,y=mag_dry,name="Dry"))
    fig.add_trace(go.Scatter(x=freqs,y=mag_ref,name="Ref"))
    fig.update_layout(title="Spectrum",xaxis_type='log',yaxis_title='Mag')
    st.plotly_chart(fig,use_container_width=True)

    # Find peaks
    bands={"Low":(20,600),"Mid":(600,1200),"High":(1200,12000)}
    peaks=[]
    for band,(fmin,fmax) in bands.items():
        idx=(freqs>=fmin)&(freqs<=fmax)
        bm,bf=mag_dry[idx],freqs[idx]
        p,props=find_peaks(bm,height=bm.max()*0.05)
        if not p.size: continue
        top=p[np.argsort(props['peak_heights'])[-3:]]
        for i in top:
            cf=bf[i]; half=bm[i]/np.sqrt(2)
            l,r=i,i
            while l>0 and bm[l]>half: l-=1
            while r<len(bm)-1 and bm[r]>half: r+=1
            bw=bf[r]-bf[l] if r<len(bf) and l>=0 else cf/10
            Q=cf/bw if bw>0 else 10
            cut=np.clip(10*np.log10(bm[i]/np.median(bm)),1,8)
            peaks.append({"Band":band,"Center":cf,"Q":Q,"Cut":cut})
    df_ed=pd.DataFrame(peaks)

    # Slider edits
    st.subheader("🔧 Surgical Notches")
    edits=[]
    for i,row in df_ed.iterrows():
        st.markdown(f"**{row['Band']} @ {row['Center']:.1f} Hz**")
        c=st.slider("Cut dB",1.0,8.0,value=float(row['Cut']),step=0.1,key=f"c{i}")
        q=st.slider("Q",1.0,10.0,value=float(row['Q']),step=0.1,key=f"q{i}")
        edits.append({"Band":row['Band'],"Center":row['Center'],"Q":q,"Cut":c})
    df_ed=pd.DataFrame(edits)

    # **Store** df_ed for chain builder
    st.session_state.chain_df=df_ed.copy()
    st.session_state.chain_ready=True
    st.success("Analysis complete. Ready to generate mix chain.")

    # Cleanup
    os.unlink(dry_path); os.unlink(ref_path)

# Chain builder
if st.session_state.chain_ready:
    st.subheader("🔧 Iterative Mix Chain Builder")
    if st.button("Generate Engineer-Ready Chain"):
        s=st.session_state.chain_df.copy()
        # compute P1–P6
        for idx,pct in enumerate([0.10,0.20,0.10,0.30,0.20,0.10],start=1):
            s[f"P{idx}"]=(s["Cut"]*pct).round(2)

        # Prepare raw steps
        raw={f"P{idx}":[] for idx in range(1,7)}
        for _,r in s.iterrows():
            raw["P1"].append(f"{r['Band']} @ {r['Center']:.1f} Hz — Q1.5, Cut {r['P1']} dB")
            raw["P2"].append(f"{r['Band']} @ {r['Center']:.1f} Hz — Q{3 if r['Band']=='Low' else 4}, Cut {r['P2']} dB")
            raw["P3"].append(f"Mid-Shelf @ 1.2kHz — Cut {s['P3'].mean():.2f} dB\nHigh Shelf @ 12kHz — Cut {s['P3'].mean():.2f} dB")
            raw["P4"].append(f"{r['Band']} ~{r['Center']:.1f} Hz — Max {r['P4']} dB, Ratio 3:1")
            raw["P5"].append(f"{r['Band']} @ {r['Center']:.1f} Hz — Q0.7, Cut {r['P5']} dB")
        # P6 handled separately

        templates={
            "P1":("Pass A: Broad Subtractive Cuts (10%)","FabFilter Pro-Q3 — HPF@100Hz; Q=1.5"),
            "P2":("Pass B: Narrow Notches (20%)","Waves Q10 / Pro-Q3 — Q=3–4"),
            "P3":("Pass C: Shelf EQ (10%)","FabFilter Pro-Q3 Dynamic — Shelf"),
            "P4":("Pass D: Dynamic Notches (30%)","FabFilter Pro-Q3 Dynamic / Waves F6"),
            "P5":("Pass E: Parallel Notches & Air (20%)","Maag EQ4 Air + Saturation Bus"),
            "P6":("Pass F: Compression & De-essing (10%)","1176 → SSL Bus → Pro-DS")
        }

        for key in ["P1","P2","P3","P4","P5"]:
            header,plugin=templates[key]
            st.markdown(f"### {header}\n*Plugin:* {plugin}")
            for item in raw[key]:
                st.write(f"• {item}")

        # Pass F details
        st.markdown(f"### {templates['P6'][0]}\n*Plugins:* {templates['P6'][1]}")
        atk_sub,atk_ms=ms_to_subdivision(15,ref_bpm)
        rel_sub,rel_ms=ms_to_subdivision(250,ref_bpm)
        st.write(f"• Comp A 4:1 — Attack~{atk_sub} ({atk_ms} ms), Release~{rel_ms} ms")
        st.write(f"• Comp B 2.5:1 — Attack~{atk_sub} ({atk_ms} ms), Release~{rel_ms} ms")
        th_min,th_max=guidelines['deesser']['default']['threshold_db']
        r_min,r_max=guidelines['deesser']['default']['ratio']
        st.write(f"• De-Esser 5–8 kHz — Threshold {th_min} to {th_max} dB, Ratio {r_min}:{r_max}")
        st.write(f"• A/B vs ref (Key: {ref_key}, BPM: {ref_bpm})")
