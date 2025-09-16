    # ─── Key & BPM Detection for Both Vocals ───
    st.subheader("🔑 Key & 🎵 Tempo Detection (Dry & Reference)")

    # Optional manual overrides
    col1, col2 = st.columns(2)
    with col1:
        manual_dry_key = st.text_input("Dry Vocal Key (e.g. A minor)", "")
        manual_dry_bpm = st.number_input("Dry Vocal BPM", min_value=30, max_value=300, value=0)
    with col2:
        manual_ref_key = st.text_input("Reference Key (e.g. C major)", "")
        manual_ref_bpm = st.number_input("Reference BPM", min_value=30, max_value=300, value=0)

    def detect_key_bpm(path, label):
        try:
            y, sr = librosa.load(path, sr=None, mono=True, duration=30)
            # Key
            chroma = librosa.feature.chroma_cens(y=y, sr=sr)
            key_idx = chroma.mean(axis=1).argmax()
            keys = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
            key = keys[key_idx]
            # BPM
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            return key, int(tempo)
        except:
            return "Unknown", 0

    # Dry vocal detection if no manual
    if manual_dry_key and manual_dry_bpm:
        dry_key, dry_bpm = manual_dry_key, int(manual_dry_bpm)
    else:
        dry_key, dry_bpm = detect_key_bpm(dry_path, "Dry")

    # Reference detection if no manual
    if manual_ref_key and manual_ref_bpm:
        ref_key, ref_bpm = manual_ref_key, int(manual_ref_bpm)
    else:
        ref_key, ref_bpm = detect_key_bpm(ref_path, "Ref")

    st.write(f"• Dry Vocal → Key: **{dry_key}**, Tempo: **{dry_bpm} BPM**")
    st.write(f"• Reference → Key: **{ref_key}**, Tempo: **{ref_bpm} BPM**")
    # ─────────────────────────────────────────────
