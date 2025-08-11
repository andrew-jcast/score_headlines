# streamlit_app.py
import time
import re
import unicodedata
import hashlib
import requests
import pandas as pd
import streamlit as st
import plotly.express as px
from collections import OrderedDict

st.set_page_config(page_title="News Headline - Sentiment Scoring", layout="centered")
st.title("News Headline - Sentiment Scoring")

# ---------- session state ----------
if "pred_cache" not in st.session_state:
    st.session_state.pred_cache = (
        OrderedDict()
    )  # key: client_side_id -> (label, timestamp)
if "headlines_list" not in st.session_state:
    st.session_state.headlines_list = []  # stores current editable headlines

# ---------- config ----------
CACHE_MAX = 50  # max headlines in cache per session


# ---------- headline parsing ----------
def clean_headline(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)  # normalize unicode
    text = text.strip()
    text = re.sub(r"\s+", " ", text)  # collapse multiple spaces/tabs
    text = text.strip(" .,!?:;\"'[]{}()")  # strip leading/trailing punctuation
    return text


def parse_headlines(raw_lines: list[str]) -> list[str]:
    seen = set()
    cleaned_lines = []
    for ln in raw_lines:
        cleaned = clean_headline(ln)
        if len(cleaned) < 3:  # skip too short
            continue
        key = cleaned.lower()
        if key not in seen:
            seen.add(key)
            cleaned_lines.append(cleaned)
    return cleaned_lines


# ---------- ID generator ----------
def client_side_id(text: str) -> str:
    normalized = text.lower().strip()
    return hashlib.blake2b(normalized.encode("utf-8"), digest_size=10).hexdigest()


# ---------- sidebar ----------
api_url = st.sidebar.text_input("API base URL", value="http://localhost:8002")
timeout_s = st.sidebar.number_input(
    "Request timeout (s)", min_value=1, max_value=60, value=10
)
show_ids = st.sidebar.checkbox("Show IDs in table", value=True)


@st.cache_data(ttl=5)
def check_status(base_url: str, timeout: int):
    try:
        r = requests.get(f"{base_url.rstrip('/')}/status", timeout=timeout)
        return (
            (r.status_code == 200 and r.json().get("status") == "OK"),
            r.status_code,
            r.text,
        )
    except Exception as e:
        return False, None, str(e)


ok, sc, txt = check_status(api_url, timeout_s)
st.sidebar.markdown(f"**Status:** {'Online' if ok else 'Offline'}")
if not ok:
    st.sidebar.caption(f"{'' if sc is None else f'{sc}: '}{txt}")

if st.sidebar.button(
    "Clear cache - previous results will be cleared from this session."
):
    st.session_state.pred_cache.clear()
    st.sidebar.success("Cache cleared.")

# ---------- input section ----------
st.subheader("Enter headlines")

# Bulk paste input
bulk_input = st.text_area(
    "Paste headlines (one per line)",
    value="",
    height=150,
    placeholder="Stocks rally as inflation cools\nOil prices drop amid global slowdown",
)

# Quick add single headline
quick_add = st.text_input("Quick add a single headline")

# File upload (.txt)
file_in = st.file_uploader("Upload a .txt file (one headline per line)", type=["txt"])
if file_in is not None:
    new_lines = parse_headlines(
        file_in.read().decode("utf-8", errors="ignore").splitlines()
    )
    st.session_state.headlines_list.extend(new_lines)

col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    if st.button("Add from text area"):
        new_lines = parse_headlines(bulk_input.splitlines())
        st.session_state.headlines_list.extend(new_lines)

with col2:
    if st.button("Add single headline"):
        if quick_add.strip():
            new_lines = parse_headlines([quick_add])
            st.session_state.headlines_list.extend(new_lines)

with col3:
    if st.button("Clear all headlines"):
        st.session_state.headlines_list = []

# Editable preview table
if st.session_state.headlines_list:
    st.markdown("### Review & Edit Headlines")
    edited_df = st.data_editor(
        pd.DataFrame({"headline": st.session_state.headlines_list}),
        num_rows="dynamic",
        use_container_width=True,
    )
    st.session_state.headlines_list = (
        edited_df["headline"].dropna().astype(str).tolist()
    )
else:
    st.info("No headlines added yet.")


# ---------- scoring with client-side cache ----------
def score_with_client_cache(headlines: list[str]):
    ids_out = [None] * len(headlines)
    labels_out = [None] * len(headlines)
    hits = 0
    misses = 0
    hit_headlines = []

    to_score = []
    to_score_idx = []

    for idx, hl in enumerate(headlines):
        cid = client_side_id(hl)
        ids_out[idx] = cid

        if cid in st.session_state.pred_cache:
            labels_out[idx] = st.session_state.pred_cache[cid][0]
            hits += 1
            hit_headlines.append(hl)
        else:
            to_score.append(hl)
            to_score_idx.append(idx)
            misses += 1

    if to_score:
        payload = {"headlines": to_score, "return_ids": False}
        resp = requests.post(
            f"{api_url.rstrip('/')}/score_headlines", json=payload, timeout=timeout_s
        )
        resp.raise_for_status()
        preds = resp.json().get("labels", [])

        for idx, label in zip(to_score_idx, preds):
            labels_out[idx] = label
            cid = ids_out[idx]
            if len(st.session_state.pred_cache) >= CACHE_MAX:
                st.session_state.pred_cache.popitem(last=False)  # FIFO eviction
            st.session_state.pred_cache[cid] = (label, time.time())

    if misses == 0:
        source_info = f"Pulled all {hits} results from cache."
    else:
        source_info = f"API call for {misses} headlines (Cache hits: {hits})."
        if hit_headlines:
            source_info += "\n\nCache hits:\n- " + "\n- ".join(hit_headlines)

    return ids_out, labels_out, hits, misses, source_info


# ---------- action ----------
if st.button("Score"):
    if not st.session_state.headlines_list:
        st.warning("Please add at least one headline.")
    else:
        try:
            ids, labels, hits, misses, source_info = score_with_client_cache(
                st.session_state.headlines_list
            )
            df = pd.DataFrame(
                {"headline": st.session_state.headlines_list, "label": labels}
            )
            if show_ids:
                df.insert(1, "id", ids)

            # Status message
            st.success(source_info)

            # Data table
            st.dataframe(df, use_container_width=True)

            # Sentiment distribution chart
            sentiment_counts = df["label"].value_counts().reset_index()
            sentiment_labels_colors = {
                "Optimistic": "green",
                "Neutral": "grey",
                "Pessimistic": "red",
            }

            sentiment_counts.columns = ["label", "count"]
            fig = px.bar(
                sentiment_counts,
                x="label",
                y="count",
                title="Sentiment Distribution",
                color="label",  # Use the 'label' column to determine colors
                text="count",
                color_discrete_map=sentiment_labels_colors,  # Map specific colors to labels
                color_discrete_sequence=px.colors.qualitative.Set2,  # Optional: fallback color sequence
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(
                yaxis_title="Number of Headlines",
                xaxis_title="Sentiment",
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

            # CSV download
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download CSV",
                data=csv,
                file_name="headline_scores.csv",
                mime="text/csv",
            )

        except requests.exceptions.RequestException as e:
            st.error(f"Request failed: {e}")
        except Exception as e:
            st.error(f"Failed: {e}")
