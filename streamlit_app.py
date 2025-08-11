# streamlit_app.py
import time
import hashlib
import requests
import pandas as pd
import streamlit as st
from collections import OrderedDict

st.set_page_config(page_title="News Headline - Sentiment Scoring", layout="centered")
st.title("News Headline - Sentiment Scoring")

# ---------- session state ----------
if "pred_cache" not in st.session_state:
    # key: client_side_id -> (label, timestamp)
    st.session_state.pred_cache = OrderedDict()

# ---------- config ----------
CACHE_MAX = 50  # max headlines in cache per session

# Local ID generator, not from API to avoid unnecessary calls
def client_side_id(text: str) -> str:
    normalized = text.lower().strip()
    return hashlib.blake2b(normalized.encode("utf-8"), digest_size=10).hexdigest()

# ---------- sidebar ----------
api_url = st.sidebar.text_input("API base URL", value="http://localhost:8002")
timeout_s = st.sidebar.number_input("Request timeout (s)", min_value=1, max_value=60, value=10)
show_ids = st.sidebar.checkbox("Show IDs in table", value=True)

@st.cache_data(ttl=5)
def check_status(base_url: str, timeout: int):
    try:
        r = requests.get(f"{base_url.rstrip('/')}/status", timeout=timeout)
        return (r.status_code == 200 and r.json().get("status") == "OK"), r.status_code, r.text
    except Exception as e:
        return False, None, str(e)

ok, sc, txt = check_status(api_url, timeout_s)
st.sidebar.markdown(f"**Status:** {'Online' if ok else 'Offline'}")
if not ok:
    st.sidebar.caption(f"{'' if sc is None else f'{sc}: '}{txt}")

if st.sidebar.button("Clear cache - previous results will be cleared from this session."):
    st.session_state.pred_cache.clear()
    st.sidebar.success("Cleared.")

# ---------- input ----------
st.subheader("Enter headlines (please seperate each additional headline with a newline).")
placeholder = "Stocks rally as inflation cools\nOil prices drop amid global slowdown"
text_in = st.text_area("Headlines", value="", height=180, placeholder=placeholder)
file_in = st.file_uploader("Upload a .txt file to bulk load headlines (one headline per line).", type=["txt"])

def parse_lines(s: str):
    return [ln.strip() for ln in s.splitlines() if ln.strip()]

headlines: list[str] = []
if text_in:
    headlines.extend(parse_lines(text_in))
if file_in is not None:
    headlines.extend(parse_lines(file_in.read().decode("utf-8", errors="ignore")))

# ---------- scoring with client-side cache ----------
def score_with_client_cache(headlines: list[str]):
    ids_out = [None] * len(headlines)
    labels_out = [None] * len(headlines)
    hits = 0
    misses = 0

    to_score = []
    to_score_idx = []
    hit_headlines = []

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
        resp = requests.post(f"{api_url.rstrip('/')}/score_headlines", json=payload, timeout=timeout_s)
        resp.raise_for_status()
        preds = resp.json().get("labels", [])

        for idx, label in zip(to_score_idx, preds):
            labels_out[idx] = label
            cid = ids_out[idx]
            # Maintain max size of 50
            if len(st.session_state.pred_cache) >= CACHE_MAX:
                st.session_state.pred_cache.popitem(last=False)  # FIFO eviction
            st.session_state.pred_cache[cid] = (label, time.time())

    # Build readable source info
    if misses == 0:
        source_info = f"Pulled all {hits} results from cache."
    else:
        source_info = (
            f"API call for {misses} headlines (Cache hits: {hits}).\n\n"
            f"Cache hits:\n- " + "\n- ".join(hit_headlines) if hit_headlines else ""
        )

    return ids_out, labels_out, hits, misses, source_info

# ---------- action ----------
if st.button("Score"):
    if not headlines:
        st.warning("Please add at least one headline.")
    else:
        try:
            ids, labels, hits, misses, source_info = score_with_client_cache(headlines)
            df = pd.DataFrame({"headline": headlines, "label": labels})
            if show_ids:
                df.insert(1, "id", ids)

            st.success(source_info)
            st.dataframe(df, use_container_width=True)

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", data=csv, file_name="headline_scores.csv", mime="text/csv")
        except requests.exceptions.RequestException as e:
            st.error(f"Request failed: {e}")
        except Exception as e:
            st.error(f"Failed: {e}")


