"""Explore landmark feature agreement with an A–Z reference workbook."""
import argparse
import json
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from gesturelab.features.asl_excel_review import (
    analyze_split,
    evaluate_samples,
    load_landmark_cache,
    load_reference_workbook,
    read_coco_split,
    save_landmark_cache,
)
from gesturelab.features.asl_schema import PARAMETER_DEFAULTS


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--dataset", default="data/external/asl_letters_v1/train")
parser.add_argument("--reference", default=str(Path(__file__).resolve().parents[1] / "assets" / "asl_feature_references.xlsx"))
parser.add_argument("--cache", default="data/reports/asl_letters_v1_train_landmarks.json")
args, _ = parser.parse_known_args()

st.set_page_config(
    page_title="ASL landmark feature check",
    page_icon=":material/back_hand:",
    layout="wide",
)
st.title("ASL-features controleren")
st.write(
    "Bereken handfeatures uit MediaPipe-landmarks en vergelijk ze met de "
    "letterwaardes uit je Excel-sheet. De app gebruikt alleen de trainingssplit "
    "en traint geen letterherkenningsmodel."
)

reference_path = Path(args.reference).expanduser()
uploaded_reference = st.file_uploader("Excel-referentie (.xlsx)", type=["xlsx"])
if uploaded_reference is not None:
    reference_bytes = uploaded_reference.getvalue()
elif reference_path.is_file():
    reference_bytes = reference_path.read_bytes()
    st.caption(f"Referentie: {reference_path}")
else:
    st.error(f"Excel-bestand niet gevonden: {reference_path}")
    st.stop()

try:
    references = load_reference_workbook(reference_bytes)
except (OSError, ValueError, json.JSONDecodeError) as error:
    st.error(f"Excel-referentie kon niet worden ingelezen: {error}")
    st.stop()

dataset_directory = Path(args.dataset).expanduser().resolve()
cache_path = Path(args.cache).expanduser().resolve()
try:
    dataset_samples = read_coco_split(dataset_directory)
except (OSError, ValueError, json.JSONDecodeError) as error:
    st.error(f"Trainingssplit kon niet worden gelezen: {error}")
    st.stop()

st.caption(
    f"Trainingssplit: {dataset_directory} · {len(dataset_samples)} afbeeldingen · "
    f"{len(references)} letterreferenties"
)

try:
    cache = load_landmark_cache(cache_path, dataset_directory)
except (OSError, ValueError, json.JSONDecodeError) as error:
    st.error(f"Landmarkcache kon niet worden gelezen: {error}")
    st.stop()

if cache is None:
    st.info(
        "Er is nog geen geldige landmarkcache. De eerste berekening kan even duren; "
        "de oorspronkelijke afbeeldingen en COCO-annotaties worden niet aangepast."
    )
    if st.button("Landmarks berekenen voor de train-split", type="primary"):
        progress = st.progress(0, text="Landmarks berekenen…")
        try:
            cache = analyze_split(
                dataset_directory,
                progress=lambda done, total: progress.progress(
                    done / total, text=f"Afbeelding {done}/{total}"
                ),
            )
            save_landmark_cache(cache, cache_path)
        except (OSError, ValueError, RuntimeError) as error:
            st.error(f"Landmarkberekening is mislukt: {error}")
        else:
            st.success(f"Landmarkcache opgeslagen: {cache_path}")
            st.rerun()
    st.stop()

with st.expander("Drempels aanpassen", expanded=False):
    st.caption(
        "Veranderingen herberekenen alleen de categorieën uit de opgeslagen landmarks; "
        "de afbeeldingen worden niet opnieuw geanalyseerd."
    )
    flexion_col, position_col, spread_col = st.columns(3)
    with flexion_col:
        extended_min = st.slider(
            "Gestrekt vanaf (graden)",
            140,
            180,
            int(PARAMETER_DEFAULTS["flexion"]["extended_min_deg"]),
        )
        bent_max = st.slider(
            "Gebogen tot (graden)",
            70,
            min(150, extended_min - 1),
            int(PARAMETER_DEFAULTS["flexion"]["bent_max_deg"]),
        )
    with position_col:
        contact_max = st.slider(
            "Contactafstand maximaal",
            0.05,
            0.60,
            float(PARAMETER_DEFAULTS["thumb_position"]["contact_max_ratio"]),
            step=0.01,
        )
        near_max = st.slider(
            "Nabijheidsafstand maximaal",
            max(0.61, round(contact_max + 0.01, 2)),
            1.20,
            max(
                float(PARAMETER_DEFAULTS["thumb_position"]["near_max_ratio"]),
                round(contact_max + 0.01, 2),
            ),
            step=0.01,
        )
        thumb_neutral = st.slider(
            "Duim-MCP neutraal tot (graden)",
            0,
            60,
            int(PARAMETER_DEFAULTS["thumb_mcp_position"]["neutral_max_deviation_deg"]),
        )
    with spread_col:
        spread_min = st.slider(
            "Spreiding vanaf (verhouding)",
            0.10,
            0.90,
            float(PARAMETER_DEFAULTS["spread"]["minimum_spread_ratio"]),
            step=0.01,
        )
        crossing_min = st.slider(
            "Kruisingmarge (verhouding)",
            0.00,
            0.25,
            float(PARAMETER_DEFAULTS["crossing"]["minimum_lateral_swap_ratio"]),
            step=0.01,
        )

parameters = {
    "flexion": {
        "extended_min_deg": float(extended_min),
        "bent_max_deg": float(bent_max),
    },
    "spread": {"minimum_spread_ratio": float(spread_min)},
    "thumb_position": {
        "contact_max_ratio": float(contact_max),
        "near_max_ratio": float(near_max),
    },
    "thumb_mcp_position": {"neutral_max_deviation_deg": float(thumb_neutral)},
    "crossing": {"minimum_lateral_swap_ratio": float(crossing_min)},
}

results, feature_stats = evaluate_samples(cache["samples"], references, parameters)
exact_matches = sum(result["image_status"] == "match" for result in results)
needs_tuning = sum(result["image_status"] == "needs_tuning" for result in results)
detection_issues = sum(
    result["image_status"] == "needs_detection_review" for result in results
)
metric_columns = st.columns(4)
metric_columns[0].metric("Trainbeelden", len(results))
metric_columns[1].metric("Alle features matchen", exact_matches)
metric_columns[2].metric("Features tweaken", needs_tuning)
metric_columns[3].metric("Detectie nalopen", detection_issues)

feature_rows = []
for feature, stats in feature_stats.items():
    feature_rows.append({
        "Feature": feature,
        "Match": stats["matches"],
        "Verschil": stats["mismatches"],
        "Onbekend": stats["unknown"],
        "Niet gescoord": stats["not_scored"],
        "Matchpercentage": (
            f"{stats['match_rate']:.1%}" if stats["match_rate"] is not None else "—"
        ),
    })
st.subheader("Resultaat per feature")
st.dataframe(pd.DataFrame(feature_rows), hide_index=True)

with st.expander("Foto's filteren"):
    filter_columns = st.columns(2)
    selected_letter = filter_columns[0].selectbox(
        "Letter", ["Alle letters", *sorted(references)]
    )
    selected_status = filter_columns[1].selectbox(
        "Resultaat",
        ["Alle", "Alles matcht", "Moet getweakt worden", "Detectie nalopen"],
    )

filtered = results
if selected_letter != "Alle letters":
    filtered = [item for item in filtered if item["label"] == selected_letter]
status_lookup = {
    "Alles matcht": "match",
    "Moet getweakt worden": "needs_tuning",
    "Detectie nalopen": "needs_detection_review",
}
if selected_status != "Alle":
    filtered = [
        item for item in filtered
        if item["image_status"] == status_lookup[selected_status]
    ]

status_labels = {
    "match": "Goed",
    "needs_tuning": "Tweak nodig",
    "needs_detection_review": "Detectie nalopen",
}
summary_rows = [{
    "Afbeelding": item["filename"],
    "Letter": item["label"],
    "Resultaat": status_labels[item["image_status"]],
    "Matchpercentage": (
        f"{item['match_rate']:.1%}" if item["match_rate"] is not None else "—"
    ),
    "Verschillen": item["mismatches"],
    "Onbekend": item["unknown"],
    "Handdetectie": item.get("status", "geen resultaat"),
} for item in filtered]
st.subheader(f"Afbeeldingen ({len(filtered)})")
st.dataframe(pd.DataFrame(summary_rows), hide_index=True)
st.download_button(
    "Download resultaten als CSV",
    data=pd.DataFrame(summary_rows).to_csv(index=False),
    file_name="asl_landmark_feature_review.csv",
    mime="text/csv",
)

if filtered:
    selected_filename = st.selectbox(
        "Bekijk afbeelding",
        [item["filename"] for item in filtered],
        format_func=lambda filename: next(
            f"{status_labels[item['image_status']]} · {item['label']} · {filename}"
            for item in filtered if item["filename"] == filename
        ),
    )
    selected = next(item for item in filtered if item["filename"] == selected_filename)
    with Image.open(selected["image_path"]) as image:
        st.image(
            image.copy(),
            caption=f"Letter {selected['label']} · {selected['filename']}",
            width=500,
        )
    detail_rows = [{
        "Feature": feature,
        "Excel verwacht": comparison["expected"],
        "Landmarkuitkomst": comparison["actual"],
        "Status": {
            "match": "Match",
            "mismatch": "Afwijking",
            "unknown": "Onbekend",
            "not_scored": "Niet gescoord",
        }[comparison["status"]],
        "Meting": json.dumps(comparison.get("measurement", {}), ensure_ascii=False),
    } for feature, comparison in selected["comparisons"].items()]
    st.dataframe(pd.DataFrame(detail_rows), hide_index=True)

st.warning(
    "De duim-MCP-vormen B/C/L zijn voorlopige geometrische schattingen: "
    "B = buiging naar de handpalm, L = van de hand af, C = duimtop nabij de "
    "wijsvingertop met gebogen IP. Beoordeel die afwijkingen visueel voordat "
    "je drempels of definities als definitief overneemt."
)
