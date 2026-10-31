"""Blind human annotation of ASL image features; no extraction or prediction."""
import argparse
import json
from pathlib import Path

import streamlit as st
from PIL import Image

from gesturelab.datasets.data import read_manifest
from gesturelab.features.asl_annotations import (
    NOT_ASSESSABLE, agreement_report, annotatable_samples, load_annotations,
    save_annotation,
)
from gesturelab.features.asl_schema import FEATURE_SCHEMA, FEATURE_SCHEMA_VERSION

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--project", default="data/asl_kaggle")
args, _ = parser.parse_known_args()
project = Path(args.project).expanduser().resolve()

st.set_page_config(page_title="ASL feature annotation", page_icon=":material/back_hand:", layout="wide")
st.title("Blinde ASL-featureannotatie")
st.write("Beoordeel alleen wat zichtbaar is op de foto. Letterlabel, referentiematrix en modelvoorspellingen worden niet getoond.")
st.caption(f"Featureschema {FEATURE_SCHEMA_VERSION} · handmatige invoer · geen extractie, kalibratie of training")

if not (project / "manifest.json").is_file():
    st.error(f"Datasetmanifest ontbreekt: {project / 'manifest.json'}")
    st.stop()

try:
    manifest = read_manifest(project)
    samples = annotatable_samples(project, manifest)
    annotation_file = load_annotations(project, manifest)
except (OSError, ValueError, json.JSONDecodeError) as error:
    st.error(f"Annotatiegegevens konden niet worden geladen: {error}")
    st.stop()

if not samples:
    st.info("Geen opgenomen developmentbeelden met een geslaagde handdetectie gevonden.")
    st.stop()

annotator = st.text_input("Annotator-ID", help="Gebruik per beoordelaar een vaste, niet-persoonlijk-identificerende code.")
existing = {
    item["sample_id"]: item
    for item in annotation_file["annotations"]
    if item["annotator"] == annotator.strip()
}
sample_by_id = {sample["id"]: sample for sample in samples}
sample_ids = sorted(sample_by_id)
selected_id = st.selectbox("Anonieme foto-ID", sample_ids, format_func=lambda value: value[:12])
sample = sample_by_id[selected_id]
image_path = (project / sample["image_path"]).resolve()
if not image_path.is_relative_to(project) or not image_path.is_file():
    st.error(f"Datasetafbeelding ontbreekt of verwijst buiten de dataset: {image_path}")
    st.stop()

st.image(Image.open(image_path), caption=f"Foto {selected_id[:12]}", use_container_width=True)
st.caption("Markeer ‘Niet te beoordelen’ wanneer de feature op deze foto niet betrouwbaar te beoordelen is. Laat ‘Nog niet beoordeeld’ staan om een waarde over te slaan.")

current = existing.get(selected_id, {}).get("features", {})
category_options = {
    feature["key"]: [category for category in feature["categories"] if category["id"] != "unknown"]
    for feature in FEATURE_SCHEMA
}
selection_labels = {"": "Nog niet beoordeeld", NOT_ASSESSABLE: "Niet te beoordelen"}
for feature_categories in category_options.values():
    selection_labels.update({item["id"]: item["label"] for item in feature_categories})

with st.form("blind_feature_annotation"):
    selected_values = {}
    for feature in FEATURE_SCHEMA:
        key = feature["key"]
        choices = ["", *(item["id"] for item in category_options[key]), NOT_ASSESSABLE]
        previous = current.get(key, "")
        index = choices.index(previous) if previous in choices else 0
        selected_values[key] = st.selectbox(
            feature["title"], choices, index=index,
            format_func=lambda value: selection_labels[value],
            help=f"{feature['measurement']} Beperk de beoordeling tot wat op deze foto aantoonbaar is.",
        )
    submitted = st.form_submit_button("Annotaties opslaan", type="primary")

if submitted:
    if not annotator.strip():
        st.error("Vul eerst een annotator-ID in.")
    else:
        values = {key: value for key, value in selected_values.items() if value}
        try:
            save_annotation(project, sample, annotator, values)
        except (OSError, ValueError) as error:
            st.error(f"Annotaties opslaan is mislukt: {error}")
        else:
            st.success("Deze annotatie is opgeslagen. Andere annotatoren blijven onafhankelijk.")
            st.rerun()

st.subheader("Interbeoordelaars-overeenstemming")
st.caption("Exacte overeenstemming wordt alleen berekend voor dezelfde foto-feature die minstens twee keer beoordeeld is; niet-beoordeelbare waarden tellen niet mee.")
agreement = agreement_report(annotation_file)
st.dataframe(agreement, hide_index=True, alt="Exact agreement between independent feature annotations")
