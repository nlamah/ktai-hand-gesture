"""Review and edit independent ASL letter-feature references; no model training."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from gesturelab.features.asl_references import (
    CELL_STATES, PATTERN_STATES, REFERENCE_PATH, export_matrix_csv, matrix_table,
    export_reference_bundle, export_matrix_html, load_reference_matrix, quality_report, reference_cell,
    save_reference_matrix,
)
from gesturelab.features.asl_schema import FEATURE_SCHEMA, FEATURE_SCHEMA_VERSION, schema_document


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--matrix", default=str(REFERENCE_PATH))
args, _ = parser.parse_known_args()
matrix_path = Path(args.matrix).expanduser().resolve()

st.set_page_config(page_title="ASL letter-feature matrix", page_icon=":material/back_hand:", layout="wide")
st.title("ASL-letterfeaturematrix")
st.write("Bekijk en bewerk onafhankelijk vastgestelde featureverwachtingen per letter. Deze app start geen extractie, tuning of training.")
st.caption(f"Featureschema {FEATURE_SCHEMA_VERSION} · {len(FEATURE_SCHEMA)} features · referentiebestand: {matrix_path}")

try:
    matrix = load_reference_matrix(matrix_path)
except (OSError, ValueError, json.JSONDecodeError) as error:
    st.error(f"De referentiematrix kon niet worden geladen: {error}", icon=":material/error:")
    st.stop()

report = quality_report(matrix)
specified_cells = sum(
    reference_cell(pattern, feature["key"])["state"] == "specified"
    for label in matrix["labels"]
    for pattern in matrix["letters"][label]["patterns"]
    for feature in FEATURE_SCHEMA
)
total_cells = sum(len(matrix["letters"][label]["patterns"]) for label in matrix["labels"]) * len(FEATURE_SCHEMA)
reviewed_patterns = sum(
    pattern["status"] == "verified"
    for label in matrix["labels"]
    for pattern in matrix["letters"][label]["patterns"]
)
metric_columns = st.columns(4)
metric_columns[0].metric("Letters", len(matrix["labels"]))
metric_columns[1].metric("Features per patroon", len(FEATURE_SCHEMA))
metric_columns[2].metric("Ingevulde verwachtingen", f"{specified_cells}/{total_cells}")
metric_columns[3].metric("Gecontroleerde patronen", reviewed_patterns)

overview, detail, diagnostics = st.tabs(["Overzicht", "Letter bewerken", "Onduidelijkheden"])
with overview:
    st.subheader("Verwachtingsmatrix")
    st.caption("‘?’ betekent niet vastgesteld; ‘n.v.t.’ betekent expliciet niet relevant. Conceptuele waarden zijn geen gevalideerde ASL-regels.")
    rows = []
    for row in matrix_table(matrix):
        display_row = {"letter": row["letter"], "patroon": row["pattern"], "status": row["reference status"]}
        display_row.update({
            feature["title"]: row[feature["key"]]
            for feature in FEATURE_SCHEMA
        })
        rows.append(display_row)
    st.dataframe(pd.DataFrame(rows), hide_index=True)

    st.subheader("Export")
    st.download_button(
        "Download JSON",
        data=json.dumps(export_reference_bundle(matrix), indent=2, ensure_ascii=False) + "\n",
        file_name="asl_letter_references.json",
        mime="application/json",
    )
    st.download_button(
        "Download CSV",
        data=export_matrix_csv(matrix),
        file_name="asl_letter_feature_matrix.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download HTML-overzicht",
        data=export_matrix_html(matrix),
        file_name="asl_letter_feature_matrix.html",
        mime="text/html",
    )
    st.download_button(
        "Download featureschema",
        data=json.dumps(schema_document(), indent=2, ensure_ascii=False) + "\n",
        file_name="asl_categorical_feature_schema.json",
        mime="application/json",
    )

with detail:
    st.subheader("Referentie per letter")
    selected_label = st.selectbox("Letter", matrix["labels"])
    entry = matrix["letters"][selected_label]
    pattern_names = [pattern["name"] for pattern in entry["patterns"]]
    selected_pattern_name = st.selectbox("Posepatroon", pattern_names)
    pattern_index = pattern_names.index(selected_pattern_name)
    pattern = entry["patterns"][pattern_index]

    with st.container(border=True):
        st.markdown(f"**ASL-voorbeeld voor {selected_label}**")
        example = entry.get("example") or {"path": "", "source": "", "reviewer": ""}
        if example["path"] and example["source"] and example["reviewer"]:
            example_path = Path(example["path"]).expanduser()
            if not example_path.is_absolute():
                example_path = (matrix_path.parent / example_path).resolve()
            if example_path.is_file():
                st.image(str(example_path), caption=f"Bron: {example['source']} · gecontroleerd door {example['reviewer']}")
            else:
                st.warning(f"ASL-voorbeeldbestand ontbreekt: {example_path}")
        else:
            st.info("Voorbeeld ontbreekt: voeg alleen een bronbeeld met gecontroleerde ASL-herkomst toe; een NGT-afbeelding of gegenereerde illustratie wordt niet als ASL-referentie gebruikt.")

    with st.form("edit_asl_example"):
        example_path_value = st.text_input("Pad naar gecontroleerd ASL-voorbeeld", value=example["path"])
        example_source = st.text_input("Bron van ASL-voorbeeld", value=example["source"])
        example_reviewer = st.text_input("Beoordelaar voorbeeld", value=example["reviewer"])
        save_example = st.form_submit_button("Voorbeeld opslaan")
    if save_example:
        updated = deepcopy(matrix)
        updated["letters"][selected_label]["example"] = {
            "path": example_path_value.strip(),
            "source": example_source.strip(),
            "reviewer": example_reviewer.strip(),
        }
        try:
            save_reference_matrix(updated, matrix_path)
        except (OSError, ValueError) as error:
            st.error(f"Opslaan is mislukt: {error}", icon=":material/error:")
        else:
            st.success("Voorbeeldmetadata opgeslagen.")
            st.rerun()

    st.markdown(f"### {selected_label} · {pattern['name']}")
    st.caption(f"Patroonstatus: {pattern['status']}. Bij ‘gecontroleerd’ zijn bron, beoordelaar en toelichting vereist.")
    feature_keys = [spec["key"] for spec in FEATURE_SCHEMA]
    selected_feature_key = st.selectbox(
        "Feature om te bewerken",
        feature_keys,
        format_func=lambda key: next(spec["title"] for spec in FEATURE_SCHEMA if spec["key"] == key),
    )
    feature = next(spec for spec in FEATURE_SCHEMA if spec["key"] == selected_feature_key)
    current_cell = reference_cell(pattern, selected_feature_key)
    category_labels = {item["id"]: item["label"] for item in feature["categories"] if item["id"] != "unknown"}
    st.caption(
        "Kies hieronder een of meer plausibele categorieën uit de keuzelijst. "
        "Meerdere keuzes zijn mogelijke alternatieven, geen gelijktijdige standen. "
        "‘Onbekend’ is alleen een uitvoer van de automatische extractor en geen referentiewaarde."
    )
    state_labels = {
        "unreviewed": "Nog niet vastgesteld",
        "specified": "Verwachte categorie(ën)",
        "not_relevant": "Niet relevant",
    }

    with st.expander("Featuredefinitie", expanded=True):
        st.write(feature["measurement"])
        st.write(f"Landmarks: {', '.join(map(str, feature['landmarks']))}.")
        st.write(feature["validity"])
        st.warning(feature["limitations"], icon=":material/info:")
        st.write("Instelbare parameters:", feature["parameters"])

    with st.form("edit_reference_cell"):
        state = st.selectbox(
            "Referentiestatus",
            CELL_STATES,
            index=CELL_STATES.index(current_cell["state"]),
            format_func=lambda value: state_labels[value],
        )
        allowed = st.multiselect(
            "Suggestie / toegestane waarde(n)",
            list(category_labels),
            default=current_cell["allowed"],
            format_func=lambda value: category_labels[value],
            disabled=state != "specified",
            help="De opties zijn alle handmatig toegestane categorieën voor deze feature. Laat leeg wanneer je het niet kunt beoordelen.",
        )
        cell_note = st.text_input("Toelichting voor deze feature", value=current_cell.get("note", ""))
        pattern_status = st.selectbox(
            "Status posepatroon",
            PATTERN_STATES,
            index=PATTERN_STATES.index(pattern["status"]),
            format_func=lambda value: "Concept" if value == "concept" else "Gecontroleerd",
        )
        source = st.text_input("ASL-bron (vereist om als gecontroleerd op te slaan)", value=pattern.get("source", ""))
        reviewer = st.text_input("Beoordelaar", value=pattern.get("reviewer", ""))
        note = st.text_input("Toelichting op dit patroon", value=pattern.get("note", ""))
        saved = st.form_submit_button("Referentie opslaan", type="primary")
    if saved:
        updated = deepcopy(matrix)
        target = updated["letters"][selected_label]["patterns"][pattern_index]
        target["status"] = pattern_status
        target["source"] = source.strip()
        target["reviewer"] = reviewer.strip()
        target["note"] = note.strip()
        target.setdefault("features", {})[selected_feature_key] = {
            "state": state,
            "allowed": allowed if state == "specified" else [],
            "note": cell_note.strip(),
        }
        try:
            save_reference_matrix(updated, matrix_path)
        except (OSError, ValueError) as error:
            st.error(f"Opslaan is mislukt: {error}", icon=":material/error:")
        else:
            st.success("Referentie opgeslagen.")
            st.rerun()

    with st.form("add_reference_pattern"):
        new_pattern_name = st.text_input("Naam nieuw toegelaten posepatroon")
        add_pattern = st.form_submit_button("Posepatroon toevoegen")
    if add_pattern:
        if not new_pattern_name.strip():
            st.error("Geef het patroon een naam.")
        elif new_pattern_name.strip() in pattern_names:
            st.error("Deze patroonnaam bestaat al voor deze letter.")
        else:
            updated = deepcopy(matrix)
            updated["letters"][selected_label]["patterns"].append({
                "name": new_pattern_name.strip(),
                "status": "concept",
                "source": "",
                "reviewer": "",
                "note": "",
                "features": {},
            })
            try:
                save_reference_matrix(updated, matrix_path)
            except (OSError, ValueError) as error:
                st.error(f"Opslaan is mislukt: {error}", icon=":material/error:")
            else:
                st.success("Leeg conceptpatroon toegevoegd; alle featurewaarden blijven nog niet vastgesteld.")
                st.rerun()

with diagnostics:
    st.subheader("Ontbrekende en onzekere referenties")
    issue_titles = {
        "missing_references": "Nog niet vastgestelde featurewaarden",
        "concept_pattern": "Conceptueel posepatroon",
        "multiple_allowed_values": "Meerdere toegestane categorieën",
        "no_specified_features": "Nog geen verwachte categorieën",
    }
    if report["issues"]:
        issue_rows = [
            {"letter": item["label"], "patroon": item["pattern"],
             "probleem": issue_titles[item["type"]], "aantal": item["count"]}
            for item in report["issues"]
        ]
        st.dataframe(pd.DataFrame(issue_rows), hide_index=True)
    else:
        st.success("Alle posepatronen hebben een ingevulde status en referentie.")

    st.subheader("Letters met onvoldoende onderscheid")
    unresolved = report["insufficiently_distinguished_pairs"]
    if unresolved:
        st.warning(f"{len(unresolved)} letterparen hebben nog geen aantoonbaar onderscheid op basis van disjuncte, vastgestelde categorieën.")
        st.dataframe(pd.DataFrame([
            {
                "letters": f"{pair['left']} / {pair['right']}",
                "posepatronen": f"{patterns['left_pattern']} / {patterns['right_pattern']}",
                "onderscheidende features": ", ".join(patterns["distinguishing_features"]) or "Geen vastgesteld verschil",
            }
            for pair in unresolved
            for patterns in pair["pattern_pairs"]
            if not patterns["distinguishable"]
        ]), hide_index=True)
    else:
        st.success("Elke lettercombinatie heeft ten minste één onderscheiden feature.")
    st.caption("‘Onvoldoende onderscheid’ betekent dat de huidige referenties geen verschil onderbouwen. Het is geen uitspraak dat de echte gebaren hetzelfde zijn.")
    identical = report["identical_reference_patterns"]
    st.subheader("Identieke volledig ingevulde referentiepatronen")
    if identical:
        st.dataframe(pd.DataFrame([
            {"letters": f"{pair['left']} / {pair['right']}",
             "identieke patronen": ", ".join(
                 f"{pattern['left_pattern']} = {pattern['right_pattern']}"
                 for pattern in pair["patterns"]
             )}
            for pair in identical
        ]), hide_index=True)
    else:
        st.info("Er zijn nog geen volledig ingevulde identieke patronen.")
