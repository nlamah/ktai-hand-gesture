"""Apply the archived slash proposal in memory; retain one source workbook."""
from io import BytesIO
import json
from pathlib import Path
from openpyxl import load_workbook
from research.workbook import load_reference_workbook as parse_workbook, REFERENCE_COLUMNS, ENGLISH_REFERENCE_COLUMNS, _normalize, _expected_values
ROOT=Path(__file__).resolve().parents[1]
def proposal_bytes(contents):
    book=load_workbook(BytesIO(contents));sheet=book.active
    rows={str(sheet.cell(r,1).value).strip():r for r in range(1,sheet.max_row+1)}
    for change in json.loads((ROOT/'config/proposal_changes.json').read_text()):
        cell=sheet.cell(rows[change['letter']],change['column']+1)
        if cell.value!=change['before']:
            feature={**REFERENCE_COLUMNS,**ENGLISH_REFERENCE_COLUMNS}.get(_normalize(sheet.cell(1,change['column']+1).value))
            if feature is None or _expected_values(feature,cell.value)!=_expected_values(feature,change['before']):
                raise ValueError('Source workbook differs from frozen proposal')
        cell.value=change['after']
    stream=BytesIO();book.save(stream);return stream.getvalue()
def load_reference_workbook(contents):return parse_workbook(proposal_bytes(contents))
