# Dutch manual alphabet (NGT)

`ngt_alphabet.png` is a lossless conversion of:
https://geocachen.be/wp-content/uploads/2017/08/handalfabet-Nederland.gif

Source page, which explicitly distinguishes Dutch and Flemish manual alphabets:
https://geocachen.be/geocaching/geocache-puzzels-oplossen/gebarentaal/
Accessed: 26 October 2026. The original creator and reuse license are not stated.
This attribution does not grant a free reuse license.

Background on the Dutch manual alphabet:
https://helenkeller.nl/doofblindheid/communiceren/nederlands-handalfabet/

The 26 files in `assets/alphabet/` are lossless crops of `ngt_alphabet.png`.
Their filenames (`A.png` through `Z.png`) provide the labels; no coordinate
JSON is needed. These images are recording references, not training data.
H, J, U, X and Z have movement arrows in this chart. A still image does not fully
explain execution or movement.

For existing datasets, rerun `python -m scripts.init_dataset --project <dataset-directory>`.
NGT references use their own `references/ngt_alphabet` directory. References to
previously supplied alphabet images are updated; custom reference paths are
preserved. Recorded photos and labels are not translated: check poses recorded
using ASL references before treating them as NGT training data.

Original source images and URLs retain their native wording.
