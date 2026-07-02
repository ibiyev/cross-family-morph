# Cross-Family Morpheme Segmentation OOD Evaluation Dataset

## Overview
800 manually-annotated words (200 each in Azerbaijani, Turkish, Hungarian,
Finnish) drawn from Wikipedia, with gold morpheme segmentations. Created to
evaluate out-of-domain generalization of morpheme segmentation models trained
on UniMorph-derived data.

## Files
- `gold_all_languages.csv` / `.jsonl` — all 800 items
- `gold_{aze,tur,hun,fin}.csv` — per-language subsets

## Fields
| Field | Description |
|---|---|
| language | ISO-style code (aze/tur/hun/fin) |
| language_full | Full language name |
| family | Turkic or Uralic |
| word | Surface word form from Wikipedia |
| gold_segmentation | Adjudicated gold morpheme segmentation (lowercase, ` + ` separated) |
| n_morphemes | Number of gold morphemes |
| model_prediction_seed42 | ByT5-small prediction (reference) |
| gold_source | "consensus" (both annotators agreed) or "adjudicated" |
| source_article | Wikipedia article the word came from |
| sentence | Full sentence context |

## Annotation
Each word was independently annotated by two native or near-native speakers
(8 annotators total). Inter-annotator agreement on the binary
model-correctness judgment was Cohen's κ = 1.00 (n=800); exact gold-string
agreement averaged 92.4%. The 64 disagreements (8%) were adjudicated per a
shared segmentation policy (compounds decomposed; buffer consonants separated).

## Segmentation conventions
- All lowercase
- Morphemes separated by ` + ` (space-plus-space)
- Compounds decomposed to constituents
- Buffer/linker consonants treated as separate morphemes
- Monomorphemic words written as-is with no separator

## License
Source text from Wikipedia (CC BY-SA 3.0). Annotations released under
[choose: CC BY 4.0 / CC BY-SA 4.0].

## Citation
[Your paper citation here]