# NLP Assignment 1 - Group 108

Part-of-Speech Tagging using a from-scratch Bigram Hidden Markov Model and Viterbi decoding.

## Team Members

| Name | BITS ID |
|---|---|
| ARTHIKA G | 2025ab05180 |
| SRINEVEDA R S | 2025ab05206 |
| ASWATHY H | 2025ab05203 |
| PRASHANT | 2025ab05113 |
| SUKANYA YADAV | 2025aa05630 |

## Project Structure

```text
notebooks/   Main Jupyter notebook for execution and export
outputs/     Generated charts and matrix images
reports/     Exported HTML/PDF reports
screenshots/ Required lab and inference screenshots
src/         Notebook builder utility
```

## How To Run

```powershell
py -m pip install -r requirements.txt
py src/build_group108_notebook.py
py -m jupyter nbconvert --execute --inplace notebooks/Group_108_HMM_POS_Tagging.ipynb --ExecutePreprocessor.timeout=900
py -m jupyter nbconvert --to html notebooks/Group_108_HMM_POS_Tagging.ipynb --output-dir reports
```

The notebook downloads required NLTK resources automatically when needed.

## Full-Marks Coverage Checklist

- Brown corpus loaded with `news` as training category and `fiction` as test category.
- EDA summary and POS tag distribution chart.
- Vocabulary/OOV handling with rare-word unknown classes.
- Sentence boundary padding using `<START>` and `<END>`.
- Explicit transition and emission probability estimation from counts.
- From-scratch Viterbi decoder using log probabilities.
- Inference output for 2 training and 2 test sentences.
- Three misclassified sentence examples with brief error analysis.
- Token-level HMM accuracy on the 20 percent fiction test set.
- Confusion matrix for the top 5 POS tags.
- Pre-trained baseline comparison using NLTK's PerceptronTagger.
- Markdown discussion explaining HMM limitations versus baseline strengths.
- Screenshots folder reserved for CSIS/OSHA lab and prediction evidence.
