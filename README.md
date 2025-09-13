# MNExplainer

This is the repo of the paper "MUSE-Explainer : Counterfactual explanations for Symbolic Music Graph classification models" published at the CMMR 2025 conference.
MUSE-Explainer is a new method that helps reveal how music Graph Neural Network models make decisions by providing clear, human-friendly explanations. Our approach generates counterfactual explanations by bringing local meaningful changes to music score graphs that alter a model’s prediction while ensuring the results remain musically coherent. Unlike existing methods, MUSE-Explainer tailors its explanations to the structure of musical data and avoids unrealistic or confusing outputs.

To use this repository, one can simply install the repo, and create an instance of the explainer in the same fashion as showcased in the ```__main__.py``` file. It is necessary to provide the model and the input graphs to explain.

# Citation

If you use our tool, please cite:

```tex
@inproceedings{hilaire2025museexplainer,
    title={MUSE-Explainer : Counterfactual explanations for Symbolic Music Graph classification models},
    author={Hilaire, Baptiste and Karystinaios, Emmanouil and Widmer, Gerhard},
    booktitle={International Symposium on Computer Music Multidisciplinary Research (CMMR)},
    year={2025}
}
```
