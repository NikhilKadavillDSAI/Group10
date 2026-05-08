# Group10 - Hindi Digit Recognition
Introduction to Image and Video Processing course project of group 10 (KEN3238 2025-26)

## Idea:
Classify handwritten Hindi digits 0–9 from 32×32 grayscale images, using techniques from the Labs.

## Approach:
Each notebook follows the same shared pipeline before the classifier-specific part:

### `knn-cnn.ipynb`
1. Median blur, CLAHE, Otsu threshold, morphological closing
2. Centering via bounding-box translation
3. Feature extraction: raw pixels, Laplacian edges, row/column projections
4. KNN classifier with sweep over feature sets and k
5. CNN ensemble (5 models, different seeds, early stopping, averaged softmax)

### `knn-rf.ipynb`
1. Loads images, balances classes by capping at 40 samples per digit during sampling 
2. KNN with k=3 as a quick baseline 
3. Random Forest sweep over n_estimators = {1, 10, 50, 100, 500, 1000}

### `svm.ipynb`
1. Polarity check (auto-invert if background is light)
2. Gaussian blur, Otsu threshold, bounding-box crop, resize digit to fit a 24×24 region in a 32×32 canvas
3. HOG features
4. Linear SVM trained via SGDClassifier with hinge loss

## How to run:
Each notebook is self-contained, thus open in Jupyter and run top to bottom.
They expect train/, test/, train.csv, and test.csv in the project root.

## Dependencies:
- numpy, pandas, opencv-python, matplotlib
- scikit-learn (KNN, RF, SVM)
- tensorflow (CNN only)
- joblib (SVM only, for model persistence)
