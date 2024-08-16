## Non-Intrusive WiFi Sensing and Machine Learning for Frailty Classification in Community-Dwelling Older Adults

This repository contains the code and resources for a novel machine learning model designed to classify frailty in older adults using a non-intrusive WiFi-based motion sensor. The model is part of a feasibility study that aims to explore remote physical activity monitoring and its effectiveness in early frailty detection.

## Background

The ageing population is accompanied by an increase in health issues such as frailty, which has become a significant public health concern. Early detection of frailty can greatly improve the quality of life and longevity of older adults while reducing the burden on healthcare systems. This project explores the feasibility of using artificial intelligence and machine learning for the early detection and management of frailty through non-intrusive methods.

## Objective

This project investigates the performance of a novel WiFi-based motion sensor combined with a machine learning model for remote physical activity monitoring and frailty classification in older adults. The goal is to develop a non-intrusive, effective, and easy-to-use system for frailty detection.

## Methodology

### Data Collection

- **Participants:** The study involved four older adult participants from Montreal, Quebec, Canada. The participants were aged 65 and above, with two identified as non-frail and two as potentially frail.
- **Data Sources:** Data were collected using WiFi signals over six months to monitor contextual human activity and sleep-related patterns. Additionally, participants completed the Geriatric Depression Scale (GDS) and Edmonton Frailty Scale (EFS) questionnaires.

### Model Architecture

The proposed model employs a three-stage architecture:

1. **Feature Selection:** Sequential Forward Selection (SFS) is used to identify the most relevant features.
2. **Dimensionality Reduction:** Principal Component Analysis (PCA) reduces the dimensionality of the selected features.
3. **Classification:** Logistic Regression (LR) classifies participants into non-frail and potentially frail categories.

The model was evaluated using various combinations of techniques for each stage, and the best-performing combination was selected.

## Results

The model demonstrated high classification performance with the following metrics:

- **Accuracy:** 90.00%
- **Sensitivity:** 95.00%
- **Precision:** 88.34%

The model successfully identified key features related to sleep interruptions that have a strong correlation with frailty.

