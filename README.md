# Amazon-Food-Reviews-Sentimental-Analysis-# Sentiment Analysis in Python

This repository contains a comprehensive tutorial and code examples demonstrating sentiment analysis on Amazon Fine Food Reviews using two different techniques:

- **VADER (Valence Aware Dictionary and sEntiment Reasoner)**: A lexicon and rule-based sentiment analysis tool that uses a bag-of-words approach.
- **RoBERTa Pretrained Model**: A transformer-based model leveraging the `cardiffnlp/twitter-roberta-base-sentiment` pretrained model for contextual sentiment classification.

Additionally, there is a demonstration of using Huggingface's Transformers pipeline for quick sentiment predictions.

---

## Dataset

The dataset used is a subset of the **Amazon Fine Food Reviews** dataset:

- Columns include:  
  `Id`, `ProductId`, `UserId`, `ProfileName`, `HelpfulnessNumerator`,  
  `HelpfulnessDenominator`, `Score`, `Time`, `Summary`, `Text`

- For demo purposes, only the first 500 reviews are utilized.

---

## Features

- Data loading and basic exploratory data analysis (EDA)
- Tokenization and named entity recognition using NLTK
- Sentiment scoring with VADER including positive/neutral/negative/compound scores
- Sentiment scoring using RoBERTa transformer model with softmax normalization
- Comparison between VADER and RoBERTa sentiment outputs
- Visualization of sentiment scores against user review stars
- Examples of cases with differing sentiments between star ratings and model predictions
- Quick sentiment analysis with Huggingface pipeline

---

## Installation

1. Clone this repository:

    ```
    git clone https://github.com/yourusername/sentiment-analysis-python.git
    cd sentiment-analysis-python
    ```

2. Create and activate a virtual environment (optional but recommended):

    ```
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. Install the dependencies:

    ```
    pip install pandas numpy matplotlib seaborn nltk tqdm transformers scipy
    ```

4. Download NLTK resources (if not already installed):

    ```
    import nltk
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('maxent_ne_chunker')
    nltk.download('words')
    nltk.download('vader_lexicon')
    ```

---

## Usage

1. Open and run the Jupyter notebook or Python script provided in this repo.
2. Follow the steps to:
    - Load the dataset
    - Explore the data
    - Run VADER sentiment analysis
    - Run RoBERTa sentiment analysis
    - Visualize results and compare models
3. Use the included **testing code** to validate the sentiment models on custom sentences.

### Example: Testing Sentiment Models on Custom Sentences

