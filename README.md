## Automatic Topic and Concept Extraction from Text Corpus
This program implements a process for automatic topic and concept extraction from a text corpus using topic modeling techniques. Specifically, it uses two main methods for topic extraction: Latent Dirichlet Allocation (LDA) and Latent Semantic Analysis (LSA). The goal of this program is to analyze a set of texts and extract the main topic. The program uses the Reuters Corpus, which contains news articles, for training and evaluating the models. Additionally, coherence scores are calculated to evaluate the quality of the extracted topics.
## Thesis Reference
This project is part of my diploma thesis at the University of West Attica. You can find the full thesis at the following link:

🔗 [Automatic Topic and Concept Extraction from Text Corpus - Polynoe Repository](https://polynoe.lib.uniwa.gr/xmlui/handle/11400/8704)

## Contents
1. Text Preprocessing: Preprocessing of text data (tokenization, punctuation removal, stopwords removal, lemmatization).
2. TF-IDF Matrix Creation: Transformation of the text data into numerical features using the TF-IDF technique.
3. Topic Modeling: Application of two methods for topic extraction: LDA and LSA.
4. Coherence Score Calculation: A method to evaluate the quality of the extracted topics.
5. Presentation of graphs and heatmaps to show topic distribution and document-topic probabilities.

## Dependencies

To run this program, you will need the following Python libraries:
nltk
- gensim
- numpy
- pandas
- matplotlib
- seaborn 
- scikit-learn
- wordcloud

You can install the required libraries using the following command:
```bash
pip install nltk gensim numpy pandas matplotlib seaborn scikit-learn wordcloud
```
## How to Use
Run the script using the following command:
```bash
python .\thesis.py
```

The program will prompt you to choose a topic modeling method:

Enter 1 for LDA (Latent Dirichlet Allocation).

Enter 2 for LSA (Latent Semantic Analysis).

The program will apply the selected method to the TF-IDF matrix and display the extracted topics.

The coherence score will be printed to help evaluate the quality of the extracted topics.

Several visualizations, including topic distribution, heatmap, and stacked bar chart, will be displayed to help understand the topic modeling results.

