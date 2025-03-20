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

# 📝 Automatic Topic and Concept Extraction from Text Corpus  

## 📖 Overview  
This project is a tool for **automatic topic and concept extraction** from a text corpus using **Latent Semantic Analysis (LSA)** and **Latent Dirichlet Allocation (LDA)**, two widely used **topic modeling** techniques. These techniques are essential in fields like:  
✔ Information Retrieval  
✔ Automatic Document Categorization  
✔ Sentiment Analysis  

The motivation behind this tool is the massive amount of **unstructured text data**, which makes manual analysis inefficient. By leveraging **LSA and LDA**, this tool can extract meaningful topics without human intervention, helping users **analyze and interpret large datasets**.  

## 🎯 Features  
✅ **Text Preprocessing**: Tokenization, punctuation removal, stopword removal, lemmatization  
✅ **TF-IDF Matrix Generation**: Conversion of text into numerical features  
✅ **Topic Modeling**: Implementation of **LDA** and **LSA** for automatic topic extraction  
✅ **Evaluation Metrics**: Coherence Score, Perplexity Score, and Topic Diversity  
✅ **Visualizations**:  
   - **WordCloud** for word importance  
   - **Heatmaps** for topic-document relationships  
   - **Stacked Bar Charts** for topic distribution  

## 📌 Thesis Reference  
This project is part of my **diploma thesis** at the **University of West Attica**. You can access the full thesis at the **Polynoe Repository**:  
📄 **[Automatic Topic and Concept Extraction from Text Corpus - Polynoe Repository](https://polynoe.lib.uniwa.gr/xmlui/handle/11400/8704)**  

## 🔧 Installation  
To use this tool, install the necessary dependencies:  

```bash
pip install nltk gensim numpy pandas matplotlib seaborn scikit-learn wordcloud
```

## 🚀 How to Run  
Run the script using the following command:  

```bash
python thesis.py
```

The program will prompt you to choose a topic modeling method:  

🔹 Enter **1** for **LDA (Latent Dirichlet Allocation)**  
🔹 Enter **2** for **LSA (Latent Semantic Analysis)**  

After execution, the tool will:  
✔ Display the **extracted topics**  
✔ Print the **coherence score**  
✔ Generate **graphs** (WordCloud, Heatmap, Stacked Bar Chart)  

## 📊 Example Output  
Example of extracted topics using **LDA**:  

```
Topic 1: economy, stock, market, trade, investment  
Topic 2: politics, government, election, policy, law  
Topic 3: technology, AI, innovation, development, software  
```

### 🔍 **Sample WordCloud**  
![WordCloud Example](https://upload.wikimedia.org/wikipedia/commons/thumb/e/e2/Tag-cloud.png/640px-Tag-cloud.png)  

## 🔬 Evaluation  
The extracted topics are evaluated using:  
📌 **Coherence Score**: Measures topic quality  
📌 **Perplexity Score**: Measures model uncertainty  
📌 **Topic Diversity**: Evaluates the uniqueness of topics  

## 📌 Future Improvements  
- 📈 Support for **more topic modeling algorithms** (e.g., BERTopic, NMF)  
- 🌍 Ability to **process multilingual text datasets**  
- 🎯 Optimization of **parameter selection** for improved accuracy  

## 🛠 Dependencies  
This project requires the following **Python libraries**:  

- `nltk` (Natural Language Toolkit)  
- `gensim` (Topic Modeling)  
- `numpy`, `pandas` (Data Processing)  
- `matplotlib`, `seaborn` (Data Visualization)  
- `scikit-learn` (ML utilities)  
- `wordcloud` (WordCloud visualization)  

## 🤝 Contributing  
If you want to contribute to this project, feel free to submit a **pull request** or open an **issue** on GitHub.  

## 📩 Contact  
For any questions or suggestions, feel free to contact me via **email** or **LinkedIn**.  
