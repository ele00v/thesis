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
   - and more

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
✔ Print the **coherence score**  , **Perplexity Score**, and **Topic Diversity ** 
✔ Generate **graphs** (WordCloud, Heatmap, Stacked Bar Chart etc.)  

## 📊 Example Output  
Example of extracted topics using **LDA**:  

```
Topics:
['economy', 'stock', 'market', 'trade', 'investment']  
['politics', 'government', 'election', 'policy', 'law']  
['technology', 'AI', 'innovation', 'development', 'software']  
```

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
