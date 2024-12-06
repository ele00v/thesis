"""--------------------------------------------------------------
    Πανεπιστήμιο Δυτικής Αττικής 
    Σχολή Μηχανικών 
    Τμήμα Μηχανικών Πληροφορικής και Υπολογιστών 
    Ονοματεπώνυμο: Ελένη Βέρα 
    Αριθμός Μητρώου:18390152
    Επιβλέπων Καθηγητής: Φοίβος Μυλωνάς, Αναπληρωτής Καθηγητής
    Διπλωματική Εργασία 
    Τίτλος: Αυτόματη εξαγωγή θέματος και εννοιών από σώμα κειμένων
    Title: Automatic topic and concept extraction from text corpus
    Ακαδημαϊκό Έτος: 2024-2025 
-------------------------------------------------------------------"""
import re
import nltk
import string
import numpy as np
import pandas as pd
from gensim import corpora
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from gensim.corpora import Dictionary
from nltk.tokenize import word_tokenize
from sklearn.model_selection import GridSearchCV
from nltk.corpus import reuters,wordnet,stopwords
from gensim.models.phrases import Phrases, Phraser
from nltk.stem import PorterStemmer,WordNetLemmatizer
from gensim.models.coherencemodel import CoherenceModel
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD,LatentDirichletAllocation

# Αναγκαία corpora και λεξικά για το NLTK
#nltk.download('reuters')
#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('wordnet')

# Αρχικοποίηση lemmatizer
lemmatizer = WordNetLemmatizer()
# Αρχικοποίηση stemmer
stemmer = PorterStemmer()
# Αρχικοποίηση stopwords
stop_words = set(stopwords.words('english'))

# Συνάρτηση για την προεπεξεργασία του κειμένου
def preprocess_text(text):
    # Μετατροπή του κειμένου σε πεζά γράμματα
    text = text.lower()
    # Αφαίρεση αριθμών
    text = ''.join([i for i in text if not i.isdigit()])
    # Αφαίρεση στίξης
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Αφαίρεση κενών χαρακτήρων στην αρχή και στο τέλος
    text = text.strip()
    # Διαχωρισμός του κειμένου σε λέξεις (Tokenization)
    tokens = word_tokenize(text)
    # Αφαίρεση των stopwords και μη αλφαβητικών λέξεων
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [word for word in tokens if word not in stop_words]
    # Εφαρμογή POS tagging
    pos_tags = nltk.pos_tag(tokens)
    # Lemmatization και stemming
    lemmatized_tokens = []
    for word, tag in pos_tags:
        pos = get_wordnet_pos(tag)
        lemmatized_word = lemmatizer.lemmatize(word, pos=pos)
        # Εφαρμογή stemming και αφαίρεση λέξεων με μήκος μικρότερο από 2 χαρακτήρες
        if len(lemmatized_word) > 2:
            stemmed_word = stemmer.stem(lemmatized_word)
            lemmatized_tokens.append(stemmed_word)
    # Συνένωση των tokens σε μία συμβολοσειρά
    return ' '.join(lemmatized_tokens)

# Συνάρτηση για τη μετατροπή των ετικετών POS του NLTK σε μορφή POS του WordNet
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ  # Επίθετο
    elif treebank_tag.startswith('V'):
        return wordnet.VERB  # Ρήμα
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN  # Ουσιαστικό
    elif treebank_tag.startswith('R'):
        return wordnet.ADV   # Επίρρημα
    else:
        return wordnet.NOUN  # Default

# Λίστα με τα διαθέσιμα έγγραφα του Reuters corpus
documents = reuters.fileids()

# Διαχωρισμός των εγγράφων σε εκπαιδευτικά (training) και δοκιμαστικά (test)
train_docs = [doc for doc in documents if doc.startswith('training/')]
test_docs = [doc for doc in documents if doc.startswith('test/')]

# Επεξεργαζόμαστε τα πρώτα n εκπαιδευτικά έγγραφα
n = 500 

# Δημιουργούμε ένα DataFrame για να αποθηκεύσουμε τα δεδομένα από το Reuters corpus
df_reuters = pd.DataFrame({
    'ID': test_docs[:n],
    'Text': [reuters.raw(doc_id) for doc_id in train_docs[:n]]
})

# Εφαρμόζουμε τη συνάρτηση preprocess_text στη στήλη με τα κείμενα
df_reuters['Processed_Text'] = df_reuters['Text'].apply(preprocess_text)

# Μετατρέπουμε τα κείμενα σε λίστες λέξεων (tokens)
tokenized_texts = [doc.split() for doc in df_reuters['Processed_Text']]

# Εντοπισμός bigrams και trigrams
phrases = Phrases(tokenized_texts, min_count=2, threshold=5)  # min_count και threshold μπορούν να προσαρμοστούν
bigram = Phraser(phrases)

# Εφαρμογή των bigrams στα tokenized texts
texts_with_bigrams = [bigram[doc] for doc in tokenized_texts]

# Δημιουργούμε τη μήτρα TF-IDF χρησιμοποιώντας τον TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df_reuters['Processed_Text'])

# Μετατρέπουμε τη μήτρα σε DataFrame για καλύτερη οπτικοποίηση
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())

# Προσθέτουμε τη στήλη 'ID'
tfidf_df.insert(0, 'ID', df_reuters['ID'])

# Εκτύπωση της μήτρας TF-IDF
#print("TF-IDF Matrix:")
#print(tfidf_df)

# Αποθήκευση της μήτρας TF-IDF σε αρχείο κειμένου
#with open('tfidf_matrix.txt', 'w', encoding='utf-8') as f:
    #f.write(tfidf_df.to_string(index=False))
    
# Συνάρτηση για την εφαρμογή του LDA και εξαγωγή των θεμάτων και της αντιστοίχισης θεμάτων-κειμένων
def apply_lda(tfidf_matrix, vectorizer, num_topics=5, num_words=10):
    # Δημιουργία του LDA μοντέλου με τον καθορισμένο αριθμό θεμάτων
    lda = LatentDirichletAllocation(n_components=num_topics, learning_method='online', max_iter=50, random_state=0)
    
    # Εκπαίδευση του μοντέλου στο TF-IDF
    lda.fit(tfidf_matrix)
    
    # Εξαγωγή των χαρακτηριστικών (λέξεων) για κάθε θέμα
    feature_names = vectorizer.get_feature_names_out()
    topics = []
    
    for topic_idx, topic in enumerate(lda.components_):
        # Τα 10 πιο σημαντικά χαρακτηριστικά για κάθε θέμα
        top_features_idx = topic.argsort()[-num_words:][::-1]
        top_features = [feature_names[i] for i in top_features_idx]
        topics.append(f"Topic {topic_idx + 1}: " + ", ".join(top_features))
    
    # Υπολογισμός του πίνακα θεμάτων-κειμένων
    document_topic_matrix = lda.transform(tfidf_matrix)
    
    return topics, document_topic_matrix


# Συνάρτηση για την εφαρμογή του LSA και εξαγωγή των θεμάτων και της αντιστοίχισης θεμάτων-κειμένων
def apply_lsa(tfidf_matrix, vectorizer, num_topics=5, num_words=10):
    # Δημιουργία του μοντέλου LSA με χρήση TruncatedSVD
    lsa = TruncatedSVD(n_components=num_topics, n_iter=100, random_state=0)
    
    # Εφαρμογή του LSA στη μήτρα TF-IDF
    lsa.fit(tfidf_matrix)
    
    # Εξαγωγή των χαρακτηριστικών (λέξεων) για κάθε θέμα
    feature_names = vectorizer.get_feature_names_out()
    topics = []
    
    for topic_idx, topic in enumerate(lsa.components_):
        # Τα 10 πιο σημαντικά χαρακτηριστικά για κάθε θέμα
        top_features_idx = topic.argsort()[-num_words:][::-1]
        top_features = [feature_names[i] for i in top_features_idx]
        topics.append(f"Topic {topic_idx + 1}: " + ", ".join(top_features))
    
    # Υπολογισμός του πίνακα θεμάτων-κειμένων
    document_topic_matrix = lsa.transform(tfidf_matrix)
    
    return topics, document_topic_matrix


# Συνάρτηση για τον υπολογισμό του coherence score
def calculate_coherence(topics, texts, dictionary, coherence_type='c_v'):
    coherence_model = CoherenceModel(topics=topics, texts=texts, dictionary=dictionary, coherence=coherence_type)
    coherence_score = coherence_model.get_coherence()
    return coherence_score

texts = [doc.split() for doc in df_reuters['Processed_Text']]
dictionary = corpora.Dictionary(texts)

 # Συνάρτηση για επιλογή μεθόδου ανάλυσης θεμάτων
def choose_topic_modeling_method(tfidf_matrix, vectorizer, num_topics=5, num_words=10):
    print("\nChoose a Topic Modeling Method:")
    print("1. Latent Dirichlet Allocation (LDA)")
    print("2. Latent Semantic Analysis (LSA)")
    choice = input("Enter 1 for LDA or 2 for LSA: ").strip()
    if choice == '1':
        print("\nApplying LDA...")
        topics, document_topic_matrix = apply_lda(tfidf_matrix, vectorizer, num_topics, num_words)
        topic_words = [topic.split(": ")[1].split(", ") for topic in topics]
        coherence_score = calculate_coherence(topic_words, texts, dictionary, coherence_type='c_v')
        print(f"\nLDA Coherence Score: {coherence_score:.4f}")
        return topics, document_topic_matrix
    elif choice == '2':
        print("\nApplying LSA...")
        topics, document_topic_matrix = apply_lsa(tfidf_matrix, vectorizer, num_topics, num_words)
        topic_words = [topic.split(": ")[1].split(", ") for topic in topics]
        coherence_score = calculate_coherence(topic_words, texts, dictionary, coherence_type='c_v')
        print(f"\nLSA Coherence Score: {coherence_score:.4f}")
        return topics, document_topic_matrix
    else:
        print("\nInvalid choice. Please enter 1 or 2.")
        return choose_topic_modeling_method(tfidf_matrix, vectorizer, num_topics, num_words)

# Κύριο πρόγραμμα
if __name__ == "__main__":
    num_topics = 5
    num_words = 10
    topics, document_topic_matrix = choose_topic_modeling_method(tfidf_matrix, vectorizer, num_topics, num_words)
    
    print("\nTopics:")
    for topic in topics:
        print(topic)
    
    #print("\nDocument-Topic Probabilities:")
    #for i, doc_probabilities in enumerate(document_topic_matrix):
        #dominant_topic = doc_probabilities.argmax()
        #formatted_probabilities = [f"{prob:.2f}" for prob in doc_probabilities]
        #print(f"Document {i + 1}: Dominant Topic {dominant_topic + 1}, Probabilities: {formatted_probabilities}")

    # Μετράμε πόσα έγγραφα αντιστοιχούν σε κάθε θέμα
    dominant_topics = document_topic_matrix.argmax(axis=1)  # Βρίσκουμε το κυρίαρχο θέμα για κάθε έγγραφο
    topic_counts = np.bincount(dominant_topics, minlength=num_topics)  # Μετράμε τον αριθμό των εγγράφων ανά θέμα

    # Δημιουργούμε ένα διάγραμμα για την κατανομή των εγγράφων στα θέματα
    plt.figure(figsize=(6, 6))
    plt.bar(range(1, num_topics + 1), topic_counts, color='skyblue', edgecolor='black')
    plt.xlabel('Topic', fontsize=14)
    plt.ylabel('Number of Documents', fontsize=14)
    plt.title('Document Distribution Across Topics', fontsize=16)
    plt.xticks(range(1, num_topics + 1))  
    plt.show()

    import seaborn as sns

    # Δημιουργία heatmap για τις πιθανότητες θεμάτων-κειμένων
    plt.figure(figsize=(6, 6))  # Μέγεθος του γραφήματος
    sns.heatmap(document_topic_matrix, annot=False, cmap='coolwarm', cbar=True)
    plt.title('Heatmap των Πιθανοτήτων Θεμάτων-Κειμένων', fontsize=16)
    plt.xlabel('Θέματα', fontsize=14)
    plt.ylabel('Έγγραφα', fontsize=14)
    plt.xticks(ticks=np.arange(num_topics) + 0.5, labels=[f'Topic {i+1}' for i in range(num_topics)], fontsize=10)
    plt.show()

   # Δημιουργία γραφήματος τύπου stacked bar chart
    plt.figure(figsize=(6, 6))  # Μέγεθος του γραφήματος

    # Κάθε μπάρα αντιπροσωπεύει ένα έγγραφο και οι διαφορετικές αποχρώσεις τα θέματα
    for i in range(num_topics):
        plt.bar(
            range(document_topic_matrix.shape[0]),
            document_topic_matrix[:, i],
            bottom=np.sum(document_topic_matrix[:, :i], axis=1),
            label=f'Θέμα {i + 1}'
        )

    plt.title('Stacked Bar Chart των Πιθανοτήτων Θεμάτων-Κειμένων', fontsize=16)
    plt.xlabel('Έγγραφα', fontsize=14)
    plt.ylabel('Πιθανότητα', fontsize=14)
    plt.legend(title='Θέματα', fontsize=10)
    plt.show()

