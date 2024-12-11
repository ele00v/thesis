"""--------------------------------------------------------------
    Πανεπιστήμιο Δυτικής Αττικής 
    Σχολή Μηχανικών 
    Τμήμα Μηχανικών Πληροφορικής και Υπολογιστών 
    Ονοματεπώνυμο: Ελένη Βέρα 
    Αριθμός Μητρώου:18390152
    Διπλωματική Εργασία 
    Επιβλέπων Καθηγητής: Φοίβος Μυλωνάς, Αναπληρωτής Καθηγητής
    Τίτλος: Αυτόματη εξαγωγή θέματος και εννοιών από σώμα κειμένων
    Title: Automatic topic and concept extraction from text corpus
    Ακαδημαϊκό Έτος: 2024-2025 
-------------------------------------------------------------------"""
import re
import nltk
import string
import numpy as np
import pandas as pd
import seaborn as sns
from nltk import pos_tag
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

# nltk.download('reuters')     
# nltk.download('punkt')      
# nltk.download('stopwords')   
# nltk.download('wordnet')     
#nltk.download('averaged_perceptron_tagger')

# Αρχικοποίηση lemmatizer και stopwords
lemmatizer = WordNetLemmatizer() # WordNetLemmatizer: Μετατροπή λέξεων στην βασική τους μορφή (lemma).
stop_words = set(stopwords.words('english')) # Stopwords: Aφαίρεση κοινών λέξεων (π.χ., "and", "the") που δεν προσφέρουν νόημα (στα αγγλικά).

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
    words = word_tokenize(text)
    # Αφαίρεση των stopwords και μη αλφαβητικών λέξεων
    words = [word for word in words if word.isalpha()]
    words = [word for word in words if word not in stop_words]
    words_with_pos = pos_tag(words)  # Επιστροφή λέξεων με τα POS tags
    # Λεμματοποίηση
    lemmatized_words = [lemmatizer.lemmatize(word, pos=get_wordnet_pos(pos)) for word, pos in words_with_pos]
    # Συνένωση των tokens σε μία συμβολοσειρά
    return ' '.join(lemmatized_words)

# Συνάρτηση για τη μετατροπή των ετικετών POS του NLTK σε μορφή POS του WordNet
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  

# Συνάρτηση για τον υπολογισμό του coherence score
def calculate_coherence(topics, texts, dictionary, coherence_type='c_v'):
    coherence_model = CoherenceModel(topics=topics, texts=texts, dictionary=dictionary, coherence=coherence_type)
    coherence_score = coherence_model.get_coherence()
    return coherence_score
   
# Συνάρτηση για την εφαρμογή του LDA και εξαγωγή των θεμάτων και της αντιστοίχισης θεμάτων-κειμένων
def apply_lda(tfidf_matrix, vectorizer, num_topics=5, num_words=10):
    try:
        # Δημιουργία του LDA μοντέλου με τον καθορισμένο αριθμό θεμάτων
        lda = LatentDirichletAllocation(n_components=num_topics, learning_method='online', max_iter=50, random_state=0)
        
        lda.fit(tfidf_matrix)
        
        # Εξαγωγή των χαρακτηριστικών (λέξεων) για κάθε θέμα
        feature_names = vectorizer.get_feature_names_out()
        topics = []
        #Ευρεση των λέξεων με το μεγαλύτερο βάρος
        for topic_idx, topic in enumerate(lda.components_):
            top_features_idx = topic.argsort()[-num_words:][::-1]
            top_features = [feature_names[i] for i in top_features_idx]
            topics.append(top_features)  

        # Υπολογισμός του πίνακα θεμάτων-κειμένων
        document_topic_matrix = lda.transform(tfidf_matrix)
        # Υπολογισμός  coherence score
        coherence_score = calculate_coherence(topics, texts, dictionary, coherence_type='c_v')

        return topics, document_topic_matrix,coherence_score 
    except ValueError as e:
        print(f"Σφάλμα κατά την εκπαίδευση του LDA: {e}")
        return None, None, None
    except Exception as e:
        print(f"Άγνωστο σφάλμα κατά την εκπαίδευση του LDA: {e}")
        return None, None, None

# Συνάρτηση για την εφαρμογή του LSA και εξαγωγή των θεμάτων και της αντιστοίχισης θεμάτων-κειμένων
def apply_lsa(tfidf_matrix, vectorizer, num_topics=5, num_words=10):
    try:
        # Δημιουργία του μοντέλου LSA με χρήση TruncatedSVD
        lsa = TruncatedSVD(n_components=num_topics, n_iter=100, random_state=0)
        
        # Εφαρμογή του LSA στη μήτρα TF-IDF
        lsa.fit(tfidf_matrix)
        
        # Εξαγωγή των χαρακτηριστικών (λέξεων) για κάθε θέμα
        feature_names = vectorizer.get_feature_names_out()
        topics = []
        
        for topic_idx, topic in enumerate(lsa.components_):
            top_features_idx = topic.argsort()[-num_words:][::-1]
            top_features = [feature_names[i] for i in top_features_idx]
            topics.append(top_features)
        
        # Υπολογισμός του πίνακα θεμάτων-κειμένων
        document_topic_matrix = lsa.transform(tfidf_matrix)
        # Υπολογισμός coherence score
        coherence_score = calculate_coherence(topics, texts, dictionary, coherence_type='c_v')
        return topics, document_topic_matrix,coherence_score 
    except ValueError as e:
        print(f"Σφάλμα κατά την εκπαίδευση του LSA: {e}")
        return None, None, None
    except Exception as e:
        print(f"Άγνωστο σφάλμα κατά την εκπαίδευση του LSA: {e}")
        return None, None, None

 # Συνάρτηση για επιλογή μεθόδου ανάλυσης θεμάτων
def choose_topic_modeling_method(tfidf_matrix, vectorizer, num_topics=5, num_words=10):
     while True:
        try:
            print("\nChoose a Topic Modeling Method:")
            print("1. Latent Dirichlet Allocation (LDA)")
            print("2. Latent Semantic Analysis (LSA)")
            choice = input("Enter 1 for LDA or 2 for LSA: ").strip()

            if choice == '1':
                print("\nApplying LDA...")
                topics, document_topic_matrix, coherence_score = apply_lda(tfidf_matrix, vectorizer, num_topics, num_words)
                if topics and document_topic_matrix is not None:
                    topic_words = [', '.join(topic) for topic in topics]  # Κατασκευή λίστας με τις λέξεις των θεμάτων
                    return topic_words, topics, document_topic_matrix, coherence_score
                else:
                    print("No topics or topic matrix found. Please check your input data.")
                    continue
            elif choice == '2':
                print("\nApplying LSA...")
                topics, document_topic_matrix, coherence_score = apply_lsa(tfidf_matrix, vectorizer, num_topics, num_words)
                if topics and document_topic_matrix is not None:
                    topic_words = [', '.join(topic) for topic in topics]  # Κατασκευή λίστας με τις λέξεις των θεμάτων
                    return topic_words, topics, document_topic_matrix, coherence_score
                else:
                    print("No topics or topic matrix found. Please check your input data.")
                    continue
            else:
                print("Invalid choice. Please enter 1 or 2.")
        except ValueError as e:
            print(f"Σφάλμα κατά την επιλογή μεθόδου: {e}")
        except Exception as e:
            print(f"Άγνωστο σφάλμα κατά την επιλογή μεθόδου: {e}")

# Συνάρτηση για τη δημιουργία και εμφάνιση των γραφημάτων
def visualizations(document_topic_matrix, num_topics):
    # 1. Βαριόμετρα για την κατανομή των εγγράφων στα θέματα
    dominant_topics = document_topic_matrix.argmax(axis=1)  # Βρίσκουμε το κυρίαρχο θέμα για κάθε έγγραφο
    topic_counts = np.bincount(dominant_topics, minlength=num_topics)  # Μετράμε τον αριθμό των εγγράφων ανά θέμα
    
    plt.figure(figsize=(6, 6))
    plt.bar(range(1, num_topics + 1), topic_counts, color='skyblue', edgecolor='black')
    plt.xlabel('Topic', fontsize=14)
    plt.ylabel('Number of Documents', fontsize=14)
    plt.title('Document Distribution Across Topics', fontsize=16)
    plt.xticks(range(1, num_topics + 1))  
    plt.show()

    # 2. Heatmap για τις πιθανότητες θεμάτων-κειμένων
    plt.figure(figsize=(6, 6)) 
    sns.heatmap(document_topic_matrix, annot=False, cmap='coolwarm', cbar=True)
    plt.title('Heatmap των Πιθανοτήτων Θεμάτων-Κειμένων', fontsize=16)
    plt.xlabel('Θέματα', fontsize=14)
    plt.ylabel('Έγγραφα', fontsize=14)
    plt.xticks(ticks=np.arange(num_topics) + 0.5, labels=[f'Topic {i+1}' for i in range(num_topics)], fontsize=10)
    plt.show()

    # 3. Stacked bar chart
    plt.figure(figsize=(6, 6))  
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

# Κύριο πρόγραμμα
if __name__ == "__main__":
    # Λίστα με τα διαθέσιμα έγγραφα του Reuters corpus
    documents = reuters.fileids()

    # Διαχωρισμός των εγγράφων σε εκπαιδευτικά (training) και δοκιμαστικά (test)
    train_docs = [doc for doc in documents if doc.startswith('training/')]
    test_docs = [doc for doc in documents if doc.startswith('test/')]
    
    n = 500 

    # Δημιουργούμε ένα DataFrame για να αποθηκεύσουμε τα δεδομένα από το Reuters corpus
    df_reuters = pd.DataFrame({
        'ID': train_docs[:n],
        'Text': [reuters.raw(doc_id) for doc_id in train_docs[:n]]
    })

    # Εφαρμόζουμε τη συνάρτηση preprocess_text στη στήλη με τα κείμενα
    df_reuters['Processed_Text'] = df_reuters['Text'].apply(preprocess_text)

    # Μετατρέπουμε τα κείμενα σε λίστες λέξεων (tokens)
    tokenized_texts = [doc.split() for doc in df_reuters['Processed_Text']]

    phrases = Phrases(tokenized_texts, min_count=5, threshold=10)
    bigram = Phraser(phrases)
    tokenized_texts = [bigram[doc] for doc in tokenized_texts]

    df_reuters['tokenized_texts'] = [' '.join(doc) for doc in tokenized_texts]

    try:
        # Δημιουργία της μήτρας TF-IDF
        vectorizer = TfidfVectorizer(stop_words='english',max_df=0.9, min_df=0.01)
        tfidf_matrix = vectorizer.fit_transform(df_reuters['tokenized_texts'])

        # Μετατροπή της μήτρας σε DataFrame για καλύτερη οπτικοποίηση
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
    except ValueError as e:
        print(f"Σφάλμα κατά τη δημιουργία της μήτρας TF-IDF: {e}")
    except Exception as e:
        print(f"Άγνωστο σφάλμα κατά τη δημιουργία της μήτρας TF-IDF: {e}")

    # Προσθέτουμε τη στήλη 'ID'
    tfidf_df.insert(0, 'ID', df_reuters['ID'])

    # Εκτύπωση  TF-IDF
    #print("TF-IDF Matrix:")
    #print(tfidf_df)

    # Αποθήκευση TF-IDF σε αρχείο κειμένου
    #with open('tfidf_matrix.txt', 'w', encoding='utf-8') as f:
    #f.write(tfidf_df.to_string(index=False))

    num_topics = 5
    num_words = 10
    texts = [doc.split() for doc in df_reuters['Processed_Text']] 
    dictionary = corpora.Dictionary(texts) 

    topic_words, topics, document_topic_matrix, coherence_score = choose_topic_modeling_method(tfidf_matrix, vectorizer, num_topics, num_words)

    print("\nTopics:")
    for topic in topics:
        print(topic)
    print(f"\nCoherence Score: {coherence_score:.2f}")
    visualizations(document_topic_matrix, num_topics)