"""--------------------------------------------------------------
    Πανεπιστήμιο Δυτικής Αττικής 
    Σχολή Μηχανικών 
    Τμήμα Μηχανικών Πληροφορικής και Υπολογιστών 
    Ονοματεπώνυμο: Ελένη Βέρα 
    Αριθμός Μητρώου:18390152
    Επιβλέπων Καθηγητής: Φοίβος Μυλωνάς
    Διπλωματική Εργασία 
    Τίτλος: Αυτόματη εξαγωγή θέματος και εννοιών από σώμα κειμένων
    Title: Automatic topic and concept extraction from text corpus
-------------------------------------------------------------------"""
import re
import nltk
import string
import pandas as pd
from nltk.corpus import reuters 
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Κατεβάστε τα αναγκαία corpora και λεξικά για το NLTK
# nltk.download('reuters')
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

# Αρχικοποίηση του lemmatizer
lemmatizer = WordNetLemmatizer()
# Αρχικοποίηση του stemmer
stemmer = PorterStemmer()
# Αρχικοποίηση των stopwords
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
n = 56  # Μπορούμε να αλλάξουμε αυτή την τιμή

# Δημιουργούμε ένα DataFrame για να αποθηκεύσουμε τα δεδομένα από το Reuters corpus
df_reuters = pd.DataFrame({
    'ID': test_docs[:n],
    'Text': [reuters.raw(doc_id) for doc_id in train_docs[:n]]
})

# Εφαρμόζουμε τη συνάρτηση preprocess_text στη στήλη με τα κείμενα
df_reuters['Processed_Text'] = df_reuters['Text'].apply(preprocess_text)

# Δημιουργούμε τη μήτρα "Bag of Words" χρησιμοποιώντας τον CountVectorizer
vectorizer = CountVectorizer(stop_words='english')
bow_matrix = vectorizer.fit_transform(df_reuters['Processed_Text'])

# Μετατρέπουμε τη μήτρα σε DataFrame για καλύτερη οπτικοποίηση
bow_df = pd.DataFrame(bow_matrix.toarray(), columns=vectorizer.get_feature_names_out())

# Προσθέτουμε τη στήλη 'ID'
bow_df.insert(0, 'ID', df_reuters['ID'])

# Εκτύπωση της μήτρας Bag of Words
print("Bag of Words Matrix:")
print(bow_df)

# Αποθήκευση της μήτρας Bag of Words σε αρχείο κειμένου
with open('bow_matrix.txt', 'w', encoding='utf-8') as f:
    f.write(bow_df.to_string(index=False))


# Συνάρτηση για την εφαρμογή του LDA και εξαγωγή των θεμάτων
def apply_lda(bow_matrix, num_topics=5, num_words=10):
    # Δημιουργία του LDA μοντέλου με τον καθορισμένο αριθμό θεμάτων
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    
    # Εκπαίδευση του μοντέλου στο Bag of Words
    lda.fit(bow_matrix)
    
    # Εξαγωγή των χαρακτηριστικών (λέξεων) για κάθε θέμα
    feature_names = bow_matrix.columns
    topics = []
    
    for topic_idx, topic in enumerate(lda.components_):
        # Τα 10 πιο σημαντικά χαρακτηριστικά για κάθε θέμα
        top_features_idx = topic.argsort()[-num_words:][::-1]
        top_features = [feature_names[i] for i in top_features_idx]
        topics.append(f"Topic {topic_idx + 1}: " + ", ".join(top_features))
    
    return topics

# Εφαρμόζουμε το LDA με 5 θέματα και 10 λέξεις για κάθε θέμα
topics = apply_lda(bow_df.drop('ID', axis=1), num_topics=5, num_words=10)

# Εκτύπωση των θεμάτων
print("\nLatent Topics:")
for topic in topics:
    print(topic)