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

# Κατεβάστε τα αναγκαία corpora και λεξικά για το NLTK
# nltk.download('reuters')
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

# Αρχικοποίηση του lemmatizer
# Lemmatizing είναι η διαδικασία μετατροπής των λέξεων στην λεξική τους μορφή, 
# π.χ. "running" -> "run", αναγνωρίζοντας το μέρος του λόγου της λέξης
lemmatizer = WordNetLemmatizer()

# Αρχικοποίηση του stemmer
# Stemming είναι η διαδικασία αφαίρεσης των καταλήξεων για την επιστροφή της βασικής ρίζας της λέξης
# π.χ. "running" -> "run", αλλά χωρίς να λαμβάνει υπόψη το μέρος του λόγου της λέξης
stemmer = PorterStemmer()

# Αρχικοποίηση των stopwords 
# Τα stopwords είναι λέξεις που συνήθως δεν έχουν σημασία στην ανάλυση κειμένου, όπως οι σύνδεσμοι ("and", "is").
stop_words = set(stopwords.words('english'))

# Συνάρτηση για την προεπεξεργασία του κειμένου
def preprocess_text(text):
    # Μετατροπή του κειμένου σε πεζά γράμματα
    text = text.lower()

    # Διαχωρισμός του κειμένου σε λέξεις (Tokenization)
    tokens = word_tokenize(text)
    
    # Αφαίρεση των σημείων στίξης και των μη αλφαβητικών χαρακτήρων
    tokens = [word for word in tokens if word.isalpha()]

    # Εφαρμογή της διαδικασίας POS tagging (μέρος του λόγου)
    pos_tags = nltk.pos_tag(tokens)

    # Lemmatization και αφαίρεση stopwords
    # Αναγνωρίζουμε το μέρος του λόγου κάθε λέξης (POS tagging) για να κάνουμε το lemmatization πιο ακριβές
    lemmatized_tokens = []
    for word, tag in pos_tags:
        pos = get_wordnet_pos(tag)
        lemmatized_word = lemmatizer.lemmatize(word, pos=pos)
        # Προσθέτουμε τη λέξη στη λίστα εάν δεν είναι stopword και έχει μήκος μεγαλύτερο από 2 χαρακτήρες
        if lemmatized_word not in stop_words and len(lemmatized_word) > 2:
            # Εφαρμόζουμε το stemming για να μειώσουμε περαιτέρω τις λέξεις στη ριζική τους μορφή
            stemmed_word = stemmer.stem(lemmatized_word)
            lemmatized_tokens.append(stemmed_word)
    
    # Συνένωση των tokens σε μία ενιαία συμβολοσειρά για επιστροφή
    return ' '.join(lemmatized_tokens)

# Συνάρτηση για τη μετατροπή των ετικετών POS του NLTK σε μορφή POS του WordNet
def get_wordnet_pos(treebank_tag):
    # Οι ετικέτες του NLTK έχουν συγκεκριμένη σύνταξη, εδώ γίνεται η αντιστοίχιση με το WordNet
    if treebank_tag.startswith('J'):
        return wordnet.ADJ  # Επίθετο
    elif treebank_tag.startswith('V'):
        return wordnet.VERB  # Ρήμα
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN  # Ουσιαστικό
    elif treebank_tag.startswith('R'):
        return wordnet.ADV   # Επίρρημα
    else:
        return wordnet.NOUN  # Αν δεν βρεθεί αντιστοιχία, θεωρούμε πως είναι ουσιαστικό

# Λίστα με τα διαθέσιμα έγγραφα του Reuters corpus
documents = reuters.fileids()

# Διαχωρισμός των εγγράφων σε εκπαιδευτικά (training) και δοκιμαστικά (test)
train_docs = [doc for doc in documents if doc.startswith('training/')]
test_docs = [doc for doc in documents if doc.startswith('test/')]

# Επεξεργαζόμαστε τα πρώτα n εκπαιδευτικά έγγραφα για να μειώσουμε τον όγκο δεδομένων
n = 50  # Μπορούμε να αλλάξουμε αυτή την τιμή για να περιορίσουμε τον αριθμό των εγγράφων

# Δημιουργούμε ένα DataFrame για να αποθηκεύσουμε τα δεδομένα από το Reuters corpus
df_reuters = pd.DataFrame({
    'ID': train_docs[:n],  # Το ID κάθε εγγράφου
    'Text': [reuters.raw(doc_id) for doc_id in train_docs[:n]]  # Το κείμενο κάθε εγγράφου
})

# Εφαρμόζουμε τη συνάρτηση preprocess_text στη στήλη με τα κείμενα
df_reuters['Processed_Text'] = df_reuters['Text'].apply(preprocess_text)

# Δημιουργούμε τη μήτρα "Bag of Words" χρησιμοποιώντας τον CountVectorizer (αφαίρεση των stop words)
vectorizer = CountVectorizer(stop_words='english')
bow_matrix = vectorizer.fit_transform(df_reuters['Processed_Text'])

# Μετατρέπουμε τη μήτρα σε DataFrame για καλύτερη οπτικοποίηση
bow_df = pd.DataFrame(bow_matrix.toarray(), columns=vectorizer.get_feature_names_out())

# Προσθέτουμε τη στήλη 'ID' για αναφορά
bow_df.insert(0, 'ID', df_reuters['ID'])

# Εκτύπωση της μήτρας Bag of Words
print("Bag of Words Matrix:")
print(bow_df)

# Αποθήκευση της μήτρας Bag of Words σε αρχείο κειμένου (bow_matrix.txt)
with open('bow_matrix.txt', 'w', encoding='utf-8') as f:
    f.write(bow_df.to_string(index=False))