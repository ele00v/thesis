"""----------------------------------------------------------
    Πανεπιστήμιο Δυτικής Αττικής 
    Σχολή Μηχανικών 
    Τμήμα Μηχανικών Πληροφορικής και Υπολογιστών 
    Ονοματεπώνυμο: Ελένη Βέρα 
    Αριθμός Μητρώου:18390152
    Επιβλέπως Καθηγητής: Φοίβος Μυλωνάς
    Διπλωματική Εργασία 
    Τίτλος: Αυτόματη εξαγωγή θέματος και εννοιών από σώμα κειμένων
    Title: Automatic topic and concept extraction from text corpus
------------------------------------------------------------"""
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import TruncatedSVD
import string

# nltk.download('punkt')
# nltk.download('stopwords')

# Αρχικοποίηση του stemmer 
# Stemming είναι η διαδικασία δημιουργίας μορφολογικών παραλλαγών της ρίζας/βασικής λέξης
stemmer = PorterStemmer()

# Αρχικοποίηση των stopwords 
# Stopwords είναι οι λέξεις που συνήθως δεν έχουν σημασία στην ανάλυση κειμένου, όπως "ο", "και", "είναι"
stop_words = set(stopwords.words('english'))

# Συνάρτηση για την προεπεξεργασία του κειμένου
def preprocess_text(text):
    # Μετατροπή σε πεζά γράμματα
    text = text.lower()
    
    # Κατακερματισμός του κειμένου σε λέξεις (Tokenization)
    tokens = word_tokenize(text)
    
    # Καθαρισμός: Αφαίρεση σημείων στίξης και μη αλφαβητικών χαρακτήρων
    tokens = [word for word in tokens if word.isalpha()]
    
    # Αφαίρεση των stopwords και εφαρμογή stemming
    processed_tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    
    # Συνένωση των tokens σε μία ενιαία συμβολοσειρά
    return ' '.join(processed_tokens)

# Άνοιγμα του αρχείου CSV και φόρτωσή του σε ένα DataFrame
df = pd.read_csv('text_corpus.csv', encoding='ISO-8859-1')

# Έλεγχος αν υπάρχει η στήλη 'Text' πριν την εφαρμογή της προεπεξεργασίας
if 'Text' in df.columns:
    # Εφαρμογή της συνάρτησης προεπεξεργασίας στη στήλη 'Text'
    df['Processed_Text'] = df['Text'].apply(preprocess_text)
    
    # Δημιουργία του πίνακα Bag of Words χρησιμοποιώντας CountVectorizer
    #Ο CountVectorizer μετατρέπει τα κείμενα σε ενα πίνακα δύο διαστάσεων όπου κάθε στήλη αντιστοιχεί σε μία λέξη 
    #από το σύνολο των κειμένων, ενώ κάθε γραμμή αντιπροσωπεύει ένα κείμενο από το σύνολο δεδομένων
    vectorizer = CountVectorizer()
    bow_matrix = vectorizer.fit_transform(df['Processed_Text'])
    
    # Μετατροπή του πίνακα Bag of Words σε DataFrame για καλύτερη οπτικοποίηση
    bow_df = pd.DataFrame(bow_matrix.toarray(), columns=vectorizer.get_feature_names_out())
    
    # Προσθήκη της στήλης 'ID' στο DataFrame του Bag of Words
    bow_df.insert(0, 'ID', df['ID']) 
    
    # Εμφάνιση του DataFrame του Bag of Words
    print("Bag of Words Matrix:")
    print(bow_df)
    
    # Αποθήκευση του DataFrame του Bag of Words σε αρχείο κειμένου (dow_matrix.txt)
    with open('bow_matrix.txt', 'w', encoding='utf-8') as f:
        f.write(bow_df.to_string(index=False)) 
else:
    print("The column 'Text' does not exist in the DataFrame.")