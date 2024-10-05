"""----------------------------------------------------------
    Πανεπιστήμιο Δυτικής Αττικής 
    Σχολή Μηχανικών 
    Τμήμα Μηχανικών Πληροφορικής και Υπολογιστών 
    Ονοματεπώνυμο: Ελένη Βέρα 
    Αριθμός Μητρώου:18390152
    Επιβλέπως Καθηγητής: Φοίβος Μυλωνάς
    Διπλωματική Εργασία 
    Αυτόματη εξαγωγή θέματος και εννοιών από σώμα κειμένων
    Automatic topic and concept extraction from text corpus
    test commit
------------------------------------------------------------"""
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import string

# Ensure necessary NLTK resources are downloaded
# Uncomment these lines if you haven't downloaded the required packages
# nltk.download('punkt')
# nltk.download('stopwords')

# Initialize stemmer 
stemmer = PorterStemmer()

# Initialize stopwords
stop_words = set(stopwords.words('english'))

# Function for preprocessing text
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Cleaning: Remove punctuation and non-alphabetic characters
    tokens = [word for word in tokens if word.isalpha()]
    
    # Remove stopwords and apply stemming
    processed_tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    
    # Join tokens back into a single string
    return ' '.join(processed_tokens)

# Open the CSV file and load it into a DataFrame
df = pd.read_csv('text_corpus.csv', encoding='ISO-8859-1')

# Check if 'Text' column exists before applying preprocessing
if 'Text' in df.columns:
    # Apply preprocessing to the 'Text' column
    df['Processed_Text'] = df['Text'].apply(preprocess_text)
    
    # Create a Bag of Words (BoW) matrix using CountVectorizer
    vectorizer = CountVectorizer()
    bow_matrix = vectorizer.fit_transform(df['Processed_Text'])
    
    # Convert BoW matrix to DataFrame for better visualization (optional)
    bow_df = pd.DataFrame(bow_matrix.toarray(), columns=vectorizer.get_feature_names_out())
    
    # Add ID column to the BoW DataFrame
    bow_df.insert(0, 'ID', df['ID']) 
    
    # Display the BoW DataFrame with IDs
    print("Bag of Words Matrix:")
    print(bow_df)
    
    # Save the BoW matrix with IDs to a text file
    with open('bow_matrix.txt', 'w', encoding='utf-8') as f:
        f.write(bow_df.to_string(index=False)) 
else:
    print("The column 'Text' does not exist in the DataFrame.")