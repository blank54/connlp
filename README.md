# connlp
A bunch of python codes to analyze text data in the construction industry.  
Mainly reconstitute the pre-exist python libraries for Natural Language Processing (NLP).

## _Project Information_
- Supported by C!LAB (@Seoul Nat'l Univ.)

## _Contributors_
- Seonghyeon Boris Moon (blank54@snu.ac.kr, https://github.com/blank54/)
- Sehwan Chung (hwani751@snu.ac.kr)
- Jungyeon Kim (janykjy@snu.ac.kr)

# Initialize

## _Setup_

```shell
pip install connlp
```

## _Test_

If the code below runs with no error, _**connlp**_ is installed successfully.

```python
from connlp.test import hello
hello()

# 'Helloworld'
```

# Preprocess

Preprocessing module supports English and Korean.  
NOTE: No plan for other languages (by 2021.04.02.).

## _Normalizer_

_**Normalizer**_ normalizes the input text by eliminating trash characters and remaining numbers, alphabets, and punctuation marks.

```python
from connlp.preprocess import Normalizer
normalizer = Normalizer()

normalizer.normalize(text='I am a boy!')

# 'i am a boy'
```

## _EnglishTokenizer_

_**EnglishTokenizer**_ tokenizes the input text in English based on word spacing.  
The ngram-based tokenization is in preparation.

```python
from connlp.preprocess import EnglishTokenizer
tokenizer = EnglishTokenizer()

tokenizer.tokenize(text='I am a boy!')

# ['I', 'am', 'a', 'boy!']
```

## _KoreanTokenizer_

_**KoreanTokenizer**_ tokenizes the input text in Korean, and is based on _**soynlp**_ (https://github.com/lovit/soynlp), an unsupervised text analyzer in Korean.

### _train_

A _**KoreanTokenizer**_ object first needs to be trained on (unlabeled) corpus. 'Word score' is calculated for every subword in the corpus.

```python
from connlp.preprocess import KoreanTokenizer
tokenizer = KoreanTokenizer(min_frequency=0) # see 'soynlp' for detailed explanation on keyword arguments

docs = ['코퍼스의 첫 번째 문서입니다.', '두 번째 문서입니다.', '마지막 문서']

tokenizer.train(text=docs)
print(tokenizer.word_score)

# {'서': 0.0, '코': 0.0, '째': 0.0, '.': 0.0, '의': 0.0, '마': 0.0, '막': 0.0, '번': 0.0, '문': 0.0, '코퍼': 1.0, '번째': 1.0, '마지': 1.0, '문서': 1.0, '코퍼스': 1.0, '문서입': 0.816496580927726, '마지막': 1.0, '코퍼스의': 1.0, '문서입니': 0.8735804647362989, '문서입니다': 0.9036020036098448, '문서입니다.': 0.9221079114817278}
```

### _tokenize_

Tokenization is based on the 'word score' calculated from _**KoreanTokenizer.train**_ method. 
For each blank-separated token, a subword that has the maximum 'word score' is selectd as an individual 'word' and separated with the remaining part.

```python
doc = docs[0] # '코퍼스의 첫 번째 문서입니다.'
tokenizer.tokenize(doc)

# ['코퍼스의', '첫', '번째', '문서', '입니다.']
```

## _StopwordRemover_

_**StopwordRemover**_ removes stopwords from a given sentence based on the user-customized stopword list.
Before utilizing _**StopwordRemover**_ the user should normalize and tokenize the docs.

```python
from connlp.preprocess import Normalizer, EnglishTokenizer, StopwordRemover
normalizer = Normalizer()
eng_tokenizer = EnglishTokenizer()
stopword_remover = StopwordRemover()

docs = ['I am a boy!', 'He is a boy..', 'She is a girl?']
tokenized_docs = []

for doc in eng_docs:
    normalized_doc = normalizer.normalize(text=doc)
    tokenized_doc = eng_tokenizer.tokenize(text=normalized_doc)
    tokenized_docs.append(tokenized_doc)

print(docs)
print(tokenized_docs)

# ['I am a boy!', 'He is a boy..', 'She is a girl?']
# [['i', 'am', 'a', 'boy'], ['he', 'is', 'a', 'boy'], ['she', 'is', 'a', 'girl']]
```

The user should prepare a customized stopword list (i.e., _stoplist_).
The _stoplist_ should include user-customized stopwords divided by '\n' and the file should be in ".txt" format.

```text
a
is
am
```

Initiate the _**StopwordRemover**_ with appropriate filepath of user-customized stopword list.
If the stoplist is absent at the filepath, the stoplist would be ramain as a blank list.

```python
fpath_stoplist = 'test/thesaurus/stoplist.txt'
stopword_remover.initiate(fpath_stoplist=fpath_stoplist)

print(stopword_remover)

# <connlp.preprocess.StopwordRemover object at 0x7f163e70c050>
```

The user can count the word frequencies and figure out additional stopwords based on the results.

```python
stopword_remover.count_freq_words(docs=tokenized_docs)

# ========================================
# Word counts
#   | [1] a: 3
#   | [2] boy: 2
#   | [3] is: 2
#   | [4] i: 1
#   | [5] am: 1
#   | [6] he: 1
#   | [7] she: 1
#   | [8] girl: 1
```

After finally updating the _stoplist_, use _**remove**_ method to remove the stopwords from text.

```python
stopword_removed_docs = []
    for doc in tokenized_docs:
        stopword_removed_docs.append(stopword_remover.remove(sent=doc))

print(stopword_removed_docs)

# [['i', 'boy'], ['he', 'boy'], ['she', 'girl']]
```

The user can check which stopword was removed with _**check_removed_words**_ methods.

```python
stopword_remover.check_removed_words(docs=tokenized_docs, stopword_removed_docs=stopword_removed_docs)

# ========================================
# Check stopwords removed
#   | [1] BEFORE: a(3) ->
#   | [2] BEFORE: boy -> AFTER: boy(2)
#   | [3] BEFORE: is(2) ->
#   | [4] BEFORE: i -> AFTER: i(1)
#   | [5] BEFORE: am(1) ->
#   | [6] BEFORE: he -> AFTER: he(1)
#   | [7] BEFORE: she -> AFTER: she(1)
#   | [8] BEFORE: girl -> AFTER: girl(1)
```

# Embedding

## _Vectorizer_

_**Vectorizer**_ includes several text embedding methods that have been commonly used for decades.  

### _tfidf_

TF-IDF is the most commonly used technique for word embedding.  
The TF-IDF model counts the term frequency(TF) and inverse document frequency(IDF) from the given documents.  
The results included the followings.  
- TF-IDF Vectorizer (a class of sklearn.feature_extraction.text.TfidfVectorizer')
- TF-IDF Matrix
- TF-IDF Vocabulary

```python
from connlp.preprocess import EnglishTokenizer
from connlp.embedding import Vectorizer
tokenizer = EnglishTokenizer()
vectorizer = Vectorizer()

docs = ['I am a boy', 'He is a boy', 'She is a girl']
tfidf_vectorizer, tfidf_matrix, tfidf_vocab = vectorizer.tfidf(docs=docs)
type(tfidf_vectorizer)

# <class 'sklearn.feature_extraction.text.TfidfVectorizer'>
```

The user can get a document vector by indexing the _**tfidf_matrix**_.

```python
tfidf_matrix[0]

# (0, 2)    0.444514311537431
# (0, 0)    0.34520501686496574
# (0, 1)    0.5844829010200651
# (0, 5)    0.5844829010200651
```

The _**tfidf_vocab**_ returns an index for every token.

```python
print(tfidf_vocab)

# {'i': 5, 'am': 1, 'a': 0, 'boy': 2, 'he': 4, 'is': 6, 'she': 7, 'girl': 3}
```

### _word2vec_

Word2Vec is a distributed representation language model for word embedding.  
The Word2vec model trains tokenized docs and returns word vectors.  
The result is a class of 'gensim.models.word2vec.Word2Vec'.

```python
from connlp.preprocess import EnglishTokenizer
from connlp.embedding import Vectorizer
tokenizer = EnglishTokenizer()
vectorizer = Vectorizer()

docs = ['I am a boy', 'He is a boy', 'She is a girl']
tokenized_docs = [tokenizer.tokenize(text=doc) for doc in docs]
w2v_model = vectorizer.word2vec(docs=tokenized_docs)
type(w2v_model)

# <class 'gensim.models.word2vec.Word2Vec'>
```

The user can get a word vector by _**.wv**_ method.

```python
w2v_model.wv['boy']

# [-2.0130998e-03 -3.5652996e-03  2.7793974e-03 ...]
```

The Word2Vec model provides the _topn_-most similar word vectors.

```python
w2v_model.wv.most_similar('boy', topn=3)

# [('He', 0.05311150848865509), ('a', 0.04154288396239281), ('She', -0.029122961685061455)]
```

### _word2vec (update)_

The user can update the Word2Vec model with new data.

```python
new_docs = ['Tom is a man', 'Sally is not a boy']
tokenized_new_docs = [tokenizer.tokenize(text=doc) for doc in new_docs]
w2v_model_updated = vectorizer.word2vec_update(w2v_model=w2v_model, new_docs=tokenized_new_docs)

w2v_model_updated.wv['man']

# [4.9649975e-03  3.8002312e-04 -1.5773597e-03 ...]
```

### _doc2vec_

Doc2Vec is a distributed representation language model for longer text (e.g., sentence, paragraph, document) embedding.  
The Doc2vec model trains tokenized docs with tags and returns document vectors.  
The result is a class of 'gensim.models.doc2vec.Doc2Vec'.

```python
from connlp.preprocess import EnglishTokenizer
from connlp.embedding import Vectorizer
tokenizer = EnglishTokenizer()
vectorizer = Vectorizer()

docs = ['I am a boy', 'He is a boy', 'She is a girl']
tagged_docs = [(idx, tokenizer.tokenize(text=doc)) for idx, doc in enumerate(docs)]
d2v_model = vectorizer.doc2vec(tagged_docs=tagged_docs)
type(d2v_model)

# <class 'gensim.models.doc2vec.Doc2Vec'>
```

The Doc2Vec model can infer a new document.

```python
test_doc = ['My', 'name', 'is', 'Peter']
d2v_model.infer_vector(doc_words=test_doc)

# [4.8494316e-03 -4.3647490e-03  1.1437446e-03 ...]
```

# Analysis

## _TopicModel_

_**TopicModel**_ is a class for topic modeling based on gensim LDA model.  
It provides a simple way to train lda model and assign topics to docs.  

_**TopicModel**_ requires two instances.  
- a dict of docs whose keys are the tag
- the number of topics for modeling

```python
from connlp.analysis import TopicModel

num_topics = 2
docs = {'doc1': ['I', 'am', 'a', 'boy'],
        'doc2': ['He', 'is', 'a', 'boy'],
        'doc3': ['Cat', 'on', 'the', 'table'],
        'doc4': ['Mike', 'is', 'a', 'boy'],
        'doc5': ['Dog', 'on', 'the', 'table'],
        }

lda_model = TopicModel(docs=docs, num_topics=num_topics)
```

### _learn_

The user can train the model with _learn_ method.
Unless parameters being provided by the user, the model trains based on default parameters.  

After _learn_, _**TopicModel**_ provides _model_ instance that is a class of <'gensim.models.ldamodel.LdaModel'>


```python
parameters = {
    'iterations': 100,
    'alpha': 0.7,
    'eta': 0.05,
}
lda_model.learn(parameters=parameters)
type(lda_model.model)

# <class 'gensim.models.ldamodel.LdaModel'>
```

### _coherence_

_**TopicModel**_ provides coherence value for model evaluation.  
The coherence value is automatically calculated right after model training.

```python
print(lda_model.coherence)

# 0.3607990279229385
```

### _assign_

The user can easily assign the most proper topic to each doc using _assign_ method.  
After _assign_, the _**TopicModel**_ provides _tag2topic_ and _topic2tag_ instances for convenience.

```python
lda_model.assign()

print(lda_model.tag2topic)
print(lda_model.topic2tag)

# defaultdict(<class 'int'>, {'doc1': 1, 'doc2': 1, 'doc3': 0, 'doc4': 1, 'doc5': 0})
# defaultdict(<class 'list'>, {1: ['doc1', 'doc2', 'doc4'], 0: ['doc3', 'doc5']})
```

## _NamedEntityRecognition_

Before using NER modules, the user should install proper versions of TensorFlow and Keras.  

```shell
pip install config==0.4.2 gensim==3.8.1 gpustat==0.6.0 GPUtil==1.4.0 h5py==2.10.0 JPype1==0.7.1 Keras==2.2.4 konlpy==0.5.2 nltk==3.4.5 numpy==1.18.1 pandas==1.0.1 scikit-learn==0.22.1 scipy==1.4.1 silence-tensorflow==1.1.1 soynlp==0.0.493 tensorflow==1.14.0 tensorflow-gpu==1.14.0
```

The modules might require the module of _keras-contrib_.  
The user can install the module by following the below.  

```shell
git clone https://www.github.com/keras-team/keras-contrib.git 
cd keras-contrib 
python setup.py install
```

### _Labels_

_**NER_Model**_ is a class to conduct named entity recognition using Bi-directional Long-Short Term Memory (Bi-LSTM) and Conditional Random Field (CRF).  

At the beginning, appropriate labels are required.  
The labels should be numbered with start of 0.

```python
from connlp.analysis import NER_Labels

label_dict = {'NON': 0,     #None
              'PER': 1,     #PERSON
              'FOD': 2,}    #FOOD

ner_labels = NER_Labels(label_dict=label_dict)
```

### _Corpus_

Next, the user should prepare data including sentences and labels, of which each data being matched by the same tag.  
The tokenized sentences and labels are then combined via _**NER_LabeledSentence**_.  
With the data, labels, and a proper size of _max_sent_len_ (i.e., the maximum length of sentence for analysis), _**NER_Corpus**_ would be developed.  
Once the corpus was developed, every data of sentences and labels would be padded with the length of _max_sent_len_.  

```python
from connlp.preprocess import EnglishTokenizer
from connlp.analysis import NER_LabeledSentence, NER_Corpus
tokenizer = EnglishTokenizer()

data_sents = {'sent1': 'Sam likes pizza',
              'sent2': 'Erik eats pizza',
              'sent3': 'Erik and Sam are drinking soda',
              'sent4': 'Flora cooks chicken',
              'sent5': 'Sam ordered a chicken',
              'sent6': 'Flora likes chicken sandwitch',
              'sent7': 'Erik likes to drink soda'}
data_labels = {'sent1': [1, 0, 2],
               'sent2': [1, 0, 2],
               'sent3': [1, 0, 1, 0, 0, 2],
               'sent4': [1, 0, 2],
               'sent5': [1, 0, 0, 2],
               'sent6': [1, 0, 2, 2],
               'sent7': [1, 0, 0, 0, 2]}

docs = []
for tag, sent in data_sents.items():
    words = [str(w) for w in tokenizer.tokenize(text=sent)]
    labels = data_labels[tag]
    docs.append(NER_LabeledSentence(tag=tag, words=words, labels=labels))

max_sent_len = 10
ner_corpus = NER_Corpus(docs=docs, ner_labels=ner_labels, max_sent_len=max_sent_len)
type(ner_corpus)

# <class 'connlp.analysis.NER_Corpus'>
```

### _Word Embedding_

Every word in the _**NER_Corpus**_ should be embedded into numeric vector space.  
The user can conduct embedding with Word2Vec which is provided in _**Vectorizer**_ of _**connlp**_.  
Note that the embedding process of _**NER_Corpus**_ only requires the dictionary of word vectors and the feature size.  

```python
from connlp.preprocess import EnglishTokenizer
from connlp.embedding import Vectorizer
tokenizer = EnglishTokenizer()
vectorizer = Vectorizer()

tokenized_sents = [tokenizer.tokenize(sent) for sent in data_sents.values()]
w2v_model = vectorizer.word2vec(docs=tokenized_sents)

word2vector = vectorizer.get_word_vectors(w2v_model)
feature_size = w2v_model.vector_size
ner_corpus.word_embedding(word2vector=word2vector, feature_size=feature_size)
print(ner_corpus.X_embedded)

# [[[-2.40120804e-03  1.74632657e-03  ...]
#   [-3.57543468e-03  2.86567654e-03  ...]
#   ...
#   [ 0.00000000e+00  0.00000000e+00  ...]] ...]
```

### _Model Initialization_

The parameters for Bi-LSTM and model training should be provided, however, they can be composed of a single dictionary.  
The user should initialize the _**NER_Model**_ with _**NER_Corpus**_ and the parameters.

```python
from connlp.analysis import NER_Model

parameters = {
    # Parameters for Bi-LSTM.
    'lstm_units': 512,
    'lstm_return_sequences': True,
    'lstm_recurrent_dropout': 0.2,
    'dense_units': 100,
    'dense_activation': 'relu',

    # Parameters for model training.
    'test_size': 0.3,
    'batch_size': 1,
    'epochs': 100,
    'validation_split': 0.1,
}

ner_model = NER_Model()
ner_model.initialize(ner_corpus=ner_corpus, parameters=parameters)
type(ner_model)

# <class 'connlp.analysis.NER_Model'>
```

### _Model Training_

The user can train the _**NER_Model**_ with customized parameters.  
The model automatically gets the dataset from the _**NER_Corpus**_.  

```python
ner_model.train(parameters=parameters)

# Train on 3 samples, validate on 1 samples
# Epoch 1/100
# 3/3 [==============================] - 3s 1s/step - loss: 1.4545 - crf_viterbi_accuracy: 0.3000 - val_loss: 1.0767 - val_crf_viterbi_accuracy: 0.8000
# Epoch 2/100
# 3/3 [==============================] - 0s 74ms/step - loss: 0.8602 - crf_viterbi_accuracy: 0.7000 - val_loss: 0.5287 - val_crf_viterbi_accuracy: 0.8000
# ...
```

### _Model Evaluation_

The model performance can be shown in the aspects of confusion matrix and F1 score.

```python
ner_model.evaluate()

# |--------------------------------------------------
# |Confusion Matrix:
# [[ 3  0  3  6]
#  [ 1  3  0  4]
#  [ 0  0  2  2]
#  [ 4  3  5 12]]
# |--------------------------------------------------
# |F1 Score: 0.757
# |--------------------------------------------------
# |    [NON]: 0.600
# |    [PER]: 0.857
# |    [FOD]: 0.571
```

### _Save_

The user can save the _**NER_Model**_.  
The model would save the model itself ("\<FileName\>.pk") and the dataset ("\<FileName\>-dataset.pk") that was used in model development.  
Note that the directory should exist before saving the model.  

```python
from connlp.util import makedir

fpath_model = 'test/ner/model.pk'
makedir(fpath=fpath_model)
ner_model.save(fpath_model=fpath_model)
```

### _Load_

If the user wants to load the already trained model, just call the model and load.  

```python
fpath_model = 'test/ner/model.pk'
ner_model = NER_Model()
ner_model.load(fpath_model=fpath_model, ner_corpus=ner_corpus, parameters=parameters)
```

### _Prediction_

_**NER_Model**_ can conduct a new NER task on the given sentence.  
The result is a class of _**NER_Result**_.  

```python
from connlp.preprocess import EnglishTokenizer
vectorizer = Vectorizer()

new_sent = 'Tom eats apple'
tokenized_sent = tokenizer.tokenize(new_sent)
ner_result = ner_model.predict(sent=tokenized_sent)
print(ner_result)

# Tom/PER eats/NON apple/FOD
```

# Visualization

## _Visualizer_

_**Visualizer**_ includes several simple tools for text visualization.

### _network_

_**network**_ method provides a word network for tokenized docs.

```python
from connlp.preprocess import EnglishTokenizer
from connlp.visualize import Visualizer
tokenizer = EnglishTokenizer()
visualizer = Visualizer()

docs = ['I am a boy', 'She is a girl']
tokenized_docs = [tokenizer.tokenize(text=doc) for doc in docs]
visualizer.network(docs=tokenized_docs, show=True)
```

# Extracting Text

## _TextConverter_

_**TextConverter**_ includes several methods that extract raw text from various types of files (e.g. PDF, HWP) and/or converts the files into plain text files (e.g. TXT).

### _hwp2txt_

_**hwp2txt**_ method converts a HWP file into a plain text file.
Dependencies: pyhwp package

Install pyhwp (you need to install the pre-release version)

```
pip install --pre pyhwp
```

Example

```python
from connlp.text_extract import TextConverter
converter = TextConverter()

hwp_fpath = '/data/raw/hwp_file.hwp'
output_fpath = '/data/processed/extracted_text.txt'

converter.hwp2txt(hwp_fpath, output_fpath) # returns 0 if no error occurs
```

# GPU Utils

## _GPUMonitor_

_**GPUMonitor**_ generates a class to monitor and display the GPU status based on nvidia-smi.  
Refer to "https://github.com/anderskm/gputil" and "https://data-newbie.tistory.com/561" for usages.

Install _GPUtils_ module with _pip_.

```
pip install GPUtils
```

Write your code between the initiation of the _**GPUMonitor**_ and _**monitor.stop()**_.

```python
from connlp.util import GPUMonitor

monitor = GPUMonitor(delay=3)
# >>>Write your code here<<<
monitor.stop()

# | ID | GPU | MEM |
# ------------------
# |  0 |  0% |  0% |
# |  1 |  1% |  0% |
# |  2 |  0% | 94% |
```
