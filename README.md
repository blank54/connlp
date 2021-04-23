# connlp
A bunch of python codes to analyze text data in the construction industry.  
Mainly reconstitute the pre-exist python libraries for Natural Language Processing (NLP).

## _Project Information_
- Supported by C!LAB (@Seoul Nat'l Univ.)

## _Contributors_
- Seonghyeon Boris Moon (blank54@snu.ac.kr, https://github.com/blank54/)
- Gitaek Lee (lgt0427@snu.ac.kr)
- Taeyeon Chang (jgwoon1838@snu.ac.kr, _a.k.a. Kowoon Chang_)
- Sehwan Chung (hwani751@snu.ac.kr)

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

# Embedding

## _Vectorizer_

_**Vectorizer**_ includes several text embedding methods that have been commonly used for decades.  

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