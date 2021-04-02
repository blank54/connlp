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
NOTE: No plan exist for other languages currently (2021.04.02.).

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

tokenizer.tokenizer(text='I am a boy!')

# ['I', 'am', 'a', 'boy!']
```