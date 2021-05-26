# Topic-Modeling-Based-Extractive-Text-Summarization

## Introduction

- Text summarization is an approach for identifying important information present within text documents. This computational technique aims to generate shorter versions of the source text, by including only the relevant and salient information present within the source text. 
- Topic Modeling Based Extractive Text Summarization is a novel method to summarize text documents by clustering its contents based on latent topics produced using topic modeling techniques and by generating extractive summaries for each of the identified text clusters. All extractive sub-summaries are later combined to generate a summary for any given source document. 
- The lesser used and challenging WikiHow dataset is used in this approach to text summarization. This dataset is unlike the commonly used news datasets which are available for text summarization. The well-known news datasets present their most important information in the first few lines of their source texts, which make their summarization a lesser challenging task when compared to summarizing the WikiHow dataset. Contrary to these news datasets, the documents in the WikiHow dataset are written using a generalized approach and have lesser abstractedness and higher compression ratio, thus proposing a greater challenge to generate summaries. 
- Most current state-of-the-art text summarization techniques tend to eliminate important information present in source documents in the favor of brevity. The proposed technique aims to capture all the varied information present in source documents. Although the dataset proved challenging, after performing extensive tests within the experimental setup, it is discovered that the model produces encouraging ROUGE results and summaries when compared to the other published text summarization models.

## Publication and Citation

- This work is published in the International Journal of Innovative Technology and Exploring Engineering (IJITEE).
- Our paper can be found [here](https://www.ijitee.org/wp-content/uploads/papers/v9i6/F4611049620.pdf).
- You can cite the work as follows:

```
@article{kalliath2020Topic,
 title = {Topic Modeling Based Extractive Text Summarization},
 author = {Issam, Kalliath Abdul Rasheed and Patel, Shivam and C. N., Subalalitha},
 journal = {International Journal of Innovative Technology and Exploring Engineering (IJITEE)},
 year = {2020},
 pages = {1710--1719},
 volume = {9},
 number = {6}
}
```

## Dependencies

1. [pyLDAvis](https://pypi.org/project/pyLDAvis/)

   ```pip install pyLDAvis```

2. [py-rouge](https://pypi.org/project/py-rouge/)

   ```pip install py-rouge```
   
3. [summa](https://pypi.org/project/summa/)

   ```pip install summa```
   
4. [nltk](https://pypi.org/project/nltk/)

   ```pip install nltk```


## Contact

shivam.patel1606@gmail.com
