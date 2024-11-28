This repository provides Arabic word embedding models designed to enhance Arabic natural language processing (NLP) tasks. 
The models achieved exceptional results, including 90% accuracy on analogy tests, 99% accuracy on sentiment analysis tasks, and a similarity score of 8.1/10. 
These embeddings are ideal for researchers and developers working with Arabic NLP.


We kindly request that users of these models cite our paper.

@Article{app142311104,
AUTHOR = {Allahim, Azzah and Cherif, Asma},
TITLE = {Advancing Arabic Word Embeddings: A Multi-Corpora Approach with Optimized Hyperparameters and Custom Evaluation},
JOURNAL = {Applied Sciences},
VOLUME = {14},
YEAR = {2024},
NUMBER = {23},
ARTICLE-NUMBER = {11104},
URL = {https://www.mdpi.com/2076-3417/14/23/11104},
ISSN = {2076-3417},
ABSTRACT = {The expanding Arabic user base presents a unique opportunity for researchers to tap into vast online Arabic resources. However, the lack of reliable Arabic word embedding models and the limited availability of Arabic corpora poses significant challenges. This paper addresses these gaps by developing and evaluating Arabic word embedding models trained on diverse Arabic corpora, investigating how varying hyperparameter values impact model performance across different NLP tasks. To train our models, we collected data from three distinct sources: Wikipedia, newspapers, and 32 Arabic books, each selected to capture specific linguistic and contextual features of Arabic. By using advanced techniques such as Word2Vec and FastText, we experimented with different hyperparameter configurations, such as vector size, window size, and training algorithms (CBOW and skip-gram), to analyze their impact on model quality. Our models were evaluated using a range of NLP tasks, including sentiment analysis, similarity tests, and an adapted analogy test designed specifically for Arabic. The findings revealed that both the corpus size and hyperparameter settings had notable effects on performance. For instance, in the analogy test, a larger vocabulary size significantly improved outcomes, with the FastText skip-gram models excelling in accurately solving analogy questions. For sentiment analysis, vocabulary size was critical, while in similarity scoring, the FastText models achieved the highest scores, particularly with smaller window and vector sizes. Overall, our models demonstrated strong performance, achieving 99% and 90% accuracies in sentiment analysis and the analogy test, respectively, along with a similarity score of 8 out of 10. These results underscore the value of our models as a robust tool for Arabic NLP research, addressing a pressing need for high-quality Arabic word embeddings.},
DOI = {10.3390/app142311104}
}




For those interested in accessing the corpus used to train these embeddings, please contact us via email.
