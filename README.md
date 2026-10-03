# Semantic Bleaching in Subreddits

## Introduction 

This repository contains several experiments on semantic bleaching and semantic bleaching trajectories. My guiding hypothesis was that nouns with high descriptive content would bleach through neighborhood detachment rather than sense extension, as Sweetser proposed. This hypothesis was an extension of Hamilton et al.'s 2016 paper "Cultural Shift or Linguistic Drift?", which posits nouns are more likely to undergo irregular cultural shifts compared to other parts of speech, specifically adverbs. Adverbs have been the focus of much work in the field; I focused on the work of Sweetser. 

I validated all the metrics used on a time-shuffled corpus. This null was motivated by Dubossarsky et al.'s 2017 work. The `BleachingTrajectories` notebook demonstrates the value of this exercise--while the results looked very positive initially, the diffuseness metric did not pass the time-shuffled null. Additionally, using subreddits introduced significantly more noise than HistWords, particularly for rare words. 

## Work 

TNPosPredictor contains the code and analysis for this hypothesis; I measured concreteness by training a model on Brysbaert et al.'s human concreteness ratings. Drift was measured through excess, a measure of how directionally persistent a word's displacement vector was. Unfortunately, excess had a 0.538 correlation with log frequency. More information about the metric is in the notebook. The concreteness metrics are resistant to frequency confounds, something I struggled to eliminate with other distributional measures; when controlled for frequency, the correlation between the imputed rankings and excess was +0.006 (compared to -0.030). Ultimately, I found no relationship between descriptive content and semantic shift. In fact, nouns moved straighter than adverbs. 

BleachingTrajectories was my first exploration of the hypothesis. I used TN, a measure of neighborhood diffuseness, as a proxy for descriptive content, and a metric to measure a word's chaining. While the trend seemed very well-defined initially, the time-shuffled corpus produced nearly the exact same numbers, killing the diffuseness metric. 

VectorExploration documents the first experiments and exploration of the vectors I produced based on three subreddits. I used three metrics, IoU, cosine similarity, and SimLex, a measure of how similar a word is to its synonyms (from Luo et al.). From these metrics, I found the list of words that appeared to be stable and a list of words that seemed to change a lot; this list anchored metric choice later. 

GeometricSignals is the earliest notebook and contains the three metrics I later used in VectorExploration. It uses vectors from Hamilton et al., which are taken from a broader, more historical English corpus. Validating the metrics on a known corpus was necessary because any noise they picked up there would likely be amplified in my compressed corpus. 

SGNSTREmbeddings is an early version of the embedding pipeline I used. Instead of relying on Procrustes alignment to make vectors comparable across years, I embedded all the words in the same space, tagging each one by year. This eliminates the need for any alignment technique and makes cosine similarity viable.

Hedging is an alternative version of the hypothesis that used the co-occurrence of hedging words to try and determine the semantic bleaching path. My intuition was that nouns should co-occur with hedging words (like "like") as their usage became more cultural: "she's like a Karen" versus "she's a Karen." There was no relationship between the two. I decided to explore this alternative because of its robustness to word frequency.

Also included in the repo is driftcache.py, a caching script authored by Claude Code to help improve performance and manage memory in my notebooks. 

## Setup
To run any of these notebooks, download the vectors included in the releases. The subreddit vectors have been cleaned to remove bot content and were embedded using temporal referencing from [Dubossarsky et al., 2019](https://aclanthology.org/P19-1044/). This technique eliminates the need for Procrustes alignment, which helps reduce noise significantly. I used word2vecf (Levy & Goldberg) under the hood.

Once you have the vectors, run the following to unzip and move them to the expected locations: 

```bash
mkdir -p vectors/5sub
zstd -d sgns.words.zst      -o vectors/5sub/sgns.words
zstd -d sgns-3sub.words.zst -o vectors/sgns.words
```

`BleachingTrajectories` and `TNPosPredictor` use the 5sub space vectors, while `VectorExploration` uses the smaller, 3 subreddit set. `GeometricSignals` uses Hamilton et al.'s vectors, available [here](https://nlp.stanford.edu/projects/histwords/). The decade files should be placed in `./sgns/`. 

`TNPosPredictor` also needs the [Brysbaert et al. ratings](https://doi.org/10.3758/s13428-013-0403-5). Save the tab-separated ratings file as `data/brysbaert_concreteness.txt`. 

Both `Hedging` and `SGNSTREmbeddings` use a slightly processed version of the raw corpus; the supporting files aren't included, but the underlying corpus is available from Cornell's ConvoKit. 

## References

Baumgartner, J., Zannettou, S., Keegan, B., Squire, M., & Blackburn, J. (2020). The
Pushshift Reddit Dataset. *Proceedings of the International AAAI Conference on Web and
Social Media*, 14(1), 830–839. https://ojs.aaai.org/index.php/ICWSM/article/view/7347

Brysbaert, M., Warriner, A. B., & Kuperman, V. (2014). Concreteness ratings for 40
thousand generally known English word lemmas. *Behavior Research Methods*, 46(3),
904–911. https://doi.org/10.3758/s13428-013-0403-5

Chang, J. P., Chiam, C., Fu, L., Wang, A. Z., Zhang, J., & Danescu-Niculescu-Mizil, C.
(2020). ConvoKit: A Toolkit for the Analysis of Conversations. *Proceedings of SIGDIAL
2020*, 57–60. https://aclanthology.org/2020.sigdial-1.8/

Dubossarsky, H., Weinshall, D., & Grossman, E. (2017). Outta Control: Laws of Semantic
Change and Inherent Biases in Word Representation Models. *EMNLP 2017*, 1136–1145.
https://aclanthology.org/D17-1118/

Dubossarsky, H., Hengchen, S., Tahmasebi, N., & Schlechtweg, D. (2019). Time-Out:
Temporal Referencing for Robust Modeling of Lexical Semantic Change. *ACL 2019*,
457–470. https://aclanthology.org/P19-1044/

Fellbaum, C. (Ed.). (1998). *WordNet: An Electronic Lexical Database*. MIT Press.

Hamilton, W. L., Leskovec, J., & Jurafsky, D. (2016a). Diachronic Word Embeddings
Reveal Statistical Laws of Semantic Change. *ACL 2016*, 1489–1501.
https://aclanthology.org/P16-1141/

Hamilton, W. L., Leskovec, J., & Jurafsky, D. (2016b). Cultural Shift or Linguistic
Drift? Comparing Two Computational Measures of Semantic Change. *EMNLP 2016*,
2116–2121. https://aclanthology.org/D16-1229/

Levy, O., & Goldberg, Y. (2014). Dependency-Based Word Embeddings. *ACL 2014*,
302–308. https://aclanthology.org/P14-2050/

Luo, Y., Jurafsky, D., & Levin, B. (2019). From Insanely Jealous to Insanely Delicious:
Computational Models for the Semantic Bleaching of English Intensifiers. *Proceedings
of the 1st International Workshop on Computational Approaches to Historical Language
Change (LChange)*. https://aclanthology.org/W19-4701/

Pavlick, E., & Callison-Burch, C. (2016). Most "babies" are "little" and most
"problems" are "huge": Compositional Entailment in Adjective-Nouns. *ACL 2016*,
2164–2173. https://aclanthology.org/P16-1204/

Ramiro, C., Srinivasan, M., Malt, B. C., & Xu, Y. (2018). Algorithms in the historical
emergence of word senses. *PNAS*, 115(10), 2323–2328.
https://doi.org/10.1073/pnas.1714730115

Schulte im Walde, S., & Frassinelli, D. (2022). Distributional Measures of Semantic
Abstraction. *Frontiers in Artificial Intelligence*, 4:796756.
https://doi.org/10.3389/frai.2021.796756

Sweetser, E. (1988). Grammaticalization and semantic bleaching. *Proceedings of the
Fourteenth Annual Meeting of the Berkeley Linguistics Society*, 389–405.

Sweetser, E. (1990). *From Etymology to Pragmatics: Metaphorical and Cultural Aspects
of Semantic Structure*. Cambridge University Press.


Wiktionary synonym data retrieved via [wiktextract](https://github.com/tatuylonen/wiktextract).
Corpus data was taken from Cornell's [ConvoKit](https://github.com/CornellNLP/ConvoKit). 

