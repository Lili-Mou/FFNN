# FFNN

This is an **obsolete** package for neural networks. However, I believe this package helps beginners to implement neural networks in a low-level fashion. Hence, the package might still be useful for pedagogical purposes.

By "feed-foward," I mean that all other structures (e.g., recurrent, recursive, convolutional) are unrolled to a feed-forward net. Also included is an example with LSTM-based network for question classification.

## Run the code

**Step 1**: Install BLAS, and CBLAS. They are also included in the package, but you may need to recompile them should they not work.

**Step 2**: Construct the network structures, each asscociated with a particular data sample.

    cd construct_nn/QC/
    python process.py
    python mix.py

**Step 3**: Train the networks

    cd ../../FFNN
    sh compile.sh
    ./QC

## Cite the paper

Please cite the paper if you use it for research purposes:

[1] Lili Mou, Hao Peng, Ge Li, Yan Xu, Lu Zhang, Zhi Jin. "Discriminative neural sentence modeling by tree-based convolution." In *Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, pages 2315--2325, 2015.

  
  
This package has also supported the following papers.

[2] Yan Xu, Lili Mou, Ge Li, Yunchuan Chen, Hao Peng and Zhi Jin. "Classifying relations via long short term memory networks along shortest dependency paths." In *Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, pages 1785--1794, 2015.

[3] Hao Peng, Lili Mou, Ge Li, Yunchuan Chen, Yangyang Lu, Zhi Jin. "A comparative study on regularization strategies for embedding-based neural networks." In *Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing (EMNLP-short)*, pages 2106--2111, 2015.

[4] Lili Mou, Ge Li, Lu Zhang, Tao Wang, Zhi Jin. "Convolutional neural networks over tree structures for programming language processing." In *Proceedings of the Thirtieth AAAI Conference on Artificial Intelligence (AAAI)*, pages 1287--1293, 2016.

[5] Lili Mou, Rui Men, Ge Li, Yan Xu, Lu Zhang, Rui Yan, Zhi Jin. "Natural language inference by tree-based convolution and heuristic matching." In *Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (ACL-short)*, volume 2, pages 130--136, 2016.

[6] Lili Mou, Zhao Meng, Rui Yan, Ge Li, Yan Xu, Lu Zhang, Zhi Jin. How transferable are neural networks in NLP applications? In *Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, pages 478--489, 2016.

[7] Yan Xu, Ran Jia, Lili Mou, Ge Li, Yunchuan Chen, Yangyang Lu, Zhi Jin. Improved relation classification by deep recurrent neural networks with data augmentation. In *Proceedings of the 26th International Conference on Computational Linguistics (COLING)*, pages 1461--1470, 2016.

[8] Lili Mou, Yiping Song, Rui Yan, Ge Li, Lu Zhang, Zhi Jin. Sequence to backward and forward sequences: A content-introducing approach to generative short-text conversation. In *Proceedings of the 26th International Conference on Computational Linguistics (COLING)*, pages 3349--3358, 2016.

# Contact

Sorry, I do not answer questions regarding implementation due to the time constaints on my side.
