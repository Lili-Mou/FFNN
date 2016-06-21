# FFNN

This is a package for feed-forward neural networks. By ``feed-foward,'' I mean that all other structures (e.g., recurrent, recursive, convolutional) are unrolled to a feed-forward net. Also included is an example with LSTM-based network for question classification.

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

  
  
Currently, this package has also supported the following papers.

[2] Yan Xu, Lili Mou, Ge Li, Yunchuan Chen, Hao Peng and Zhi Jin. "Classifying relations via long short term memory networks along shortest dependency paths." In *Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, pages 1785--1794, 2015.

[3] Hao Peng, Lili Mou, Ge Li, Yunchuan Chen, Yangyang Lu, Zhi Jin. "A comparative study on regularization strategies for embedding-based neural networks." In *Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing (EMNLP-short)*, pages 2106--2111, 2015.

[4] Lili Mou, Ge Li, Lu Zhang, Tao Wang, Zhi Jin. "Convolutional neural networks over tree structures for programming language processing." In *Proceedings of the Thirtieth AAAI Conference on Artificial Intelligence (AAAI)*, pages 1287--1293, 2016.

[5] Lili Mou, Rui Men, Ge Li, Yan Xu, Lu Zhang, Rui Yan, Zhi Jin. "Natural language inference by tree-based convolution and heuristic matching." To appear in *ACL(2)*, 2016.

# Contact

If you have any problem, please don't hesitate to contact me: 
doublepower.mou at the mail server provided by google (gmail)
