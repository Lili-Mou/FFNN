rm QC 
rm src/*.o
g++ -I ../BLAS/CBLAS/include -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"src/FFNN.d" -MT"src/FFNN.d" -o "src/FFNN.o" "src/FFNN.cpp"
g++ -I ../BLAS/CBLAS/include -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"src/activation.d" -MT"src/activation.d" -o "src/activation.o" "src/activation.cpp"
g++ -I ../BLAS/CBLAS/include -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"src/global.d" -MT"src/global.d" -o "src/global.o" "src/global.cpp"
g++ -I ../BLAS/CBLAS/include -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"src/main.d" -MT"src/main.d" -o "src/main.o" "src/main.cpp"
g++ -I ../BLAS/CBLAS/include -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"src/read_data.d" -MT"src/read_data.d" -o "src/read_data.o" "src/read_data.cpp"
#g++ -I../BLAS/CBLAS/include -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"src/test.d" -MT"src/test.d" -o "src/test.o" "src/test.cpp"
g++ -I ../BLAS/CBLAS/include -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"src/train.d" -MT"src/train.d" -o "src/train.o" "src/train.cpp"
g++ -o "QC" ./src/FFNN.o ./src/activation.o ./src/global.o ./src/main.o ./src/read_data.o  ./src/train.o ../BLAS/CBLAS/lib/cblas_LINUX.a ../BLAS/BLAS/blas_LINUX.a -lgfortran -lpthread

