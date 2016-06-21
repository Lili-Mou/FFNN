/*
 * read_data.h
 *
 *  Created on: Mar 10, 2015
 *      Author: mou
 */

#ifndef READ_DATA_H_
#define READ_DATA_H_

#include"global.h"
#include"FFNN.h"

Layer ** ReadOneNet(const char * filepath, int& len);
void ReadLabels(const char * labelfile);
void ReadAllData();
void DeleteClass(Layer ** layers, int len);

void ConstructNetFromBuf( Layer ** & one_net, int i,int number, char * & cursor);
void ReadOneFile(const char * filename, int num_sample, Layer *** &net_array, int *& len);

Layer** ReadOneXFromBuf(int i ,int number,char *& cursor);
void ReadToBuf(const char * filename);

#endif /* READ_DATA_H_ */
