#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "D:\\imageprocessing\\lab4\\tensors.h"


// ReLU function
float relu(float x) {
    return x > 0 ? x : 0;
}

// Softmax function
void softmax(float input[], int n) {
    float max_val = input[0];
    float sum = 0;

    // Tìm giá trị lớn nhất để tránh hiện tượng tràn số mũ
    for (int i = 1; i < n; ++i) {
        if (input[i] > max_val) max_val = input[i];
    }

    // Tính giá trị softmax
    for (int i = 0; i < n; ++i) {
        input[i] = exp(input[i] - max_val);
        sum += input[i];
    }

    // Chuẩn hóa để tổng xác suất bằng 1
    for (int i = 0; i < n; ++i) {
        input[i] /= sum;
    }
}


void Prediction(float image[28][28],
                float w_conv1[6][1][1],
                float w_conv2[16][6][5][5],
                float w_fc1[120][400],
                float w_fc2[84][120],
                float w_fc3[10][84],
                float b_conv1[6],
                float b_conv2[16],
                float b_fc1[120],
                float b_fc2[84],
                float b_fc3[10],
                float probs[10])
{

    // 1. Conv1 + Activation
    float conv1_output[6][28][28] = {0};
    for (int f = 0; f < 6; ++f) {
        for (int i = 0; i < 28; ++i) {
            for (int j = 0; j < 28; ++j) {
                conv1_output[f][i][j] = image[i][j] * w_conv1[f][0][0] + b_conv1[f];
                conv1_output[f][i][j] = relu(conv1_output[f][i][j]);
            }
        }
    }

    // 2. Pool1
    float pool1_output[6][14][14] = {0};
    for (int f = 0; f < 6; ++f) {
        for (int i = 0; i < 14; ++i) {
            for (int j = 0; j < 14; ++j) {
                float sum = 0;
                for (int di = 0; di < 2; ++di) {
                    for (int dj = 0; dj < 2; ++dj) {
                        sum += conv1_output[f][i * 2 + di][j * 2 + dj];
                    }
                }
                pool1_output[f][i][j] = sum / 4;
            }
        }
    }

    // 3. Conv2 + Activation
    float conv2_output[16][10][10] = {0};
    for (int f = 0; f < 16; ++f) {
        for (int i = 0; i < 10; ++i) {
            for (int j = 0; j < 10; ++j) {
                for (int c = 0; c < 6; ++c) {
                    for (int di = 0; di < 5; ++di) {
                        for (int dj = 0; dj < 5; ++dj) {
                            conv2_output[f][i][j] += pool1_output[c][i + di][j + dj] * w_conv2[f][c][di][dj];
                        }
                    }
                }
                conv2_output[f][i][j] += b_conv2[f];
                conv2_output[f][i][j] = relu(conv2_output[f][i][j]);
            }
        }
    }

    // 4. Pool2
    float pool2_output[16][5][5] = {0};
    for (int f = 0; f < 16; ++f) {
        for (int i = 0; i < 5; ++i) {
            for (int j = 0; j < 5; ++j) {
                float sum = 0;
                for (int di = 0; di < 2; ++di) {
                    for (int dj = 0; dj < 2; ++dj) {
                        sum += conv2_output[f][i * 2 + di][j * 2 + dj];
                    }
                }
                pool2_output[f][i][j] = sum / 4;
            }
        }
    }

    // Flatten Pool2
    float flat_output[400] = {0};
    for (int f = 0; f < 16; ++f) {
        for (int i = 0; i < 5; ++i) {
            for (int j = 0; j < 5; ++j) {
                flat_output[f * 25 + i * 5 + j] = pool2_output[f][i][j];
            }
        }
    }

    // 5. FC1 + Activation
    float fc1_output[120] = {0};
    for (int i = 0; i < 120; ++i) {
        for (int j = 0; j < 400; ++j) {
            fc1_output[i] += flat_output[j] * w_fc1[i][j];
        }
        fc1_output[i] += b_fc1[i];
        fc1_output[i] = relu(fc1_output[i]);
    }

    // 6. FC2 + Activation
    float fc2_output[84] = {0};
    for (int i = 0; i < 84; ++i) {
        for (int j = 0; j < 120; ++j) {
            fc2_output[i] += fc1_output[j] * w_fc2[i][j];
        }
        fc2_output[i] += b_fc2[i];
        fc2_output[i] = relu(fc2_output[i]);
    }

    // 7. FC3 (Output Layer)
    for (int i = 0; i < 10; ++i) {
        probs[i] = 0;
        for (int j = 0; j < 84; ++j) {
            probs[i] += fc2_output[j] * w_fc3[i][j];
        }
        probs[i] += b_fc3[i];
    }

    // 8. Softmax
    softmax(probs, 10);

}
                     
int main(int argc, char** argv){

   //float image[28][28];
   float w_conv1[6][1][1];
   float w_conv2[16][6][5][5];
   float w_fc1[120][400];
   float w_fc2[84][120];
   float w_fc3[10][84];
   float b_conv1[6];
   float b_conv2[16];
   float b_fc1[120];
   float b_fc2[84];
   float b_fc3[10];
   float probs[10];

   int i,j,m,n,index;
   FILE *fp;

    /* Load Weights from DDR->LMM */
   fp = fopen("D:\\imageprocessing\\lab4\\data\\weights\\w_conv1.txt", "r");
   for(i=0;i<6;i++)
       fscanf(fp, "%f ",  &(w_conv1[i][0][0]));  fclose(fp);

   fp = fopen("D:\\imageprocessing\\lab4\\data\\weights\\w_conv2.txt", "r");
   for(i=0;i<16;i++){
       for(j=0;j<6;j++){
           for(m=0;m<5;m++){
               for(n=0;n<5;n++){
                   index = 16*i + 6*j + 5*m + 5*n;
                   fscanf(fp, "%f ",  &(w_conv2[i][j][m][n]));
               }
           }
       }
   }
   fclose(fp);

   fp = fopen("D:\\imageprocessing\\lab4\\data\\weights\\w_fc1.txt", "r");
   for(i=0;i<120;i++){
       for(j=0;j<400;j++)
           fscanf(fp, "%f ",  &(w_fc1[i][j]));
   }
   fclose(fp);

   fp = fopen("D:\\imageprocessing\\lab4\\data\\weights\\w_fc2.txt", "r");
   for(i=0;i<84;i++){
       for(j=0;j<120;j++)
           fscanf(fp, "%f ",  &(w_fc2[i][j]));
   }
   fclose(fp);

   fp = fopen("D:\\imageprocessing\\lab4\\data\\weights\\w_fc3.txt", "r");
   for(i=0;i<10;i++){
       for(j=0;j<84;j++)
           fscanf(fp, "%f ",  &(w_fc3[i][j]));
   }
   fclose(fp);

   fp = fopen("D:\\imageprocessing\\lab4\\data\\weights\\b_conv1.txt", "r");
   for(i=0;i<6;i++)
       fscanf(fp, "%f ",  &(b_conv1[i]));  fclose(fp);

   fp = fopen("D:\\imageprocessing\\lab4\\data\\weights\\b_conv2.txt", "r");
   for(i=0;i<16;i++)
       fscanf(fp, "%f ",  &(b_conv2[i]));  fclose(fp);

   fp = fopen("D:\\imageprocessing\\lab4\\data\\weights\\b_fc1.txt", "r");
   for(i=0;i<120;i++)
       fscanf(fp, "%f ",  &(b_fc1[i]));  fclose(fp);

   fp = fopen("D:\\imageprocessing\\lab4\\data\\weights\\b_fc2.txt", "r");
   for(i=0;i<84;i++)
       fscanf(fp, "%f ",  &(b_fc2[i]));  fclose(fp);

   fp = fopen("D:\\imageprocessing\\lab4\\data\\weights\\b_fc3.txt", "r");
   for(i=0;i<10;i++)
       fscanf(fp, "%f ",  &(b_fc3[i]));  fclose(fp);

   float *dataset = (float*)malloc(LABEL_LEN*28*28 *sizeof(float));
   int target[LABEL_LEN];

   fp = fopen("D:\\imageprocessing\\lab4\\data\\mnist-test-target.txt", "r");
   for(i=0;i<LABEL_LEN;i++)
       fscanf(fp, "%d ",  &(target[i]));  fclose(fp);

   fp = fopen("D:\\imageprocessing\\lab4\\data\\mnist-test-image.txt", "r");
   for(i=0;i<LABEL_LEN*28*28;i++)
       fscanf(fp, "%f ",  &(dataset[i]));  fclose(fp);

   float image[28][28];
   float *datain;
   int acc = 0;
   int mm, nn;
   for(i=0;i<LABEL_LEN;i++) {

       datain = &dataset[i*28*28];
       for(mm=0;mm<28;mm++)
           for(nn=0;nn<28;nn++)
               image[mm][nn] = *(float*)&datain[28*mm + nn];

       Prediction(   image,
                     w_conv1,
                     w_conv2,
                     w_fc1,
                     w_fc2,
                     w_fc3,
                     b_conv1,
                     b_conv2,
                     b_fc1,
                     b_fc2,
                     b_fc3,
                     probs
                     );

       int index = 0;
       float max = probs[0];
       for (j=1;j<10;j++) {
            if (probs[j] > max) {
                index = j;
                max = probs[j];
            }
       }

       if (index == target[i]) acc++;
       printf("true lable : %d\n",target[i]);
       printf("Predicted label: %d\n", index);
       printf("Prediction: %d/%d\n", acc, i+1);
   }
   printf("Accuracy = %f\n", acc*1.0f/LABEL_LEN);
   

    return 0;
}

