#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
/*
forward:
A=M×N  3×4
C=A×B   3×3
C1=Relu(C) 3×3
O=M×C1 3×3
O1=O×B1 3×2
LOSS=CrossEntropy(O1)
*/

// 矩阵尺寸定义
#define M_ROWS 34
#define M_COLS 34
#define N_ROWS M_COLS
#define N_COLS 34

#define A_ROWS 34
#define A_COLS 34
#define B_ROWS A_COLS
#define B_COLS 7

#define B1_ROWS B_COLS
#define B1_COLS 4

#define train_size 4
#define test_size 28

//优化器参数定义
#define Beta1 0.9
#define Beta2 0.999
#define Alpha 0.01

//添加函数声明
void matrix_multiply(float *A, float *B, float *C, int A_rows, int A_cols, int B_cols);
void relu(float *matrix, int rows, int cols);
void relu_derivative(float *matrix, float *grad, int rows, int cols);
void softmax(float *matrix, int rows, int cols);
float cross_entropy_loss(float *predictions, int *labels, float *grad, int rows, int cols);
void compute_gradient(float *A, float *C_grad, float *B_grad, int A_rows, int A_cols, int B_cols);
void compute_gradient_relu(float *A, float *C_grad, float *B_grad, float *Relu_grad,int A_rows, int A_cols, int B_cols);
void adam_optimizer(float *params, float *grad, float *m, float *v, int row, int col, int t, float beta1, float beta2, float alpha);
void reset_weight(float *matrix, int rows, int cols);
float compute_accuracy(float *predictions, int *labels, int rows, int cols);
float compute_accuracy_train(float *predictions, int *labels, int *train, int tra, int p, int rows, int cols);
void train(float* M,float *N,float *B,float* B1,int *labels,int step);


// 矩阵乘法函数
/**
 * Multiplies two matrices and stores the result in a third matrix.
 *
 * @param A       Pointer to the first matrix (A_rows x A_cols)
 * @param B       Pointer to the second matrix (A_cols x B_cols)
 * @param C       Pointer to the resulting matrix (A_rows x B_cols)
 * @param A_rows  Number of rows in matrix A
 * @param A_cols  Number of columns in matrix A
 * @param B_cols  Number of columns in matrix B
 */
void matrix_multiply(float *A, float *B, float *C, int A_rows, int A_cols, int B_cols) {
    for (int i = 0; i < A_rows; i++) {
        for (int j = 0; j < B_cols; j++) {
            C[i * B_cols + j] = 0;
            for (int k = 0; k < A_cols; k++) {
                C[i * B_cols + j] += A[i * A_cols + k] * B[k * B_cols + j];
            }
        }
    }
}

void matrix_multiply_1(float *A, float *B, float *C, int A_rows, int A_cols, int B_cols) {
    printf("A:\n");
    for (int i = 0; i < A_rows; i++) {
        for (int j = 0; j < A_cols; j++) {
            printf("%0.4f ", A[i * A_cols + j]);
        }
        printf("\n");
    }
    printf("B:\n");
    for (int i = 0; i < A_cols; i++) {
        for (int j = 0; j < B_cols; j++) {
            printf("%0f ", B[i * B_cols + j]);
        }
        printf("\n");
    }

    for (int i = 0; i < A_rows; i++) {
        for (int j = 0; j < B_cols; j++) {
            C[i * B_cols + j] = 0;
            for (int k = 0; k < A_cols; k++) {
                C[i * B_cols + j] += A[i * A_cols + k] * B[k * B_cols + j];
            }
        }
    }
}

// ReLU激活函数
/**
 * Applies the Rectified Linear Unit (ReLU) activation function to a matrix.
 *
 * @param matrix The matrix to apply the ReLU function to.
 * @param rows The number of rows in the matrix.
 * @param cols The number of columns in the matrix.
 */
void relu(float *matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        if (matrix[i] < 0) {
            matrix[i] = 0;
        }
    }
}
//ReLU激活函数的导数
/**
 * Calculates the derivative of the ReLU (Rectified Linear Unit) function for each element in a matrix.
 *
 * @param matrix The input matrix.
 * @param grad The output matrix to store the derivative values.
 * @param rows The number of rows in the matrix.
 * @param cols The number of columns in the matrix.
 */
void relu_derivative(float *matrix, float *grad, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        if (matrix[i] > 0) {
            grad[i] = 1;
        } else {
            grad[i] = 0;
        }
    }
}


// Softmax函数
/**
 * Applies the softmax function to a matrix.
 *
 * The softmax function is a mathematical function that takes a matrix as input
 * and normalizes each row of the matrix so that the values in each row sum up
 * to 1.0. This is commonly used in machine learning algorithms for tasks such
 * as multi-class classification.
 *
 * @param matrix The input matrix to apply the softmax function to.
 * @param rows The number of rows in the matrix.
 * @param cols The number of columns in the matrix.
 */
void softmax(float *matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        float sum = 0.0;
        for (int j = 0; j < cols; j++) {
            matrix[i * cols + j] = exp(matrix[i * cols + j]);
            sum += matrix[i * cols + j];
        }
        for (int j = 0; j < cols; j++) {
            matrix[i * cols + j] /= sum;
        }
    }
}

// 交叉熵损失函数
/**
 * Calculates the cross-entropy loss and gradient for a given set of predictions and labels.
 *
 * @param predictions The array of predicted probabilities for each class.
 * @param labels The array of true labels for each sample.
 * @param grad The array to store the computed gradient.
 * @param rows The number of samples.
 * @param cols The number of classes.
 * @return The average cross-entropy loss.
 */
float cross_entropy_loss(float *predictions, int *labels, float *grad, int rows, int cols) {
    float loss = 0.0;

    // Compute the gradient and loss for each sample
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            grad[i * cols + j] = predictions[i * cols + j];
        }
        grad[i * cols + labels[i]] -= 1;
        loss -= log(predictions[i * cols + labels[i]]);
    }

    // Normalize the gradient by dividing it by the number of samples
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            grad[i * cols + j] /= rows;
        }
    }

    // Calculate the average loss
    return loss / rows;
}


float cross_entropy_loss_train(float *predictions, int *labels, float *grad,int *train,int tra,int p,int rows, int cols) {
    float loss = 0.0;

    // Compute the gradient and loss for each sample
    for (int i = 0; i < rows; i++) {
        if(train[i]==p){
        for (int j = 0; j < cols; j++) {
            grad[i * cols + j] = predictions[i * cols + j];
        }

        grad[i * cols + labels[i]] -= 1;
        loss -= log(predictions[i * cols + labels[i]]);
        }
    }

    // Normalize the gradient by dividing it by the number of samples
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            grad[i * cols + j] /= tra;
        }
    }

    // Calculate the average loss
    return loss / tra;
}

float cross_entropy_loss_test(float *predictions, int *labels,int *train,int tra,int p,int rows, int cols) {
    float loss = 0.0;

    // Compute the gradient and loss for each sample
    for (int i = 0; i < rows; i++) {
        if(train[i]==p){
        loss -= log(predictions[i * cols + labels[i]]);
        }
    }

    // Calculate the average loss
    return loss / tra;
}

// 计算矩阵B的梯度
/**
 * Computes the gradient of matrix B with respect to matrix A and matrix C.
 *
 * @param A         Pointer to the input matrix A
 * @param C_grad    Pointer to the gradient of matrix C
 * @param B_grad    Pointer to the output gradient matrix B
 * @param A_rows    Number of rows in matrix A
 * @param A_cols    Number of columns in matrix A
 * @param B_cols    Number of columns in matrix B
 */
void compute_gradient_relu(float *A, float *C_grad, float *B_grad, float *Relu_grad,int A_rows, int A_cols, int B_cols) {
    for (int i = 0; i < A_cols; i++) {
        for (int j = 0; j < B_cols; j++) {
            B_grad[i * B_cols + j] = 0;
            for (int k = 0; k < A_rows; k++) {
                B_grad[i * B_cols + j] += A[k * A_cols + i] * C_grad[k * B_cols + j]*Relu_grad[k * B_cols + j];
            }
        }
    }
}

/**
 * Computes the gradient of matrix B with respect to matrix A and matrix C.
 *
 * @param A         Pointer to the input matrix A
 * @param C_grad    Pointer to the gradient of matrix C
 * @param B_grad    Pointer to the output gradient matrix B
 * @param A_rows    Number of rows in matrix A
 * @param A_cols    Number of columns in matrix A
 * @param B_cols    Number of columns in matrix B
 */
void compute_gradient(float *A, float *C_grad, float *B_grad, int A_rows, int A_cols, int B_cols) {
    for (int i = 0; i < A_cols; i++) {
        for (int j = 0; j < B_cols; j++) {
            B_grad[i * B_cols + j] = 0;
            for (int k = 0; k < A_rows; k++) {
                B_grad[i * B_cols + j] += A[k * A_cols + i] * C_grad[k * B_cols + j];
            }
        }
    }
}


/**
 * Computes the gradient of matrix A with respect to matrix B and matrix C.
 *
 * @param B         Pointer to the matrix B.
 * @param C_grad    Pointer to the gradient of matrix C.
 * @param A_grad    Pointer to store the gradient of matrix A.
 * @param A_rows    Number of rows in matrix A.
 * @param A_cols    Number of columns in matrix A.
 * @param B_cols    Number of columns in matrix B.
 */
void compute_gradient_left(float *B, float *C_grad, float *A_grad, int A_rows, int A_cols, int B_cols) {
    for (int i = 0; i < A_cols; i++) {
        for (int j = 0; j < A_rows; j++) {
            A_grad[j * A_cols + i] = 0;
            for (int k = 0; k < B_cols; k++) {
                A_grad[j * A_cols + i] += B[i * B_cols + k] * C_grad[j * B_cols + k];
            }
        }
    }
}

//adam优化器
/**
 * Performs the Adam optimization algorithm on the given parameters.
 *
 * @param params The array of parameters to be optimized.
 * @param grad The array of gradients corresponding to the parameters.
 * @param m The array of first moment estimates.
 * @param v The array of second moment estimates.
 * @param row The number of rows in the arrays.
 * @param col The number of columns in the arrays.
 * @param t The current iteration step.
 * @param beta1 The exponential decay rate for the first moment estimates.
 * @param beta2 The exponential decay rate for the second moment estimates.
 * @param alpha The learning rate.
 */
void adam_optimizer(float *params, float *grad, float *m, float *v, int row, int col, int t, float beta1, float beta2, float alpha) {
    float beta1_t = pow(beta1, t);
    float beta2_t = pow(beta2, t);
    for (int i = 0; i < row; i++) {
        for(int j = 0; j < col; j++){
            m[i*col+j] = beta1 * m[i*col+j] + (1 - beta1) * grad[i*col+j];
            v[i*col+j] = beta2 * v[i*col+j] + (1 - beta2) * grad[i*col+j] * grad[i*col+j];
            float m_hat = m[i*col+j] / (1 - beta1_t);
            float v_hat = v[i*col+j] / (1 - beta2_t);
            params[i*col+j] -= alpha * m_hat / (sqrt(v_hat) + 1e-8);
        }
    }
}

// 初始化权重
/**
 * Resets the weights of a matrix using random values within a specified range.
 *
 * @param matrix The matrix whose weights need to be reset.
 * @param rows The number of rows in the matrix.
 * @param cols The number of columns in the matrix.
 */
void reset_weight(float *matrix, int rows, int cols) {
    
    float std = 1.0 / sqrt(cols);
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = (rand() / (float)RAND_MAX) * 2 * std - std;
    }
}
//计算正确率
/**
 * Computes the accuracy of a set of predictions given the corresponding labels.
 *
 * @param predictions An array of floats representing the predicted values.
 * @param labels An array of integers representing the true labels.
 * @param rows The number of rows in the predictions and labels arrays.
 * @param cols The number of columns in the predictions array.
 * @return The accuracy of the predictions as a float value.
 */
float compute_accuracy(float *predictions, int *labels, int rows, int cols) {
    int correct = 0;
    float acc = 0;

    for (int i = 0; i < rows; i++) {
        int max_index = 0;
        float max_value = predictions[i * cols];

        for (int j = 1; j < cols; j++) {
            if (predictions[i * cols + j] > max_value) {
                max_value = predictions[i * cols + j];
                max_index = j;
            }
        }

        if (max_index == labels[i]) {
            correct++;
        }
    }

    acc = (float)correct / rows;

    return acc;
}

/**
 * Computes the accuracy of the training predictions.
 *
 * @param predictions The array of predicted values.
 * @param labels The array of actual labels.
 * @param train The array indicating whether a data point is used for training or not.
 * @param tra The total number of training data points.
 * @param p The value indicating the class for which accuracy is computed.
 * @param rows The number of rows in the predictions and labels arrays.
 * @param cols The number of columns in the predictions array.
 * @return The accuracy of the training predictions as a float value.
 */
float compute_accuracy_train(float *predictions, int *labels, int *train, int tra, int p, int rows, int cols) {
    int correct = 0;
    float acc = 0;

    for (int i = 0; i < rows; i++) {
        if (train[i] == p) {
            int max_index = 0;
            float max_value = predictions[i * cols];

            for (int j = 1; j < cols; j++) {
                if (predictions[i * cols + j] > max_value) {
                    max_value = predictions[i * cols + j];
                    max_index = j;
                }
            }

            if (max_index == labels[i]) {
                correct++;
            }
        }
    }

    acc = (float)correct / tra;

    return acc;
}

//整合训练过程
void train(float* M,float *N,float *B,float* B1,int *labels,int step){
    float A[A_ROWS][A_COLS];
	matrix_multiply((float *)M, (float *)N, (float *)A, M_ROWS, M_COLS, N_COLS);
    float C[A_ROWS][B_COLS];
    float O[M_ROWS][B_COLS];
    float O1[M_ROWS][B1_COLS];
    float grad_relu[A_ROWS][B_COLS];
    float grad[M_ROWS][B1_COLS];
    float B1_grad[M_ROWS][B1_COLS];
    float O_grad[M_ROWS][B_COLS];
    float C_grad[M_ROWS][B_COLS];
    float B_grad[A_COLS][B_COLS];
    int epoch=0;
    while(epoch<step){
    matrix_multiply((float *)A, (float *)B, (float *)C, A_ROWS, A_COLS, B_COLS);
    relu_derivative((float *)C, (float *)grad_relu, A_ROWS, B_COLS);
    relu((float *)C, A_ROWS, B_COLS);
    matrix_multiply((float *)M, (float *)C, (float *)O, M_ROWS,M_ROWS, B_COLS);
    matrix_multiply((float *)O, (float *)B1, (float *)O1, M_ROWS, B_COLS, B1_COLS);
    softmax((float *)O1, M_ROWS, B1_COLS);
    float loss = cross_entropy_loss((float *)O1, labels, (float *)grad, M_ROWS, B1_COLS);
    //printf("step:%d   Loss: %f  \n", step,loss);
    compute_gradient((float *)O, (float *)grad, (float *)B1_grad, M_ROWS, B_COLS, B1_COLS);
    compute_gradient_left((float *)B1, (float *)grad, (float *)O_grad, M_ROWS, B_COLS, B1_COLS);
    compute_gradient((float *)M, (float *)O_grad, (float *)C_grad, M_ROWS, M_COLS, B_COLS);
    compute_gradient_relu((float *)A, (float *)C_grad, (float *)B_grad, (float*) grad_relu,A_ROWS, A_COLS, B_COLS);
    float m_B[A_COLS][B_COLS]={0};
    float v_B[A_COLS][B_COLS]={0};
    float m_B1[B_COLS][B1_COLS]={0};
    float v_B1[B_COLS][B1_COLS]={0};
    adam_optimizer((float *)B, (float *)B_grad, (float *)m_B, (float *)v_B, A_COLS, B_COLS, step, Beta1, Beta2, Alpha);   
    adam_optimizer((float *)B1, (float *)B1_grad, (float *)m_B1, (float *)v_B1, B_COLS, B1_COLS, step, Beta1, Beta2, Alpha);
    epoch++;
    }
}



int main() {
    // 定义矩阵m、n、b、b1
    float M[M_ROWS][M_COLS] = {
0.0588,0.0588,0.0588,0.0588,0.0588,0.0588,0.0588,0.0588,0.0588,0.0000,0.0588,0.0588,0.0588,0.0588,0.0000,0.0000,0.0000,0.0588,0.0000,0.0588,0.0000,0.0588,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0588,0.0000,0.0000,
0.1000,0.1000,0.1000,0.1000,0.0000,0.0000,0.0000,0.1000,0.0000,0.0000,0.0000,0.0000,0.0000,0.1000,0.0000,0.0000,0.0000,0.1000,0.0000,0.1000,0.0000,0.1000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.1000,0.0000,0.0000,0.0000,
0.0909,0.0909,0.0909,0.0909,0.0000,0.0000,0.0000,0.0909,0.0909,0.0909,0.0000,0.0000,0.0000,0.0909,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0909,0.0909,0.0000,0.0000,0.0000,0.0909,0.0000,
0.1429,0.1429,0.1429,0.1429,0.0000,0.0000,0.0000,0.1429,0.0000,0.0000,0.0000,0.0000,0.1429,0.1429,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,
0.2500,0.0000,0.0000,0.0000,0.2500,0.0000,0.2500,0.0000,0.0000,0.0000,0.2500,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,
0.2000,0.0000,0.0000,0.0000,0.0000,0.2000,0.2000,0.0000,0.0000,0.0000,0.2000,0.0000,0.0000,0.0000,0.0000,0.0000,0.2000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,
0.2000,0.0000,0.0000,0.0000,0.2000,0.2000,0.2000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.2000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,
0.2000,0.2000,0.2000,0.2000,0.0000,0.0000,0.0000,0.2000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,
0.1667,0.0000,0.1667,0.0000,0.0000,0.0000,0.0000,0.0000,0.1667,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.1667,0.0000,0.1667,0.1667,
0.0000,0.0000,0.3333,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.3333,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.3333,
0.2500,0.0000,0.0000,0.0000,0.2500,0.2500,0.0000,0.0000,0.0000,0.0000,0.2500,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,
0.5000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.5000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,
0.3333,0.0000,0.0000,0.3333,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.3333,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,
0.1667,0.1667,0.1667,0.1667,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.1667,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.1667,
0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.3333,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.3333,0.3333,
0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.3333,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.3333,0.3333,
0.0000,0.0000,0.0000,0.0000,0.0000,0.3333,0.3333,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.3333,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,
0.3333,0.3333,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.3333,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,
0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.3333,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.3333,0.3333,
0.2500,0.2500,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.2500,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.2500,
0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.3333,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.3333,0.3333,
0.3333,0.3333,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.3333,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,
0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.3333,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.3333,0.3333,
0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.1667,0.0000,0.1667,0.0000,0.1667,0.0000,0.1667,0.0000,0.0000,0.1667,0.1667,
0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.2500,0.2500,0.0000,0.2500,0.0000,0.0000,0.0000,0.2500,0.0000,0.0000,
0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.2500,0.2500,0.2500,0.0000,0.0000,0.0000,0.0000,0.0000,0.2500,0.0000,0.0000,
0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.3333,0.0000,0.0000,0.3333,0.0000,0.0000,0.0000,0.3333,
0.0000,0.0000,0.2000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.2000,0.2000,0.0000,0.0000,0.2000,0.0000,0.0000,0.0000,0.0000,0.0000,0.2000,
0.0000,0.0000,0.2500,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.2500,0.0000,0.0000,0.2500,0.0000,0.2500,
0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.2000,0.0000,0.0000,0.2000,0.0000,0.0000,0.2000,0.0000,0.0000,0.2000,0.2000,
0.0000,0.2000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.2000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.2000,0.0000,0.2000,0.2000,
0.1429,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.1429,0.1429,0.0000,0.0000,0.1429,0.0000,0.0000,0.1429,0.1429,0.1429,
0.0000,0.0000,0.0769,0.0000,0.0000,0.0000,0.0000,0.0000,0.0769,0.0000,0.0000,0.0000,0.0000,0.0000,0.0769,0.0769,0.0000,0.0000,0.0769,0.0000,0.0769,0.0000,0.0769,0.0769,0.0000,0.0000,0.0000,0.0000,0.0000,0.0769,0.0769,0.0769,0.0769,0.0769,
0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0556,0.0556,0.0000,0.0000,0.0000,0.0556,0.0556,0.0556,0.0000,0.0000,0.0556,0.0556,0.0556,0.0000,0.0556,0.0556,0.0000,0.0000,0.0556,0.0556,0.0556,0.0556,0.0556,0.0556,0.0556,0.0556

                               };
    float N[N_ROWS][N_COLS] = {
1.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,
0.0000,1.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,
0.0000,0.0000,1.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,
0.0000,0.0000,0.0000,1.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,
0.0000,0.0000,0.0000,0.0000,1.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,
0.0000,0.0000,0.0000,0.0000,0.0000,1.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,
0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,1.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,
0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,1.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,
0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,1.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,
0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,1.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,
0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,1.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,
0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,1.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,
0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,1.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,
0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,1.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,
0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,1.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,
0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,1.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,
0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,1.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,
0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,1.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,
0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,1.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,
0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,1.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,
0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,1.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,
0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,1.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,
0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,1.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,
0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,1.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,
0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,1.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,
0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,1.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,
0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,1.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,
0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,1.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,
0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,1.0000,0.0000,0.0000,0.0000,0.0000,0.0000,
0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,1.0000,0.0000,0.0000,0.0000,0.0000,
0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,1.0000,0.0000,0.0000,0.0000,
0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,1.0000,0.0000,0.0000,
0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,1.0000,0.0000,
0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,1.0000
                               };
    int labels[M_ROWS] = {1, 1, 1, 1, 3, 3, 3, 1, 0, 1, 3, 1, 1, 1, 0, 0, 3, 1, 0, 1, 0, 1,0, 0, 2, 2, 0, 0, 2, 0, 0, 2, 0, 0};
    float B[B_ROWS][B_COLS]={
-0.0067, -0.0979,  0.1555,  0.1624, -0.3613,  0.1765, -0.1926,
-0.1469, -0.2907,  0.3215, -0.3097, -0.3466, -0.2912,  0.1274,
-0.0346, -0.0537, -0.0518,  0.1331,  0.0956,  0.2211,  0.3203,
-0.2334, -0.0721, -0.2071, -0.3647,  0.1919,  0.0802,  0.3545,
0.2658,  0.0565,  0.2383, -0.0985, -0.2572, -0.2338,  0.2705,
-0.0675,  0.0326, -0.2796, -0.0086,  0.3588, -0.0083,  0.2505,
 0.0551,  0.2199, -0.3640,  0.3416, -0.1995,  0.0864, -0.0613,
-0.0009, -0.0545,  0.1702, -0.3650, -0.3699, -0.0327,  0.0987,
-0.3217,  0.0211,  0.0316,  0.0160, -0.0559, -0.2106, -0.1664,
 0.2924,  0.0965,  0.2484,  0.3271, -0.2917, -0.0824, -0.0667,
 0.1384,  0.1329, -0.0228, -0.0709,  0.2052,  0.0793,  0.1876,
 0.0108, -0.0514,  0.0372, -0.0606,  0.3406,  0.3222, -0.0585,
-0.3446,  0.2260, -0.2815, -0.1424,  0.3563, -0.0613,  0.0836,
 0.0778, -0.0999, -0.3064,  0.2081,  0.0870, -0.3312,  0.0705,
 0.0902,  0.3731, -0.2942,  0.0602, -0.1989, -0.1406,  0.0168,
 0.0665,  0.0411, -0.0019,  0.1037,  0.3393, -0.2706,  0.0531,
 0.2649, -0.0616,  0.3002,  0.2477, -0.3505,  0.0652, -0.2607,
 0.1687, -0.0340, -0.0337, -0.3665, -0.1554, -0.1063,  0.1555,
 0.1429,  0.3735, -0.0870, -0.2123,  0.3259,  0.0839, -0.2072,
 -0.0520,  0.2944,  0.0963,  0.2360, -0.1308,  0.2217,  0.3773,
 0.3160, -0.3254,  0.1366, -0.1335,  0.3639, -0.0809,  0.3182,
 0.1531,  0.3170, -0.0296,  0.2408,  0.2688,  0.2937,  0.1375,
-0.0789,  0.2180,  0.3144, -0.1388,  0.2979,  0.2826, -0.1092,
-0.0174,  0.0308, -0.1514, -0.2622, -0.2129, -0.3462,  0.2828,
-0.2724, -0.1512,  0.0490, -0.1736,  0.1885,  0.2685,  0.1665,
 0.2356,  0.0329, -0.2088,  0.3659,  0.2555, -0.3038, -0.2782,
 0.3767, -0.2362,  0.3413, -0.1426,  0.2768,  0.0349, -0.2989,
 0.3263,  0.1610, -0.2293, -0.2160, -0.2750, -0.1942,  0.2584,
 0.0763,  0.3688,  0.0166, -0.0153,  0.0363, -0.3639,  0.2911,
-0.0228,  0.2492, -0.1204, -0.1872, -0.0891,  0.0171, -0.3567,
-0.3175,  0.1809,  0.0608, -0.0320, -0.0588, -0.0368,  0.2116,
 0.0320, -0.0156, -0.0399, -0.1726, -0.3114,  0.2177, -0.1038,
-0.3649,  0.0619, -0.2170,  0.1325,  0.1947,  0.3417,  0.2214,
-0.0919, -0.0247, -0.0388, -0.0421,  0.3415, -0.1473, -0.2749

    };
    
    float B1[B1_ROWS][B1_COLS]={

-0.4285,  0.4888,  0.3976,  0.1055,
0.2318,  0.3024,  0.2644, -0.4615,
-0.2953, -0.2766, -0.2321,  0.1456,
-0.3224, -0.0291,  0.2187,  0.3660,
 0.0181, -0.3088, -0.1990, -0.0752,
 0.2805,  0.4219, -0.3647,  0.0464,
 0.1903,  0.4961, -0.0397,  0.4157
    
};
    reset_weight((float *)B, B_ROWS, B_COLS);
    
    
    reset_weight((float *)B1, B1_ROWS, B1_COLS);
    int idx_train[M_ROWS]={1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0};
    //int idx_test[test_size]={1,2,3,5,6,7,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,25,26,27,28,29,30,31,32,33};
    int train_tra=4;
    int test_tra=28;
    int train_p=1;
    int test_p=0;

    float A[A_ROWS][A_COLS];
    matrix_multiply((float *)M, (float *)N, (float *)A, M_ROWS, M_COLS, N_COLS);
    
    float C[A_ROWS][B_COLS];
    float O[M_ROWS][B_COLS];
    float O1[M_ROWS][B1_COLS];
    float grad_Relu[A_ROWS][B_COLS];
    float grad[A_ROWS][B1_COLS];
    float B1_grad[B_COLS][B1_COLS];
    float O_grad[M_ROWS][B_COLS];
    float C_grad[A_ROWS][B_COLS];
    float B_grad[A_COLS][B_COLS];
    int step=1;
    train((float *)M,(float *)N,(float *)B,(float *)B1,labels,200);



while(step<2){
    // 前向传播
    matrix_multiply((float *)A, (float *)B, (float *)C, A_ROWS, A_COLS, B_COLS);
    //ReLU激活函数以及激活函数的梯度
    relu_derivative((float *)C, (float *)grad_Relu, A_ROWS, B_COLS);
    relu((float *)C, A_ROWS, B_COLS);
    matrix_multiply((float *)M, (float *)C, (float *)O, M_ROWS,M_ROWS, B_COLS);
    matrix_multiply((float *)O, (float *)B1, (float *)O1, M_ROWS, B_COLS, B1_COLS);

    // 计算损失和梯度
    softmax((float *)O1, M_ROWS, B1_COLS);
    for(int i=0;i<A_ROWS;i++){
        for(int j=0;j<B1_COLS;j++){
            grad[i][j]=0;
        }
    }
    float loss = cross_entropy_loss((float *)O1, labels, (float *)grad, M_ROWS, B1_COLS);
    //float loss = cross_entropy_loss_train((float *)O1, labels, (float *)grad,idx_train,train_tra,train_p, M_ROWS, B1_COLS);
    printf("step:%d   Loss: %f  ", step,loss);
    //输出正确率
    float accuracy=0;
    accuracy=compute_accuracy((float *)O1, labels, M_ROWS, B1_COLS);
    //accuracy=compute_accuracy_train((float *)O1, labels, idx_train,train_tra,train_p,M_ROWS, B1_COLS);
    printf("accuracy: %f    ",accuracy);

    //float loss_test = cross_entropy_loss_test((float *)O1, labels,idx_train,test_tra,test_p, M_ROWS, B1_COLS);
    //printf("test:  Loss: %f  ",loss_test);
    //float accuracy_test=0;
    //accuracy_test=compute_accuracy_train((float *)O1, labels, idx_train,test_tra,test_p,M_ROWS, B1_COLS);
    //printf("accuracy_test: %f\n",accuracy_test);



    compute_gradient((float *)O, (float *)grad, (float *)B1_grad, M_ROWS, B_COLS, B1_COLS);
    /*
    printf("Gradient of B1:\n");
    for (int i = 0; i < B_COLS; i++) {
        for (int j = 0; j < B1_COLS; j++) {
            printf("%0.4f ", B1_grad[i][j]);
        }
        printf("\n");
    }
    printf("******************************\n");
    */
    compute_gradient_left((float *)B1, (float *)grad, (float *)O_grad, M_ROWS, B_COLS, B1_COLS);
    compute_gradient((float *)M, (float *)O_grad, (float *)C_grad, M_ROWS, M_COLS, B_COLS);
    compute_gradient_relu((float *)A, (float *)C_grad, (float *)B_grad, (float*) grad_Relu,A_ROWS, A_COLS, B_COLS);
    /*
    printf("Gradient of B:\n");
    for (int i = 0; i < A_COLS; i++) {
        for (int j = 0; j < B_COLS; j++) {
            printf("%0.4f ", B_grad[i][j]);
        }
        printf("\n");
    }
    printf("******************************\n");
    */
    //初始化m、v
    float m_B[A_COLS][B_COLS]={0};
    float v_B[A_COLS][B_COLS]={0};
    float m_B1[B_COLS][B1_COLS]={0};
    float v_B1[B_COLS][B1_COLS]={0};
    adam_optimizer((float *)B, (float *)B_grad, (float *)m_B, (float *)v_B, A_COLS, B_COLS, step, Beta1, Beta2, Alpha);   
    adam_optimizer((float *)B1, (float *)B1_grad, (float *)m_B1, (float *)v_B1, B_COLS, B1_COLS, step, Beta1, Beta2, Alpha);
    step+=1;

}


    return 0;
}
