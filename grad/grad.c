#include <stdio.h>
#include <stdlib.h>
#include <math.h>
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
#define M_ROWS 3
#define M_COLS 3
#define N_ROWS M_COLS
#define N_COLS 4

#define A_ROWS 3
#define A_COLS 4
#define B_ROWS A_COLS
#define B_COLS 3

#define B1_ROWS B_COLS
#define B1_COLS 2

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
void matrix_multiply(double *A, double *B, double *C, int A_rows, int A_cols, int B_cols) {
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
void relu(double *matrix, int rows, int cols) {
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
void relu_derivative(double *matrix, double *grad, int rows, int cols) {
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
void softmax(double *matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        double sum = 0.0;
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
double cross_entropy_loss(double *predictions, int *labels, double *grad, int rows, int cols) {
    double loss = 0.0;

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
void compute_gradient_relu(double *A, double *C_grad, double *B_grad, double *Relu_grad,int A_rows, int A_cols, int B_cols) {
    for (int i = 0; i < A_cols; i++) {
        for (int j = 0; j < B_cols; j++) {
            B_grad[i * B_cols + j] = 0;
            for (int k = 0; k < A_rows; k++) {
                B_grad[i * B_cols + j] += A[k * A_cols + i] * C_grad[k * B_cols + j]*Relu_grad[k * B_cols + j];
            }
        }
    }
}

void compute_gradient(double *A, double *C_grad, double *B_grad,int A_rows, int A_cols, int B_cols) {
    for (int i = 0; i < A_cols; i++) {
        for (int j = 0; j < B_cols; j++) {
            B_grad[i * B_cols + j] = 0;
            for (int k = 0; k < A_rows; k++) {
                B_grad[i * B_cols + j] += A[k * A_cols + i] * C_grad[k * B_cols + j];
            }
        }
    }
}

int main() {
    // 定义矩阵m、n、b、b1
    double M[M_ROWS][M_COLS] = {{1, 0, 0},
                                {1, 0, 0},
                                {0, 1, 0},
                               };
    double N[N_ROWS][N_COLS] = {{1, 0, 1, 0},
                                {1, 2, 0, 1},
                                {1, 0, 1, 1},
                               };

    double B[B_ROWS][B_COLS] = {{1, 1, 0.5},
                            {1, 1, 1},
                            {1, 1, 0.2},
                            {1, 1, 1},
                           };
    
    double B1[B1_ROWS][B1_COLS] = {{1, 0.2},
                                   {1, 1},
                                   {0.5, 1},
                                  };

    double A[A_ROWS][A_COLS];
    double C[A_ROWS][B_COLS];
    double O[M_ROWS][B_COLS];
    double O1[M_ROWS][B1_COLS];
    int labels[M_ROWS] = {0, 1, 0};

    // 前向传播
    matrix_multiply((double *)M, (double *)N, (double *)A, M_ROWS, M_COLS, N_COLS);
    for (int i = 0; i < M_ROWS; i++) {
        for (int j = 0; j < N_COLS; j++) {
            printf("%f ", A[i][j]);
        }
        printf("\n");
    }
    printf("******************************\n");
    matrix_multiply((double *)A, (double *)B, (double *)C, A_ROWS, A_COLS, B_COLS);
    for (int i = 0; i < A_ROWS; i++) {
        for (int j = 0; j < B_COLS; j++) {
            printf("%f ", C[i][j]);
        }
        printf("\n");
    }
    printf("******************************\n");
    //ReLU激活函数以及激活函数的梯度
    double grad_Relu[A_ROWS][B_COLS];
    relu_derivative((double *)C, (double *)grad_Relu, A_ROWS, B_COLS);
    relu((double *)C, A_ROWS, B_COLS);
    matrix_multiply((double *)M, (double *)C, (double *)O, M_ROWS, B_COLS, B_COLS);
    for (int i = 0; i < M_ROWS; i++) {
        for (int j = 0; j < B_COLS; j++) {
            printf("%f ", O[i][j]);
        }
        printf("\n");
    }
    printf("******************************\n");
    matrix_multiply((double *)O, (double *)B1, (double *)O1, M_ROWS, B_COLS, B1_COLS);
    for (int i = 0; i < M_ROWS; i++) {
        for (int j = 0; j < B1_COLS; j++) {
            printf("%f ", O1[i][j]);
        }
        printf("\n");
    }
    printf("******************************\n");

    // 计算损失和梯度
    softmax((double *)O1, M_ROWS, B1_COLS);
    double grad[A_ROWS][B1_COLS];
    double loss = cross_entropy_loss((double *)O1, labels, (double *)grad, M_ROWS, B1_COLS);
    printf("Loss: %f\n", loss);
    double B1_grad[B_COLS][B1_COLS];
    compute_gradient((double *)O, (double *)grad, (double *)B1_grad, M_ROWS, B_COLS, B1_COLS);
    printf("Gradient of B1:\n");
    for (int i = 0; i < B_COLS; i++) {
        for (int j = 0; j < B1_COLS; j++) {
            printf("%f ", B1_grad[i][j]);
        }
        printf("\n");
    }
    printf("******************************\n");
    double O_grad[M_ROWS][B_COLS];
    compute_gradient((double *)B1, (double *)grad, (double *)O_grad, M_ROWS,B_COLS,B_COLS);
    printf("Gradient of O:\n");
    for (int i = 0; i < M_ROWS; i++) {
        for (int j = 0; j < B_COLS; j++) {
            printf("%f ", O_grad[i][j]);
        }
        printf("\n");
    }
    printf("******************************\n");
    double C1_grad[A_ROWS][B_COLS];
    compute_gradient((double *)M, (double *) O_grad, (double *)C1_grad, M_ROWS, M_COLS, B_COLS);
    printf("Gradient of C1:\n");
    for (int i = 0; i < A_ROWS; i++) {
        for (int j = 0; j < B_COLS; j++) {
            printf("%f ", C1_grad[i][j]);
        }
        printf("\n");
    }
    printf("******************************\n");
    double B_grad[A_COLS][B_COLS];
    compute_gradient((double *)A, (double *)C1_grad, (double *)B_grad, A_ROWS, A_COLS, B_COLS);
    printf("Gradient of B:\n");
    for (int i = 0; i < A_COLS; i++) {
        for (int j = 0; j < B_COLS; j++) {
            printf("%f ", B_grad[i][j]);
        }
        printf("\n");
    }


/*
    double grad_Relu[A_ROWS][B_COLS];
    relu_derivative((double *)C, (double *)grad_Relu, A_ROWS, B_COLS);
    relu((double *)C, A_ROWS, B_COLS);
    softmax((double *)C, A_ROWS, B_COLS);

    // 计算损失和梯度
    double grad[A_ROWS][B_COLS];
    double loss = cross_entropy_loss((double *)C, labels, (double *)grad, A_ROWS, B_COLS);
    printf("Loss: %f\n", loss);
    printf("Gradient of Relu:\n");
    for (int i = 0; i < A_ROWS; i++) {
        for (int j = 0; j < B_COLS; j++) {
            printf("%f ", grad_Relu[i][j]);
        }
        printf("\n");
    }

    // 反向传播
    double B_grad[A_COLS][B_COLS];
    compute_gradient((double *)A, (double *)grad, (double *)B_grad,(double *)grad_Relu, A_ROWS, A_COLS, B_COLS);

    // 打印梯度
    printf("Gradient of B:\n");
    for (int i = 0; i < A_COLS; i++) {
        for (int j = 0; j < B_COLS; j++) {
            printf("%f ", B_grad[i][j]);
        }
        printf("\n");
    }
*/

    return 0;
}
