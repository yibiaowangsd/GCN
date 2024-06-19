#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "sha.h"

typedef struct Node {
    char hash[SHA256_DIGEST_LENGTH * 2 + 1];
    struct Node* left;
    struct Node* right;
} Node;

Node* createNode(const char* data) {
    Node* node = (Node*)malloc(sizeof(Node));
    if (node == NULL) {
        fprintf(stderr, "Failed to allocate memory for node\n");
        exit(1);
    }
    strncpy(node->hash, data, SHA256_DIGEST_LENGTH * 2 + 1);
    node->left = NULL;
    node->right = NULL;
    return node;
}

Node* createLeafNode(const char* data) {
    Node* node = createNode(data);
    return node;
}

Node* createParentNode(Node* left, Node* right) {
    Node* node = createNode("");
    snprintf(node->hash, SHA256_DIGEST_LENGTH * 2 + 1, "%s%s", left->hash, right->hash);
    node->left = left;
    node->right = right;
    return node;
}

Node* buildMerkleTree(const char** data, int numData) {
    if (numData == 0) {
        return NULL;
    }
    if (numData == 1) {
        return createLeafNode(data[0]);
    }
    int mid = numData / 2;
    Node* left = buildMerkleTree(data, mid);
    Node* right = buildMerkleTree(data + mid, numData - mid);
    return createParentNode(left, right);
}

void printMerkleTree(Node* root) {
    if (root == NULL) {
        return;
    }
    printf("%s\n", root->hash);
    printMerkleTree(root->left);
    printMerkleTree(root->right);
}

int main() {

    /*
    const char* data[] = {"A", "B", "C", "D", "E", "F", "G", "H"};
    int numData = sizeof(data) / sizeof(data[0]);

    Node* root = buildMerkleTree(data, numData);
    printMerkleTree(root);

    return 0;
    */
    const char* message = "111";
    uint8_t hash[32];
    sha256_hash(hash, (const uint8_t*)message, strlen(message));
    for (int i = 0; i < 32; i++) {
        printf("%02x", hash[i]);
    }
    printf("\n");

    return 0;


}