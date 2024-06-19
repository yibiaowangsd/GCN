#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include"sha.h"


// SHA-256 compression function
void sha256_compress(uint32_t* state, const uint32_t* W) {
    uint32_t a = state[0];
    uint32_t b = state[1];
    uint32_t c = state[2];
    uint32_t d = state[3];
    uint32_t e = state[4];
    uint32_t f = state[5];
    uint32_t g = state[6];
    uint32_t h = state[7];

    for (int i = 0; i < 64; i++) {
        uint32_t S1 = ROTR(e, 6) ^ ROTR(e, 11) ^ ROTR(e, 25);
        uint32_t ch = (e & f) ^ (~e & g);
        uint32_t temp1 = h + S1 + ch + K[i] + W[i];
        uint32_t S0 = ROTR(a, 2) ^ ROTR(a, 13) ^ ROTR(a, 22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t temp2 = S0 + maj;

        h = g;
        g = f;
        f = e;
        e = d + temp1;
        d = c;
        c = b;
        b = a;
        a = temp1 + temp2;
    }

    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
}

// SHA-256 hash function
void sha256_hash(uint8_t* hash, const uint8_t* data, size_t length) {
    uint32_t state[8];
    memcpy(state, H, sizeof(H));

    size_t blocks = (length + 8) / 64 + 1;
    uint8_t* padded_data = malloc(blocks * 64);
    memcpy(padded_data, data, length);
    padded_data[length] = 0x80;
    for (size_t i = length + 1; i < blocks * 64 - 8; i++) {
        padded_data[i] = 0x00;
    }
    uint64_t bit_length = length * 8;
    for (int i = 0; i < 8; i++) {
        padded_data[blocks * 64 - 8 + i] = (bit_length >> (56 - i * 8)) & 0xFF;
    }

    for (size_t i = 0; i < blocks; i++) {
        uint32_t W[64];
        sha256_schedule(W, padded_data + i * 64);
        sha256_compress(state, W);
    }

    for (int i = 0; i < 8; i++) {
        hash[i * 4] = (state[i] >> 24) & 0xFF;
        hash[i * 4 + 1] = (state[i] >> 16) & 0xFF;
        hash[i * 4 + 2] = (state[i] >> 8) & 0xFF;
        hash[i * 4 + 3] = state[i] & 0xFF;
    }

    free(padded_data);
}