#include <iostream>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <ctime>
#include <sstream>
#include <string>
#include "test_map.hpp"
#include "gpu_hashtable.hpp"

using namespace std;

#define LOADFACTOR 0.8
#define DESIRED_LOADFACTOR 0.6

__device__ unsigned int hashFunction(int key, int tableSize) {
    unsigned int hash = static_cast<unsigned int>(key);

    hash = ((hash >> 16) ^ hash) * 0x45d9f3b;
    hash = ((hash >> 16) ^ hash) * 0x45d9f3b;
    hash = (hash >> 16) ^ hash;

    return hash % tableSize;
}

__global__ void reshapeKernel(int *keys, int *values, int *numItems, int capacity,
                              int *newKeys, int *newValues, int newCapacity) {
    // Calculate global index
    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;

    if (i < capacity) {
        // Calculate the hash
        unsigned int newHash = hashFunction(newKeys[i], newCapacity);

        // Try and insert the key
        while(true) {
            // If the key is -1, insert it
            if (atomicCAS(&newKeys[newHash], -1, keys[i]) == -1) {
                // Insert the value
                atomicCAS(&newValues[newHash], -1, values[i]);

                // Break the loop
                break;
            } else {
                // If the key is not -1, try and insert it in the next position
                newHash = (newHash + 1) % newCapacity;
            }
        }
    }
}

__global__ void insertKernel(int *keys, int *values, int *numItems, int capacity,
                             int *insertKeys, int *insertValues, int numInsertItems) {
    // Calculate global index
  	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;

    // If not out of bounds
    if (i < numInsertItems) {
        // Calculate the hash
        unsigned int insertHash = hashFunction(insertKeys[i], capacity);

        // Try and insert the key
        while(true) {
            // If the key is -1, insert it
            if (atomicCAS(&keys[insertHash], -1, insertKeys[i]) == -1) {
                // Insert the value
                atomicCAS(&values[insertHash], -1, insertValues[i]);

                // Increment the number of items
                atomicAdd(numItems, 1);

                // Break the loop
                break;
            } else {
                // If the key is the same, update the value
                if (keys[insertHash] == insertKeys[i]) {
                    // Update the value
                    values[insertHash] = insertValues[i];

                    // Break the loop
                    break;
                } else {
                    // If the key is not -1, try and insert it in the next position
                    insertHash = (insertHash + 1) % capacity;
                }
            }
        }
    }
}

__global__ void getKernel(int *keys, int *values, int *numItems, int capacity,
                          int *getKeys, int *getValues, int numGetItems) {
    // Calculate global index
    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;

    if (i < numGetItems) {
        // Calculate the hash
        unsigned int getHash = hashFunction(getKeys[i], capacity);

        // Try and get the key
        while(true) {
            // If the key is the same, get the value
            if (keys[getHash] == getKeys[i]) {
                // Get the value
                getValues[i] = values[getHash];

                // Break the loop
                break;
            } else {
                // If the key is not -1, try and get it from the next position
                getHash = (getHash + 1) % capacity;
            }
        }
    }                                   
}

GpuHashTable::GpuHashTable(int size) {
    // Allocate memory for the hashtable
    glbGpuAllocator->_cudaMalloc((void**)&this->keys, size * sizeof(int));
    glbGpuAllocator->_cudaMalloc((void**)&this->values, size * sizeof(int));
    glbGpuAllocator->_cudaMallocManaged((void**)&this->numItems, sizeof(int));

    // Set numItems and capacity
    *this->numItems = 0;
    this->capacity = size;

    // Initialize the hashtable with -1
    cudaMemset(this->keys, -1, size * sizeof(int));
    cudaMemset(this->values, -1, size * sizeof(int));
}

GpuHashTable::~GpuHashTable() {
    // Free the memory allocated for the hashtable
    glbGpuAllocator->_cudaFree(this->keys);
    glbGpuAllocator->_cudaFree(this->values);
    glbGpuAllocator->_cudaFree(this->numItems);

    // Set numItems and capacity to 0
    *this->numItems = 0;
    this->capacity = 0;
}

void GpuHashTable::reshape(int numBucketsReshape) {
    // Allocate memory for the new hashtable
    int *newKeys, *newValues;
    glbGpuAllocator->_cudaMalloc((void**)&newKeys, numBucketsReshape * sizeof(int));
    glbGpuAllocator->_cudaMalloc((void**)&newValues, numBucketsReshape * sizeof(int));

    // Initialize the hashtable with -1
    cudaMemeset(newKeys, -1, numBucketsReshape * sizeof(int));
    cudaMemset(newValues, -1, numBucketsReshape * sizeof(int));

    // Calculate the number of blocks and threads
    const size_t block_size = 256;
  	size_t blocks_no = numBucketsReshape / block_size;

    if (numBucketsReshape % block_size) {
        ++blocks_no;
    }

    // Call the kernel
    reshapeKernel<<<blocks_no, block_size>>>(this->keys, this->values, this->numItems, this->capacity,
                                             newKeys, newValues, numBucketsReshape);

    // Synchronize the threads
    cudaDeviceSynchronize();

    // Free the memory allocated for the old hashtable
    glbGpuAllocator->_cudaFree(this->keys);
    glbGpuAllocator->_cudaFree(this->values);

    // Set the new hashtable
    this->keys = newKeys;
    this->values = newValues;
    this->capacity = numBucketsReshape;
}

bool GpuHashTable::insertBatch(int* keys, int* values, int numKeys) {
    // Other checks
    if (numKeys <= 0) {
        return false;
    }

    // Verify if the hashtable needs to be resized
    float loadFactor = (*numItems + numKeys) / (float) this->capacity;
    if (loadFactor > LOADFACTOR) {
        // Calculate the resize capacity
        int resizeCapacity = (*numItems + numKeys) / DESIRED_LOADFACTOR;

        // Reshape the hashtable
        this->reshape(resizeCapacity);
    }

    // Allocate memory for the keys and values
    int *d_keys, *d_values;
    glbGpuAllocator->_cudaMalloc((void**)&d_keys, numKeys * sizeof(int));
    glbGpuAllocator->_cudaMalloc((void**)&d_values, numKeys * sizeof(int));

    // Copy the keys and values to the device
    cudaMemcpy(d_keys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);

    // Calculate the number of blocks and threads
    const size_t block_size = 256;
  	size_t blocks_no = numKeys / block_size;

	if (numKeys % block_size) {
		++blocks_no;
    }

    // Call the kernel
    insertKernel<<<blocks_no, block_size>>>(this->keys, this->values, this->numItems, this->capacity,
                                            d_keys, d_values, numKeys);

    // Synchronize the threads
    cudaDeviceSynchronize();

    // Free the memory allocated for the keys and values
    glbGpuAllocator->_cudaFree(d_keys);
    glbGpuAllocator->_cudaFree(d_values);

    return true;
}

int* GpuHashTable::getBatch(int* keys, int numKeys) {
    // Allocate memory for the values
    int *values;
    glbGpuAllocator->_cudaMallocManaged((void**)&values, numKeys * sizeof(int));

    // Copy the keys to the device
    int *d_keys;
    glbGpuAllocator->_cudaMalloc((void**)&d_keys, numKeys * sizeof(int));
    cudaMemcpy(d_keys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);

    // Calculate the number of blocks and threads
    const size_t block_size = 256;
  	size_t blocks_no = numKeys / block_size;

    if (numKeys % block_size) {
        ++blocks_no;
    }

    // Call the kernel
    getKernel<<<blocks_no, block_size>>>(this->keys, this->values, this->numItems, this->capacity,
                                         d_keys, values, numKeys);

    // Synchronize the threads
    cudaDeviceSynchronize();

    // Free the memory allocated for the keys
    glbGpuAllocator->_cudaFree(d_keys);

    return values;
}
