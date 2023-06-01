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

#define LOADFACTOR 0.75
#define DESIRED_LOADFACTOR 0.55

__device__ unsigned int hashFunction(int key, int tableSize) {
    unsigned int hash = static_cast<unsigned int>(key);

    hash = ((hash >> 16) ^ hash) * 0x45d9f3b;
    hash = ((hash >> 16) ^ hash) * 0x45d9f3b;
    hash = (hash >> 16) ^ hash;

    return hash % tableSize;
}

__global__ void reshapeKernel(int* keys, int* values, int numItems, int capacity,
                              int* newKeys, int* newValues, int newCapacity) {
    // Calculate global index
    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;

    if (i < capacity && keys[i] != -1) {
        // Calculate the hash
        unsigned int reshapeHash = hashFunction(keys[i], newCapacity);

        // Try and insert the key
        while (true) {
            // If the key is -1, insert it
            if (atomicCAS(&newKeys[reshapeHash], -1, keys[i]) == -1) {
                // Insert the value
                newValues[reshapeHash] = values[i];

                // Break the loop
                break;
            } else {
                // If the key is not -1, try and insert it in the next position
                reshapeHash = (reshapeHash + 1) % newCapacity;
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
                values[insertHash] = insertValues[i];

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

__global__ void getKernel(int *keys, int *values, int numItems, int capacity,
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
    cudaError_t ret;
    printf("GpuHashTable::GpuHashTable\n");

    // Allocate memory for the hashtable
    ret = glbGpuAllocator->_cudaMalloc((void**)&this->keys, size * sizeof(int));
    if (ret != cudaSuccess) {
        printf("Error allocating memory for keys: %s\n", cudaGetErrorString(ret));
        exit(1);
    }

    ret = glbGpuAllocator->_cudaMalloc((void**)&this->values, size * sizeof(int));
    if (ret != cudaSuccess) {
        printf("Error allocating memory for values: %s\n", cudaGetErrorString(ret));
        exit(1);
    }

    // Set numItems and capacity
    this->numItems = 0;
    this->capacity = size;

    // Initialize the hashtable with -1
    ret = cudaMemset(this->keys, -1, size * sizeof(int));
    if (ret != cudaSuccess) {
        printf("Error initializing keys: %s\n", cudaGetErrorString(ret));
        exit(1);
    }

    ret = cudaMemset(this->values, -1, size * sizeof(int));
    if (ret != cudaSuccess) {
        printf("Error initializing values: %s\n", cudaGetErrorString(ret));
        exit(1);
    }

    printf("Finished GpuHashTable::GpuHashTable\n");
}

GpuHashTable::~GpuHashTable() {
    cudaError_t ret;

    printf("GpuHashTable::~GpuHashTable\n");

    // Free the memory allocated for the hashtable
    ret = glbGpuAllocator->_cudaFree(this->keys);
    if (ret != cudaSuccess) {
        printf("Error freeing memory for keys: %s\n", cudaGetErrorString(ret));
        exit(1);
    }

    ret = glbGpuAllocator->_cudaFree(this->values);
    if (ret != cudaSuccess) {
        printf("Error freeing memory for values: %s\n", cudaGetErrorString(ret));
        exit(1);
    }

    // Set numItems and capacity to 0
    this->numItems = 0;
    this->capacity = 0;

    printf("Finished GpuHashTable::~GpuHashTable\n");
}

void GpuHashTable::reshape(int numBucketsReshape) {
    cudaError_t ret;
    printf("GpuHashTable::reshape\n");

    // Allocate memory for the new hashtable
    int *newKeys, *newValues;
    ret = glbGpuAllocator->_cudaMalloc((void**)&newKeys, numBucketsReshape * sizeof(int));
    if (ret != cudaSuccess) {
        printf("Error allocating memory for new keys: %s\n", cudaGetErrorString(ret));
        exit(1);
    }

    ret = glbGpuAllocator->_cudaMalloc((void**)&newValues, numBucketsReshape * sizeof(int));
    if (ret != cudaSuccess) {
        printf("Error allocating memory for new values: %s\n", cudaGetErrorString(ret));
        exit(1);
    }

    // Initialize the hashtable with -1
    ret = cudaMemset(newKeys, -1, numBucketsReshape * sizeof(int));
    if (ret != cudaSuccess) {
        printf("Error initializing new keys: %s\n", cudaGetErrorString(ret));
        exit(1);
    }

    ret = cudaMemset(newValues, -1, numBucketsReshape * sizeof(int));
    if (ret != cudaSuccess) {
        printf("Error initializing new values: %s\n", cudaGetErrorString(ret));
        exit(1);
    }

    // Calculate the number of blocks and threads
    const size_t block_size = 256;
  	size_t blocks_no = this->capacity / block_size;

    if (this->capacity % block_size) {
        ++blocks_no;
    }

    // Call the kernel
    reshapeKernel<<<blocks_no, block_size>>>(this->keys, this->values, this->numItems, this->capacity,
                                             newKeys, newValues, numBucketsReshape);

    // Synchronize the threads
    ret = cudaDeviceSynchronize();
    if (ret != cudaSuccess) {
        printf("Error synchronizing threads: %s\n", cudaGetErrorString(ret));
        exit(1);
    }

    // Free the memory allocated for the old hashtable
    ret = glbGpuAllocator->_cudaFree(this->keys);
    if (ret != cudaSuccess) {
        printf("Error freeing memory for old keys: %s\n", cudaGetErrorString(ret));
        exit(1);
    }

    ret = glbGpuAllocator->_cudaFree(this->values);
    if (ret != cudaSuccess) {
        printf("Error freeing memory for old values: %s\n", cudaGetErrorString(ret));
        exit(1);
    }

    // Set the new hashtable
    this->keys = newKeys;
    this->values = newValues;
    this->capacity = numBucketsReshape;

    printf("Finished GpuHashTable::reshape\n");
}

bool GpuHashTable::insertBatch(int* keys, int* values, int numKeys) {
    cudaError_t ret;
    printf("GpuHashTable::insertBatch\n");

    // Other checks
    if (numKeys <= 0) {
        return false;
    }

    if (keys != NULL || values != NULL) {
        return false;
    }

    // Verify if the hashtable needs to be resized
    float loadFactor = 1.0 * (this->numItems + numKeys) / this->capacity;
    printf("Load factor: %f\n", loadFactor);
    if (loadFactor >= LOADFACTOR) {
        // Calculate the resize capacity
        int resizeCapacity = (this->numItems + numKeys) / DESIRED_LOADFACTOR;

        // Reshape the hashtable
        this->reshape(resizeCapacity);
    }

    // Allocate memory for the keys and values
    int *d_keys, *d_values;
    ret = glbGpuAllocator->_cudaMalloc((void**)&d_keys, numKeys * sizeof(int));
    if (ret != cudaSuccess) {
        printf("Error allocating memory for keys: %s\n", cudaGetErrorString(ret));
        exit(1);
    }

    ret = glbGpuAllocator->_cudaMalloc((void**)&d_values, numKeys * sizeof(int));
    if (ret != cudaSuccess) {
        printf("Error allocating memory for values: %s\n", cudaGetErrorString(ret));
        exit(1);
    }

    // Copy the keys and values to the device
    ret = cudaMemcpy(d_keys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
    if (ret != cudaSuccess) {
        printf("Error copying keys to device: %s\n", cudaGetErrorString(ret));
        exit(1);
    }

    ret = cudaMemcpy(d_values, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);
    if (ret != cudaSuccess) {
        printf("Error copying values to device: %s\n", cudaGetErrorString(ret));
        exit(1);
    }

    // Calculate the number of blocks and threads
    const size_t block_size = 256;
  	size_t blocks_no = numKeys / block_size;

	if (numKeys % block_size) {
		++blocks_no;
    }

    int *numAddedItems;
    ret = glbGpuAllocator->_cudaMallocManaged((void**)&numAddedItems, sizeof(int));
    if (ret != cudaSuccess) {
        printf("Error allocating memory for numAddedItems, error: %s\n", cudaGetErrorString(ret));
        exit(1);
    }

    *numAddedItems = 0;

    // Call the kernel
    insertKernel<<<blocks_no, block_size>>>(this->keys, this->values, numAddedItems, this->capacity,
                                            d_keys, d_values, numKeys);

    printf("Finished Kernel\n");

    // Add the keys - numAddedItems to the numItems
    this->numItems += *numAddedItems;

    // Synchronize the threads
    printf("Started Synchronizing\n");

    ret = cudaDeviceSynchronize();
    if (ret != cudaSuccess) {
        printf("Error synchronizing threads: %s\n", cudaGetErrorString(ret));
        exit(1);
    }

    printf("Finished Synchronizing\n");

    // Free the memory allocated for the keys and values
    ret = glbGpuAllocator->_cudaFree(d_keys);
    if (ret != cudaSuccess) {
        printf("Error freeing memory for keys: %s\n", cudaGetErrorString(ret));
        exit(1);
    }
    
    ret = glbGpuAllocator->_cudaFree(d_values);
    if (ret != cudaSuccess) {
        printf("Error freeing memory for values: %s\n", cudaGetErrorString(ret));
        exit(1);
    }

    ret = glbGpuAllocator->_cudaFree(numAddedItems);
    if (ret != cudaSuccess) {
        printf("Error freeing memory for numAddedItems: %s\n", cudaGetErrorString(ret));
        exit(1);
    }

    printf("Finished GpuHashTable::insertBatch\n");

    return true;
}

int* GpuHashTable::getBatch(int* keys, int numKeys) {
    cudaError_t ret;
    printf("GpuHashTable::getBatch\n");

    // Allocate memory for the values
    int *values;
    ret = glbGpuAllocator->_cudaMallocManaged((void**)&values, numKeys * sizeof(int));
    if (ret != cudaSuccess) {
        printf("Error allocating memory for values: %s\n", cudaGetErrorString(ret));
        exit(1);
    }

    // Copy the keys to the device
    int *d_keys;
    ret = glbGpuAllocator->_cudaMalloc((void**)&d_keys, numKeys * sizeof(int));
    if (ret != cudaSuccess) {
        printf("Error allocating memory for keys: %s\n", cudaGetErrorString(ret));
        exit(1);
    }

    ret = cudaMemcpy(d_keys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
    if (ret != cudaSuccess) {
        printf("Error copying keys to device: %s\n", cudaGetErrorString(ret));
        exit(1);
    }

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
    ret = cudaDeviceSynchronize();
    if (ret != cudaSuccess) {
        printf("Error synchronizing threads: %s\n", cudaGetErrorString(ret));
        exit(1);
    }

    // Free the memory allocated for the keys
    ret = glbGpuAllocator->_cudaFree(d_keys);
    if (ret != cudaSuccess) {
        printf("Error freeing memory for keys: %s\n", cudaGetErrorString(ret));
        exit(1);
    }

    printf("Finished GpuHashTable::getBatch\n");

    return values;
}
