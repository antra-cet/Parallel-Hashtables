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

struct GpuHashTable {
	int numBuckets;
	int numElements;
	int* keys;
	int* values;
};

using namespace std;

__device__ unsigned int hashFunction(int key, int tableSize) {
    unsigned int hash = static_cast<unsigned int>(key);

    hash = ((hash >> 16) ^ hash) * 0x45d9f3b;
    hash = ((hash >> 16) ^ hash) * 0x45d9f3b;
    hash = (hash >> 16) ^ hash;

    return hash % tableSize;
}

__device__ int* getKey(const int* keys, const int* values, int numKeys, int keyToFind) {
    unsigned int hash = hashFunction(keyToFind, numKeys);
  
    for (int i = 0; i < numKeys; ++i) {
        int index = (hash + i) % numKeys;
        if (keys[index] == keyToFind) {
            return &values[index];
        }
    }
  
    return nullptr;
}

__device__ int getValue(const int* keys, const int* values, int numKeys, int keyToFind) {
    unsigned int hash = hashFunction(keyToFind, numKeys);
  
    for (int i = 0; i < numKeys; ++i) {
        int index = (hash + i) % numKeys;
        if (keys[index] == keyToFind) {
            return values[index];
        }
    }
  
    return -1;
}


/*
Allocate CUDA memory only through glbGpuAllocator
cudaMalloc -> glbGpuAllocator->_cudaMalloc
cudaMallocManaged -> glbGpuAllocator->_cudaMallocManaged
cudaFree -> glbGpuAllocator->_cudaFree
*/

/**
 * Function constructor GpuHashTable
 * Performs init
 * Example on using wrapper allocators _cudaMalloc and _cudaFree
 */
GpuHashTable::GpuHashTable(int size) {
	// Allocate memory for the hashtable
	glbGpuAllocator->_cudaMalloc((void**)&this->keys, size * sizeof(int));
	glbGpuAllocator->_cudaMalloc((void**)&this->values, size * sizeof(int));
	this->numBuckets = size;
	this->numElements = 0;

	// Initialize the hashtable with -1
	for (int i = 0; i < size; ++i) {
		this->keys[i] = -1;
		this->values[i] = -1;
	}
}

/**
 * Function desctructor GpuHashTable
 */
GpuHashTable::~GpuHashTable() {
	// Free the memory allocated for the hashtable
	glbGpuAllocator->_cudaFree(this->keys);
	glbGpuAllocator->_cudaFree(this->values);

	this->numBuckets = 0;
	this->numElements = 0;
}

/**
 * Function reshape
 * Performs resize of the hashtable based on load factor
 */
void GpuHashTable::reshape(int numBucketsReshape) {
	// Allocate memory for the new hashtable
	int* newKeys;
	int* newValues;
	glbGpuAllocator->_cudaMalloc((void**)&newKeys, numBucketsReshape * sizeof(int));
	glbGpuAllocator->_cudaMalloc((void**)&newValues, numBucketsReshape * sizeof(int));

	// Initialize the new hashtable with -1
	for (int i = 0; i < numBucketsReshape; ++i) {
		newKeys[i] = -1;
		newValues[i] = -1;
	}

	// Copy the old hashtable into the new one
	for (int i = 0; i < this->numBuckets; ++i) {
		if (this->keys[i] != -1) {
			int* value = getKey(newKeys, newValues, numBucketsReshape, this->keys[i]);
			*value = this->values[i];
		}
	}

	// Free the memory allocated for the old hashtable
	glbGpuAllocator->_cudaFree(this->keys);
	glbGpuAllocator->_cudaFree(this->values);

	// Update the hashtable
	this->keys = newKeys;
	this->values = newValues;
	this->numBuckets = numBucketsReshape;
}

/**
 * Function insertBatch
 * Inserts a batch of key:value, using GPU and wrapper allocators
 */
bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	// Check if the hashtable needs to be resized
	if (this->numElements + numKeys > this->numBuckets * 0.8) {
		this->reshape(this->numBuckets * 2);
	}

	// Insert the keys and values into the hashtable
	for (int i = 0; i < numKeys; ++i) {
		int* value = getKey(this->keys, this->values, this->numBuckets, keys[i]);
		if (value == nullptr) {
			int index = hashFunction(keys[i], this->numBuckets);
			while (this->keys[index] != -1) {
				index = (index + 1) % this->numBuckets;
			}
			this->keys[index] = keys[i];
			this->values[index] = values[i];
			this->numElements++;
		}
		else {
			*value = values[i];
		}
	}

	return true;
}

/**
 * Function getBatch
 * Gets a batch of key:value, using GPU
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {
	// Allocate memory for the values
	int* values;
	glbGpuAllocator->_cudaMallocManaged((void**)&values, numKeys * sizeof(int));

	// Get the values from the hashtable
	for (int i = 0; i < numKeys; ++i) {
		values[i] = getValue(this->keys, this->values, this->numBuckets, keys[i]);
	}

	return values;
}
