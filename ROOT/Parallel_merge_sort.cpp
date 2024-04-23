#include <iostream>
#include <vector>
#include <omp.h>

using namespace std;

// Function to merge two sorted subarrays into one sorted array
void merge(vector<int>& arr, int l, int m, int r) {
    // Calculate sizes of the two subarrays
    int n1 = m - l + 1;
    int n2 = r - m;

    // Create temporary arrays to store the elements of the two subarrays
    vector<int> L(n1), R(n2);

    // Copy data to temporary arrays L[] and R[]
    for (int i = 0; i < n1; i++) {
        L[i] = arr[l + i];
    }
    for (int j = 0; j < n2; j++) {
        R[j] = arr[m + 1 + j];
    }

    // Merge the two sorted subarrays into the original array
    int i = 0, j = 0, k = l;
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k++] = L[i++];
        } else {
            arr[k++] = R[j++];
        }
    }

    // Copy the remaining elements of L[], if any
    while (i < n1) {
        arr[k++] = L[i++];
    }

    // Copy the remaining elements of R[], if any
    while (j < n2) {
        arr[k++] = R[j++];
    }
}

// Function to perform merge sort recursively
void merge_sort(vector<int>& arr, int l, int r) {
    if (l < r) {
        // Calculate the middle index
        int m = l + (r - l) / 2;

        // Recursively sort the first and second halves
        merge_sort(arr, l, m);
        merge_sort(arr, m + 1, r);

        // Merge the sorted halves
        merge(arr, l, m, r);
    }
}

// Function to perform parallel merge sort using OpenMP tasks
void parallel_merge_sort(vector<int>& arr) {
    // Create parallel region
#pragma omp parallel
    {
        // Execute a single thread to start the parallel merge sort
#pragma omp single
        merge_sort(arr, 0, arr.size() - 1);
    }
}

int main() {
    // Initialize the input array
    vector<int> arr = {5, 2, 9, 1, 7, 6, 8, 3, 4};
    double start, end;

    // Measure the performance of sequential merge sort
    start = omp_get_wtime();
    merge_sort(arr, 0, arr.size() - 1);
    end = omp_get_wtime();
    cout << "Sequential merge sort time: " << end - start << " seconds" << endl;

    // Measure the performance of parallel merge sort
    arr = {5, 2, 9, 1, 7, 6, 8, 3, 4};
    start = omp_get_wtime();
    parallel_merge_sort(arr);
    end = omp_get_wtime();
    cout << "Parallel merge sort time: " << end - start << " seconds" << endl;

    return 0;
}

