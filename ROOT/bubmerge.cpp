#include <iostream>
#include <vector>
#include <omp.h>
using namespace std;

// Function to perform parallel bubble sort using odd-even transposition
void bubble_sort_odd_even(vector<int>& arr) {
    bool isSorted = false;
    // Continue sorting until the array is sorted
    while (!isSorted) {
        isSorted = true;
        // Parallel loop for even indexed elements
        #pragma omp parallel for shared(arr, isSorted)
        for (int i = 0; i < arr.size() - 1; i += 2) {
            if (arr[i] > arr[i + 1]) {
                swap(arr[i], arr[i + 1]);
                isSorted = false;
            }
        }
        // Parallel loop for odd indexed elements
        #pragma omp parallel for shared(arr, isSorted)
        for (int i = 1; i < arr.size() - 1; i += 2) {
            if (arr[i] > arr[i + 1]) {
                swap(arr[i], arr[i + 1]);
                isSorted = false;
            }        
        }
    }
}

// Function to merge two subarrays
void merge(vector<int>& arr, int l, int m, int r) {
    int i, j, k;
    int n1 = m - l + 1;
    int n2 = r - m;
    vector<int> L(n1), R(n2);
    // Copy data to temporary arrays L[] and R[]
    for (i = 0; i < n1; i++) {
        L[i] = arr[l + i];
    }
    for (j = 0; j < n2; j++) {
        R[j] = arr[m + 1 + j];
    }
    i = 0;
    j = 0;
    k = l;
    // Merge the temporary arrays back into arr[l..r]
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

// Function to perform merge sort on a subarray
void merge_sort(vector<int>& arr, int l, int r) {
    if (l < r) {
        int m = l + (r - l) / 2;
        // Parallelize the sorting of left and right subarrays
        #pragma omp parallel sections
        {
            #pragma omp section
            merge_sort(arr, l, m); // Sort the left subarray
            #pragma omp section
            merge_sort(arr, m + 1, r); // Sort the right subarray
        }
        merge(arr, l, m, r); // Merge the sorted subarrays
    }
}

int main() {
    vector<int> arr1 = {5, 2, 9, 1, 7, 6, 8, 3, 4};
    vector<int> arr2 = {5, 2, 9, 1, 7, 6, 8, 3, 4};
    double start, end;

    // Measure performance of parallel bubble sort using odd-even transposition
    start = omp_get_wtime();
    bubble_sort_odd_even(arr1);
    end = omp_get_wtime();
    cout << "Parallel bubble sort using odd-even transposition time: " << end - start << endl;
    cout << "Sorted array (bubble sort): ";
    for(int a : arr1){
        cout << a << " ";
    }
    cout << endl << endl;

    // Measure performance of parallel merge sort
    start = omp_get_wtime();
    merge_sort(arr2, 0, arr2.size() - 1);
    end = omp_get_wtime();
    cout << "Parallel merge sort time: " << end - start << endl;
    cout << "Sorted array (merge sort): ";
    for(int a : arr2){
        cout << a << " ";
    }
    cout << endl;

    return 0;
}

