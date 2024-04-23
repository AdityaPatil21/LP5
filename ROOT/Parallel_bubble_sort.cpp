#include <iostream>
#include <vector>
#include <omp.h>
using namespace std;
// Function to perform parallel bubble sort using odd-even transposition
void bubble_sort_odd_even(vector<int>& arr) {
	bool isSorted = false;
	while (!isSorted) {
		isSorted = true;// Assume the array is sorted initially
		// Odd phase: compare and swap elements at odd indices
		#pragma omp parallel for
		for (int i = 0; i < arr.size() - 1; i += 2) {
			 // Compare and swap adjacent elements if necessary
			if (arr[i] > arr[i + 1]) {
				swap(arr[i], arr[i + 1]);
				isSorted = false; // Set flag to indicate that array is not sorted
			}
		}
		// Even phase: compare and swap elements at even indices
		#pragma omp parallel for
		for (int i = 1; i < arr.size() - 1; i += 2) {
			if (arr[i] > arr[i + 1]) {
				swap(arr[i], arr[i + 1]);
				isSorted = false;
			}		
		}
	}
}

int main() {
	vector<int> arr = {5, 2, 9, 1, 7, 6, 8, 3, 4};
	double start, end;
	// Measure performance of parallel bubble sort using odd-
	//even transposition
	start = omp_get_wtime(); // Get the start time
	bubble_sort_odd_even(arr);// Perform parallel bubble sort
	end = omp_get_wtime();// Get the end time
	cout << "Parallel bubble sort using odd-even transposition time: " << end - start << endl;
	for(int a:arr){
		cout<<a<<" ";
	}
	cout<<endl;
}
