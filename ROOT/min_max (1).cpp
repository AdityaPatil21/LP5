#include <iostream>
#include <vector>
#include <omp.h>
#include <climits>

using namespace std;

// Function to find the minimum value in the array using parallel reduction
void min_reduction(vector<int>& arr) {
  int min_value = INT_MAX;
  // OpenMP directive for parallelizing the loop with reduction(min: min_value)
  #pragma omp parallel for reduction(min: min_value)
  for (int i = 0; i < arr.size(); i++) {
    // Each thread finds the minimum value in its assigned portion of the array
    if (arr[i] < min_value) {
      min_value = arr[i];
    }
  }
  cout << "Minimum value: " << min_value << endl;
}

// Function to find the maximum value in the array using parallel reduction
void max_reduction(vector<int>& arr) {
  int max_value = INT_MIN;
  // OpenMP directive for parallelizing the loop with reduction(max: max_value)
  #pragma omp parallel for reduction(max: max_value)
  for (int i = 0; i < arr.size(); i++) {
    // Each thread finds the maximum value in its assigned portion of the array
    if (arr[i] > max_value) {
      max_value = arr[i];
    }
  }
  cout << "Maximum value: " << max_value << endl;
}

// Function to calculate the sum of all elements in the array using parallel reduction
void sum_reduction(vector<int>& arr) {
  int sum = 0;
  // OpenMP directive for parallelizing the loop with reduction(+: sum)
  #pragma omp parallel for reduction(+: sum)
  for (int i = 0; i < arr.size(); i++) {
    // Each thread calculates the sum of elements in its assigned portion of the array
    sum += arr[i];
  }
  cout << "Sum: " << sum << endl;
}

// Function to calculate the average value of elements in the array using parallel reduction
void average_reduction(vector<int>& arr) {
  int sum = 0;
  // OpenMP directive for parallelizing the loop with reduction(+: sum)
  #pragma omp parallel for reduction(+: sum)
  for (int i = 0; i < arr.size(); i++) {
    // Each thread calculates the sum of elements in its assigned portion of the array
    sum += arr[i];
  }
  // Divide the total sum by the number of elements in the array to find the average
  cout << "Average: " << (double)sum / arr.size() << endl;
}

int main() {
  // Initialize a vector with some values
  vector<int> arr;
  arr.push_back(5);
  arr.push_back(2);
  arr.push_back(9);
  arr.push_back(1);
  arr.push_back(7);
  arr.push_back(6);
  arr.push_back(8);
  arr.push_back(3);
  arr.push_back(4);

  // Call each reduction operation function to perform the respective reduction
  min_reduction(arr);
  max_reduction(arr);
  sum_reduction(arr);
  average_reduction(arr);

  return 0;
}

