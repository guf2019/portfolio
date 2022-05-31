class Solution {
public:
    int peakIndexInMountainArray(vector<int>& arr) {
        int middle = (int)arr.size() / 2, start = 0, end = (int)arr.size();

        while(!(arr[middle - 1] < arr[middle] && arr[middle] > arr[middle + 1]))
        {
            if(arr[middle] < arr[middle + 1])
            {
                start = middle;
                middle += (end - start) / 2;
            }
            else if(arr[middle] > arr[middle + 1])
            {
                end = middle;
                middle -= (end - start) / 2;
            }
        }
        return middle;
    }
};