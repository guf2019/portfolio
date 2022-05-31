class Solution {
public:
    vector<int> sortArrayByParity(vector<int>& A) {
        short int l = 0, r = A.size() - 1;
        while (l < r)
        {
            if ((A[l]&1) == 0)
            {
                l++;
                continue;
            }
            if ((A[r]&1) != 0)
            {
                r--;
                continue;
            }

            swap(A[l], A[r]);
        }
        return A;
    }
};
