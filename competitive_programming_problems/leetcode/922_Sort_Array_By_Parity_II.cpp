class Solution {
public:
    vector<int> sortArrayByParityII(vector<int>& A) {
        int s = (int)A.size() - 1;
        int even = 1;
        int odd = 0;
        while (true)
        {
            if (odd > s || even > s)
                break;

            if (odd <= s && (A[odd] & 1) == 0)
            {
                odd += 2;
            }

            if (even <= s && (A[even] & 1) == 1)
            {
                even += 2;
            }

            if (odd <= s && even <= s && ((A[odd] & 1) == 1) && ((A[even] & 1) == 0))
            {
                swap(A[odd], A[even]);
                odd += 2;
                even += 2;
            }
        }
        return move(A);
    }
};