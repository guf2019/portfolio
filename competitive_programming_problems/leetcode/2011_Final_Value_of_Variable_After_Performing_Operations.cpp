class Solution {
public:
    int finalValueAfterOperations(vector<string>& operations) {
        int ans = 0;
        for (auto i: operations) {
            if ('+' == i[0] or '+' == i[2])
                ans++;
            else
                ans--;
        }
        return ans;
    }
};