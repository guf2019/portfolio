class Solution {
public:
    int maximumWealth(vector<vector<int>>& accounts) {
        int ans = 0;
        for (int i = 0; i < accounts.size(); ++i) {
            int customer_sum = 0;
            for (int j = 0; j < accounts[i].size(); ++j) {
                customer_sum += accounts[i][j];
            }
            ans = max(ans, customer_sum);
        }
        return  ans;
    }
};