class Solution {
public:
    vector<bool> kidsWithCandies(vector<int>& candies, int extraCandies) {
        vector<bool> ans;
        int max_candies = 0;
        for (int i = 0; i < candies.size(); ++i) {
            max_candies = max(max_candies, candies[i]);
        }
        for (int i = 0; i < candies.size(); ++i) {
            ans.push_back(candies[i] + extraCandies >= max_candies);
        }
        return ans;
    }
};