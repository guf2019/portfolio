class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        map<int, int> hashmap;
        vector<int> ans;
        for (int i = 0; i < nums.size(); ++i) {
            int required = target - nums[i];
            if (hashmap.find(required) != hashmap.end()){
                ans.push_back(i);
                ans.push_back(hashmap[required]);
                return ans;
            }
            hashmap[nums[i]] = i;
        }
        return ans;
    }
};