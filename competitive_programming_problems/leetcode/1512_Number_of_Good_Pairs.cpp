class Solution {
public:
    int numIdenticalPairs(vector<int>& nums) {

        int buff[101] ={0};
        int res=0;

        for(int i=0; i < nums.size(); i++){
            res += buff[nums[i]];
            ++buff[nums[i]];
        }

        return res;
    }
};