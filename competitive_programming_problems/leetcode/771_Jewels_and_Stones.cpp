class Solution {
public:
    int numJewelsInStones(string jewels, string stones) {
        map<char, bool> isJewel;
        int ans = 0;
        for (auto jewel: jewels) {
            isJewel[jewel] = true;
        }
        for (auto stone: stones) {
            if (isJewel.find(stone) != isJewel.end())
                ans = isJewel[stone] ? ans + 1 : ans;
        }
        return ans;
    }
};