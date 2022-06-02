class Solution {
        public:
        int mostWordsFound(vector<string>& sentences) {
            int ans = 0;
            for (int i = 0; i < sentences.size(); ++i) {
                int words = 1;
                for (auto j: sentences[i]) {
                    if (' ' == j)
                        words++;
                }
                ans = max(ans, words);
            }
            return ans;
        }
};