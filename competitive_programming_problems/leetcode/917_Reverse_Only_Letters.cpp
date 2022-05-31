class Solution {
public:
    string reverseOnlyLetters(string S) {
        int l = 0;
        int r = S.length() - 1;

        while (l < r)
        {
            if ((0 < (int)S[l]) && (65 > (int)S[l]))
            {
                l++;
                continue;
            }

            if ((90 < (int)S[l])  && (97 > (int)S[l]))
            {
                l++;
                continue;
            }

            if ((0 < (int)S[r]) && (65 > (int)S[r]))
            {
                r--;
                continue;
            }

            if ((90 < (int)S[r])  && (97 > (int)S[r]))
            {
                r--;
                continue;
            }

            swap(S[l], S[r]);
            l++;
            r--;
        }

        return S;
    }
};