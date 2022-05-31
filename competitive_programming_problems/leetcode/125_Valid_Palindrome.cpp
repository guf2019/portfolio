class Solution {
public:
    bool isPalindrome(string s) {
        int left = 0, right = s.length() - 1;
        while(left < right)
        {
            char l = tolower(s[left]);
            while(!(isalpha(l) || isdigit(l)) && left < right)
            {
                left++;
                l = tolower(s[left]);
            }

            char r = tolower(s[right]);
            while(!(isalpha(r) || isdigit(r)) && right > left)
            {
                right--;
                r = tolower(s[right]);
            }

            if ( l != r)
            {
                return false;
            }
            left++;
            right--;

        }
        return true;
    }
};