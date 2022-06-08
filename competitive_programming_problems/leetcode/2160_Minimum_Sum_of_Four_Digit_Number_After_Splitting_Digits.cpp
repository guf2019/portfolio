class Solution {
public:
    int minimumSum(int num) {
        int first = num % 10;
        int second = (num / 10) % 10;
        int third = (num / 100) % 10;
        int fourth = (num / 1000) % 10;
        vector<int> digits = {first, second, third, fourth};
        sort(begin(digits), end(digits));
        return (digits[0] * 10 + digits[2]) + (digits[1] * 10 + digits[3]);
    }
};