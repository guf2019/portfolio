class SubrectangleQueries {
public:
    vector<vector<int>> rect, updates;
    SubrectangleQueries(vector<vector<int>>& rectangle) {
        swap(rect, rectangle);
    }

    void updateSubrectangle(int row1, int col1, int row2, int col2, int newValue) {
        updates.push_back({row1, col1, row2, col2, newValue});
    }

    int getValue(int row, int col) {
        for (int i = updates.size() - 1; i >= 0; i--) {
            if (row >= updates[i][0] && row <= updates[i][2] && col >= updates[i][1] && col <= updates[i][3])
                return updates[i][4];
        }
        return rect[row][col];
    }
};

/**
 * Your SubrectangleQueries object will be instantiated and called as such:
 * SubrectangleQueries* obj = new SubrectangleQueries(rectangle);
 * obj->updateSubrectangle(row1,col1,row2,col2,newValue);
 * int param_2 = obj->getValue(row,col);
 */