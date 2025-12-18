// ArcadiaEngine.cpp - STUDENT TEMPLATE
// TODO: Implement all the functions below according to the assignment requirements

#define INF INT_MAX

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <map>
#include <numeric>
#include <queue>
#include <set>
#include <string>
#include <vector>
#include <functional>

#include "ArcadiaEngine.h"

using namespace std;

// =========================================================
// PART A: DATA STRUCTURES (Concrete Implementations)
// =========================================================

// =========================================================
// 1. PlayerTable (Double Hashing)
// =========================================================
class ConcretePlayerTable : public PlayerTable
{
private:
    vector<int> keys;      // Stores player IDs
    vector<string> values; // Stores associated names
    int capacity;          // Hash table capacity
    int count;             // Number of stored elements

    // Primary hash function using multiplication method
    int h1(int key) const
    {
        return key % capacity;
    }

    // Secondary hash for double hashing (must be non-zero)
    int h2(int key) const
    {
        return 7 - (key % 7); // Ensures step is non-zero
    }

public:
    ConcretePlayerTable()
    {
        // Arcadian Method initialization
        capacity = 10007;
        keys.assign(capacity, -1); // -1 indicates empty slot
        values.assign(capacity, "");
        count = 0;
    }

    void insert(int playerID, string name) override
    {
        // Arcadian Method applied here (optimized for Olympian performance)
        int index = h1(playerID);
        int step = h2(playerID);

        for (int i = 0; i < capacity; i++)
        {
            int pos = (index + i * step) % capacity;
            if (keys[pos] == -1 || keys[pos] == playerID)
            {
                keys[pos] = playerID;
                values[pos] = name;
                return;
            }
        }
        cerr << "PlayerTable overflow — cannot insert.\n";
    }

    string search(int playerID) override
    {
        int index = h1(playerID);
        int step = h2(playerID);

        for (int i = 0; i < capacity; i++)
        {
            int pos = (index + i * step) % capacity;
            if (keys[pos] == -1)
                return ""; // not found
            if (keys[pos] == playerID)
                return values[pos];
        }
        return "";
    }
};

// =========================================================
// 2. Leaderboard (Skip List)
// =========================================================
class ConcreteLeaderboard : public Leaderboard
{
private:
    // Node structure for skip list
    struct Node
    {
        int playerID;
        int score;
        vector<Node *> forward; // Multiple levels
        Node(int id, int sc, int level) : playerID(id), score(sc), forward(level, nullptr) {}
    };

    int maxLevel;      // Maximum levels in skip list
    float probability; // Probability for level increase
    Node *head;        // Sentinel head node
    int currentLevel;  // Current top level

    // Random level generator
    int randomLevel()
    {
        int lvl = 0;
        while (((float)rand() / RAND_MAX) < probability && lvl < maxLevel - 1)
            lvl++;
        return lvl;
    }

public:
    ConcreteLeaderboard()
    {
        maxLevel = 16;
        probability = 0.5f;
        currentLevel = 0;
        head = new Node(-1, INT_MAX, maxLevel); // sentinel
    }

    void addScore(int playerID, int score) override
    {
        // Arcadian Method applied here (optimized for Olympian performance)
        vector<Node *> predecessors(maxLevel, nullptr);
        Node *current = head;

        // Find insertion points at each level
        for (int i = currentLevel; i >= 0; i--)
        {
            while (current->forward[i] &&
                   (current->forward[i]->score > score ||
                    (current->forward[i]->score == score && current->forward[i]->playerID < playerID)))
            {
                current = current->forward[i];
            }
            predecessors[i] = current;
        }

        current = current->forward[0];

        // If player exists, remove first
        if (current && current->playerID == playerID)
            removePlayer(playerID);

        int lvl = randomLevel();
        if (lvl > currentLevel)
        {
            for (int i = currentLevel + 1; i <= lvl; i++)
                predecessors[i] = head;
            currentLevel = lvl;
        }

        Node *newNode = new Node(playerID, score, lvl + 1);
        for (int i = 0; i <= lvl; i++)
        {
            newNode->forward[i] = predecessors[i]->forward[i];
            predecessors[i]->forward[i] = newNode;
        }
    }

    void removePlayer(int playerID) override
    {
        vector<Node *> predecessors(maxLevel, nullptr);
        Node *current = head;

        for (int i = currentLevel; i >= 0; i--)
        {
            while (current->forward[i] && current->forward[i]->playerID < playerID)
                current = current->forward[i];
            predecessors[i] = current;
        }

        current = current->forward[0];
        if (current && current->playerID == playerID)
        {
            for (int i = 0; i <= currentLevel; i++)
            {
                if (predecessors[i]->forward[i] != current)
                    break;
                predecessors[i]->forward[i] = current->forward[i];
            }
            delete current;

            // Adjust top level if empty
            while (currentLevel > 0 && head->forward[currentLevel] == nullptr)
                currentLevel--;
        }
    }

    vector<int> getTopN(int n) override
    {
        vector<int> result;
        Node *current = head->forward[0];
        while (current && (int)result.size() < n)
        {
            result.push_back(current->playerID);
            current = current->forward[0];
        }
        return result;
    }

    int getScore(int playerID)
    {
        Node *current = head;
        for (int i = currentLevel; i >= 0; i--)
        {
            while (current->forward[i] && current->forward[i]->playerID < playerID)
                current = current->forward[i];
        }
        current = current->forward[0];
        if (current && current->playerID == playerID)
            return current->score;
        return -1; // not found
    }
};

// =========================================================
// 3. AuctionTree (Red-Black Tree)
// =========================================================
class ConcreteAuctionTree : public AuctionTree
{
private:
    enum Color
    {
        RED,
        BLACK
    };

    struct Node
    {
        int itemID;
        int price;
        Color color;
        Node *left;
        Node *right;
        Node *parent;
        Node(int id = -1, int p = 0)
            : itemID(id), price(p), color(RED), left(nullptr), right(nullptr), parent(nullptr) {}
    };

    Node *root;
    Node *nil_; // sentinel node

    void rotateLeft(Node *x)
    {
        Node *y = x->right;
        x->right = y->left;
        if (y->left != nil_)
            y->left->parent = x;
        y->parent = x->parent;
        if (x->parent == nil_)
            root = y;
        else if (x == x->parent->left)
            x->parent->left = y;
        else
            x->parent->right = y;
        y->left = x;
        x->parent = y;
    }

    void rotateRight(Node *y)
    {
        Node *x = y->left;
        y->left = x->right;
        if (x->right != nil_)
            x->right->parent = y;
        x->parent = y->parent;
        if (y->parent == nil_)
            root = x;
        else if (y == y->parent->left)
            y->parent->left = x;
        else
            y->parent->right = x;
        x->right = y;
        y->parent = x;
    }

    void fixInsert(Node *z)
    {
        while (z->parent->color == RED)
        {
            if (z->parent == z->parent->parent->left)
            {
                Node *y = z->parent->parent->right;
                if (y->color == RED)
                {
                    z->parent->color = BLACK;
                    y->color = BLACK;
                    z->parent->parent->color = RED;
                    z = z->parent->parent;
                }
                else
                {
                    if (z == z->parent->right)
                    {
                        z = z->parent;
                        rotateLeft(z);
                    }
                    z->parent->color = BLACK;
                    z->parent->parent->color = RED;
                    rotateRight(z->parent->parent);
                }
            }
            else
            {
                Node *y = z->parent->parent->left;
                if (y->color == RED)
                {
                    z->parent->color = BLACK;
                    y->color = BLACK;
                    z->parent->parent->color = RED;
                    z = z->parent->parent;
                }
                else
                {
                    if (z == z->parent->left)
                    {
                        z = z->parent;
                        rotateRight(z);
                    }
                    z->parent->color = BLACK;
                    z->parent->parent->color = RED;
                    rotateLeft(z->parent->parent);
                }
            }
        }
        root->color = BLACK;
    }

    Node *treeMinimum(Node *x)
    {
        while (x->left != nil_)
            x = x->left;
        return x;
    }

    void transplant(Node *u, Node *v)
    {
        if (u->parent == nil_)
            root = v;
        else if (u == u->parent->left)
            u->parent->left = v;
        else
            u->parent->right = v;
        v->parent = u->parent;
    }

    void deleteFixup(Node *x)
    {
        while (x != root && x->color == BLACK)
        {
            if (x == x->parent->left)
            {
                Node *w = x->parent->right;
                if (w->color == RED)
                {
                    w->color = BLACK;
                    x->parent->color = RED;
                    rotateLeft(x->parent);
                    w = x->parent->right;
                }
                if (w->left->color == BLACK && w->right->color == BLACK)
                {
                    w->color = RED;
                    x = x->parent;
                }
                else
                {
                    if (w->right->color == BLACK)
                    {
                        w->left->color = BLACK;
                        w->color = RED;
                        rotateRight(w);
                        w = x->parent->right;
                    }
                    w->color = x->parent->color;
                    x->parent->color = BLACK;
                    w->right->color = BLACK;
                    rotateLeft(x->parent);
                    x = root;
                }
            }
            else
            {
                Node *w = x->parent->left;
                if (w->color == RED)
                {
                    w->color = BLACK;
                    x->parent->color = RED;
                    rotateRight(x->parent);
                    w = x->parent->left;
                }
                if (w->right->color == BLACK && w->left->color == BLACK)
                {
                    w->color = RED;
                    x = x->parent;
                }
                else
                {
                    if (w->left->color == BLACK)
                    {
                        w->right->color = BLACK;
                        w->color = RED;
                        rotateLeft(w);
                        w = x->parent->left;
                    }
                    w->color = x->parent->color;
                    x->parent->color = BLACK;
                    w->left->color = BLACK;
                    rotateRight(x->parent);
                    x = root;
                }
            }
        }
        x->color = BLACK;
    }

    void deleteNode(Node *z)
    {
        Node *y = z;
        Color yOriginalColor = y->color;
        Node *x;

        if (z->left == nil_)
        {
            x = z->right;
            transplant(z, z->right);
        }
        else if (z->right == nil_)
        {
            x = z->left;
            transplant(z, z->left);
        }
        else
        {
            y = treeMinimum(z->right);
            yOriginalColor = y->color;
            x = y->right;
            if (y->parent == z)
                x->parent = y;
            else
            {
                transplant(y, y->right);
                y->right = z->right;
                y->right->parent = y;
            }
            transplant(z, y);
            y->left = z->left;
            y->left->parent = y;
            y->color = z->color;
        }
        delete z;
        if (yOriginalColor == BLACK)
            deleteFixup(x);
    }

    Node *findByItemID(Node *node, int itemID)
    {
        if (node == nil_ || node == nullptr)
            return nil_;
        if (node->itemID == itemID)
            return node;
        Node *leftRes = findByItemID(node->left, itemID);
        if (leftRes != nil_)
            return leftRes;
        return findByItemID(node->right, itemID);
    }

public:
    ConcreteAuctionTree()
    {
        nil_ = new Node(-1, 0);
        nil_->color = BLACK;
        nil_->left = nil_->right = nil_->parent = nil_;
        root = nil_;
    }

    ~ConcreteAuctionTree()
    {
        function<void(Node *)> dfs = [&](Node *n)
        {
            if (n == nil_)
                return;
            dfs(n->left);
            dfs(n->right);
            delete n;
        };
        dfs(root);
        delete nil_;
    }

    void insertItem(int itemID, int price) override
    {
        Node *z = new Node(itemID, price);
        z->left = z->right = z->parent = nil_;
        Node *y = nil_;
        Node *x = root;

        while (x != nil_)
        {
            y = x;
            if (z->price < x->price || (z->price == x->price && z->itemID < x->itemID))
                x = x->left;
            else
                x = x->right;
        }

        z->parent = y;
        if (y == nil_)
            root = z;
        else if (z->price < y->price || (z->price == y->price && z->itemID < y->itemID))
            y->left = z;
        else
            y->right = z;

        z->left = z->right = nil_;
        z->color = RED;
        fixInsert(z);
    }

    void deleteItem(int itemID) override
    {
        Node *z = findByItemID(root, itemID);
        if (z == nil_)
            return;
        deleteNode(z);
    }

    void debug_inorder()
    {
        function<void(Node *)> inorder = [&](Node *n)
        {
            if (n == nil_)
                return;
            inorder(n->left);
            cout << "[" << n->price << ":" << n->itemID << ":"
                 << (n->color == RED ? "R" : "B") << "] ";
            inorder(n->right);
        };
        inorder(root);
        cout << "\n";
    }
};

// =========================================================
// PART B: INVENTORY SYSTEM (Dynamic Programming)
// =========================================================

int InventorySystem::optimizeLootSplit(int n, vector<int> &coins)
{
    int total = accumulate(coins.begin(), coins.end(), 0);
    int target = total / 2;
    vector<char> dp(target + 1, false);
    dp[0] = true;
    for (int coin : coins)
    {
        for (int s = target; s >= coin; --s)
        {
            if (dp[s - coin])
                dp[s] = true;
        }
    }
    for (int s = target; s >= 0; --s)
    {
        if (dp[s])
        {
            return total - 2 * s; // minimal achievable difference
        }
    }
    return total;
}
int InventorySystem::maximizeCarryValue(int capacity, vector<pair<int, int>> &items)
{
    // TODO: Implement 0/1 Knapsack using DP
    vector<int> dp(capacity + 1, 0);

    // items = {weight, value} pairs
    for (const auto &item : items)
    {
        int weight = item.first;
        int value = item.second;

        for (int w = capacity; w >= weight; --w)
        {
            dp[w] = max(dp[w], dp[w - weight] + value);
        }
    }

    // Return maximum value achievable within capacity
    return dp[capacity];
}

long long InventorySystem::countStringPossibilities(string s)
{
    const int MOD = 1e9 + 7;
    int n = s.length();

    if (n == 0)
        return 0;

    vector<long long> dp(n + 1);
    dp[0] = 1;
    dp[1] = 1;

    for (int i = 2; i <= n; ++i)
    {
        dp[i] = dp[i - 1];

        if (s[i - 1] == 'u' && s[i - 2] == 'u')
        {
            dp[i] = (dp[i] + dp[i - 2]) % MOD;
        }

        if (s[i - 1] == 'n' && s[i - 2] == 'n')
        {
            dp[i] = (dp[i] + dp[i - 2]) % MOD;
        }
    }

    return dp[n];
}

// =========================================================
// PART C: WORLD NAVIGATOR (Graphs)
// =========================================================

// Some Stuff for Kruskal's Algortihm

class DisjointSet
{
private:
    vector<int> parent, rank;

    int find(int u)
    {
        if (parent[u] != u)
            parent[u] = find(parent[u]);
        return parent[u];
    }

    void unite(int u, int v)
    {
        int RootU = find(u);
        int RootV = find(v);

        if (RootU == RootV)
            return; // in the same set

        if (rank[RootU] < rank[RootV])
        {
            parent[RootU] = RootV;
        }
        else if (rank[RootU] > rank[RootV])
        {
            parent[RootV] = RootU;
        }
        else
        {
            parent[RootV] = RootU;
            rank[RootU]++;
        }
    }

public:
    DisjointSet(int n)
    {
        parent.resize(n);
        rank.resize(n, 0);
        iota(parent.begin(), parent.end(), 0); // parent[i] = i
    }

    bool connected(int u, int v)
    {
        return find(u) == find(v);
    }

    void doUnite(int u, int v)
    {
        unite(u, v);
    }
};

struct Edge
{
    int u, v, weight;
    Edge(int u_, int v_, int w_) : u(u_), v(v_), weight(w_) {}
    bool operator<(const Edge &other) const { return weight < other.weight; }
};

//=================================================================================================

bool WorldNavigator::pathExists(int n, vector<vector<int>> &edges, int source, int dest)
{
    if (source == dest)
        return true;
    vector<vector<int>> adj(n);
    for (const auto &e : edges)
    {
        if (e.size() < 2)
            continue;
        int u = e[0], v = e[1];
        if (u >= 0 && u < n && v >= 0 && v < n)
        {
            adj[u].push_back(v);
            adj[v].push_back(u);
        }
    }
    vector<char> vis(n, false);
    queue<int> q;
    vis[source] = true;
    q.push(source);
    while (!q.empty())
    {
        int u = q.front();
        q.pop();
        for (int v : adj[u])
        {
            if (!vis[v])
            {
                if (v == dest)
                    return true;
                vis[v] = true;
                q.push(v);
            }
        }
    }
    return false;
}

/// @brief Implemented using Kruskal's Algorithm
/// @param n Number of cities
/// @param m Number of roads
/// @param goldRate Gold ratio for entier roads
/// @param silverRate Silver ratio for entier roads
/// @param roadData Conatains Roads info as following { Start , End , GoldCostPerRoad , SilverCostPerRoad }
/// @return
long long WorldNavigator::minBribeCost(int n, int m, long long goldRate, long long silverRate, vector<vector<int>> &roadData)
{
    // TODO: Implement Minimum Spanning Tree (Kruskal's or Prim's)
    // roadData[i] = {u, v, goldCost, silverCost}
    // Total cost = goldCost * goldRate + silverCost * silverRate
    // Return -1 if graph cannot be fully connected

    vector<Edge> edges;
    edges.reserve(roadData.size());
    for (const auto &row : roadData)
    {
        int currentCost = goldRate * row[2] + silverRate * row[3];
        edges.emplace_back(row[0], row[1], currentCost);
    }

    sort(edges.begin(), edges.end());

    DisjointSet ds = DisjointSet(n);
    long long totalCost = 0;
    int edgeCount = 0;

    for (const Edge &e : edges)
    {
        if (!ds.connected(e.u, e.v)) // only add if not connected (no cycle)
        {
            totalCost += e.weight;
            ds.doUnite(e.u, e.v);
            edgeCount++;
        }
    }
    if (edgeCount == n - 1)
        return totalCost;
    return -1;
}

string WorldNavigator::sumMinDistancesBinary(int n, vector<vector<int>> &roads)
{
    // TODO: Implement All-Pairs Shortest Path (Floyd-Warshall)
    // Sum all shortest distances between unique pairs (i < j)
    // Return the sum as a binary string
    // Hint: Handle large numbers carefully

    // n = number of vertices
    vector<vector<long long>> dist(n, vector<long long>(n, INF));
    // Initialize distances
    for (int i = 0; i < n; i++)
    {
        dist[i][i] = 0;
    }
    for (const auto &edge : roads)
    {
        if (edge.size() >= 3)
        {
            int u = edge[0], v = edge[1], w = edge[2];
            dist[u][v] = min(dist[u][v], (long long)w);
            dist[v][u] = min(dist[v][u], (long long)w);
        }
    }
    // Floyd-Warshall
    for (int k = 0; k < n; k++)
    {
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if (dist[i][k] < INF && dist[k][j] < INF && dist[i][k] + dist[k][j] < dist[i][j])
                {
                    dist[i][j] = dist[i][k] + dist[k][j];
                }
            }
        }
    }
    long long sum = 0;
    for (int i = 0; i < n; i++)
    {
        for (int j = i + 1; j < n; j++)
        {
            if (dist[i][j] < INF)
                sum += dist[i][j];
        }
    }
    if (sum == 0)
        return "0";
    string binary;
    while (sum > 0)
    {
        binary = char('0' + (sum & 1)) + binary;
        sum >>= 1;
    }
    return binary;
}

// =========================================================
// PART D: SERVER KERNEL (Greedy)
// =========================================================

int ServerKernel::minIntervals(vector<char> &tasks, int n)
{
    // TODO: Implement task scheduler with cooling time
    // Same task must wait 'n' intervals before running again
    // Return minimum total intervals needed (including idle time)
    // Hint: Use greedy approach with frequency counting

    vector<int> freq(26, 0);

    for (char c : tasks)
    {
        freq[c - 'A']++;
    }

    int maxFreq = 0;
    for (int f : freq)
    {
        if (f > maxFreq)
            maxFreq = f;
    }

    int maxCount = 0;
    for (int f : freq)
    {
        if (f == maxFreq)
            maxCount++;
    }

    int partCount = maxFreq - 1;
    int partLength = n - (maxCount - 1);
    int emptySlots = partCount * partLength;
    int remainingTasks = tasks.size() - (maxFreq * maxCount);
    int idle = max(0, emptySlots - remainingTasks);

    return tasks.size() + idle;
    return 0;
}

// =========================================================
// FACTORY FUNCTIONS (Required for Testing)
// =========================================================

extern "C"
{
    PlayerTable *createPlayerTable() { return new ConcretePlayerTable(); }

    Leaderboard *createLeaderboard() { return new ConcreteLeaderboard(); }

    AuctionTree *createAuctionTree() { return new ConcreteAuctionTree(); }
}

//===================================================================================================
//===================================================================================================
//===================================================================================================
//===================================================================================================

/**
 * main_test_student.cpp
 * Basic "Happy Path" Test Suite for ArcadiaEngine
 * Use this to verify your basic logic against the assignment examples.
 */

#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <functional>
#include "ArcadiaEngine.h"

using namespace std;

// ==========================================
// FACTORY FUNCTIONS (LINKING)
// ==========================================
// These link to the functions at the bottom of your .cpp file
extern "C"
{
    PlayerTable *createPlayerTable();
    Leaderboard *createLeaderboard();
    AuctionTree *createAuctionTree();
}

// ==========================================
// TEST UTILITIES
// ==========================================
class StudentTestRunner
{
    int count = 0;
    int passed = 0;
    int failed = 0;

public:
    void runTest(string testName, bool condition)
    {
        count++;
        cout << "TEST: " << left << setw(50) << testName;
        if (condition)
        {
            cout << "[ PASS ]";
            passed++;
        }
        else
        {
            cout << "[ FAIL ]";
            failed++;
        }
        cout << endl;
    }

    void printSummary()
    {
        cout << "\n==========================================" << endl;
        cout << "SUMMARY: Passed: " << passed << " | Failed: " << failed << endl;
        cout << "==========================================" << endl;
        cout << "TOTAL TESTS: " << count << endl;
        if (failed == 0)
        {
            cout << "Great job! All basic scenarios passed." << endl;
            cout << "Now make sure to handle edge cases (empty inputs, collisions, etc.)!" << endl;
        }
        else
        {
            cout << "Some basic tests failed. Check your logic against the PDF examples." << endl;
        }
    }
};

StudentTestRunner runner;

// ==========================================
// PART A: DATA STRUCTURES
// ==========================================

void test_PartA_DataStructures()
{
    cout << "\n--- Part A: Data Structures ---" << endl;

    // 1. PlayerTable (Double Hashing)
    // Requirement: Basic Insert and Search
    PlayerTable *table = createPlayerTable();
    runner.runTest("PlayerTable: Insert 'Alice' and Search", [&]()
                   {
        table->insert(101, "Alice");
        return table->search(101) == "Alice"; }());
    delete table;

    // 2. Leaderboard (Skip List)
    Leaderboard *board = createLeaderboard();

    // Test A: Basic High Score
    runner.runTest("Leaderboard: Add Scores & Get Top 1", [&]()
                   {
        board->addScore(1, 100);
        board->addScore(2, 200); // 2 is Leader
        vector<int> top = board->getTopN(1);
        return (!top.empty() && top[0] == 2); }());

    // Test B: Tie-Breaking Visual Example (Crucial!)
    // PDF Visual Example: Player A (ID 10) 500pts, Player B (ID 20) 500pts.
    // Correct Order: ID 10 then ID 20.
    runner.runTest("Leaderboard: Tie-Break (ID 10 before ID 20)", [&]()
                   {
        board->addScore(10, 500);
        board->addScore(20, 500);
        vector<int> top = board->getTopN(2);
        // We expect {10, 20} NOT {20, 10}
        if (top.size() < 2) return false;
        return (top[0] == 10 && top[1] == 20); }());

    delete board;

    // 3. AuctionTree (Red-Black Tree)
    // Requirement: Insert items without crashing
    AuctionTree *tree = createAuctionTree();
    runner.runTest("AuctionTree: Insert Items", [&]()
                   {
                       tree->insertItem(1, 100);
                       tree->insertItem(2, 50);
                       return true; // Pass if no crash
                   }());
    delete tree;
}

// ==========================================
// PART B: INVENTORY SYSTEM
// ==========================================

void test_PartB_Inventory()
{
    cout << "\n--- Part B: Inventory System ---" << endl;

    // 1. Loot Splitting (Partition)
    // PDF Example: coins = {1, 2, 4} -> Best split {4} vs {1,2} -> Diff 1
    runner.runTest("LootSplit: {1, 2, 4} -> Diff 1", [&]()
                   {
        vector<int> coins = {1, 2, 4};
        return InventorySystem::optimizeLootSplit(3, coins) == 1; }());

    // 2. Inventory Packer (Knapsack)
    // PDF Example: Cap=10, Items={{1,10}, {2,20}, {3,30}}. All fit. Value=60.
    runner.runTest("Knapsack: Cap 10, All Fit -> Value 60", [&]()
                   {
        vector<pair<int, int>> items = {{1, 10}, {2, 20}, {3, 30}};
        return InventorySystem::maximizeCarryValue(10, items) == 60; }());

    // 3. Chat Autocorrect (String DP)
    // PDF Example: "uu" -> "uu" or "w" -> 2 possibilities
    runner.runTest("ChatDecorder: 'uu' -> 2 Possibilities", [&]()
                   { return InventorySystem::countStringPossibilities("uu") == 2; }());
}

// ==========================================
// PART C: WORLD NAVIGATOR
// ==========================================

void test_PartC_Navigator()
{
    cout << "\n--- Part C: World Navigator ---" << endl;

    // 1. Safe Passage (Path Exists)
    // PDF Example: 0-1, 1-2. Path 0->2 exists.
    runner.runTest("PathExists: 0->1->2 -> True", [&]()
                   {
        vector<vector<int>> edges = {{0, 1}, {1, 2}};
        return WorldNavigator::pathExists(3, edges, 0, 2) == true; }());

    // 2. The Bribe (MST)
    // PDF Example: 3 Nodes. Roads: {0,1,10}, {1,2,5}, {0,2,20}. Rate=1.
    // MST should pick 10 and 5. Total 15.
    runner.runTest("MinBribeCost: Triangle Graph -> Cost 15", [&]()
                   {
        vector<vector<int>> roads = {
            {0, 1, 10, 0}, 
            {1, 2, 5, 0}, 
            {0, 2, 20, 0}
        };
        // n=3, m=3, goldRate=1, silverRate=1
        return WorldNavigator::minBribeCost(3, 3, 1, 1, roads) == 15; }());

    // 3. Teleporter (Binary Sum APSP)
    // PDF Example: 0-1 (1), 1-2 (2). Distances: 1, 2, 3. Sum=6 -> "110"
    runner.runTest("BinarySum: Line Graph -> '110'", [&]()
                   {
        vector<vector<int>> roads = {
            {0, 1, 1},
            {1, 2, 2}
        };
        return WorldNavigator::sumMinDistancesBinary(3, roads) == "110"; }());
}

// ==========================================
// PART D: SERVER KERNEL
// ==========================================

void test_PartD_Kernel()
{
    cout << "\n--- Part D: Server Kernel ---" << endl;

    // 1. Task Scheduler
    // PDF Example: Tasks={A, A, B}, n=2.
    // Order: A -> B -> idle -> A. Total intervals: 4.
    runner.runTest("Scheduler: {A, A, B}, n=2 -> 4 Intervals", [&]()
                   {
        vector<char> tasks = {'A', 'A', 'B'};
        return ServerKernel::minIntervals(tasks, 2) == 4; }());
}

int main()
{
    cout << "Arcadia Engine - Student Happy Path Tests" << endl;
    cout << "-----------------------------------------" << endl;

    test_PartA_DataStructures();
    test_PartB_Inventory();
    test_PartC_Navigator();
    test_PartD_Kernel();

    runner.printSummary();

    return 0;
}