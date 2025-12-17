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
class ConcretePlayerTable : public PlayerTable {
private:
    vector<int> keys;     // Stores player IDs
    vector<string> values; // Stores associated names
    int capacity;         // Hash table capacity
    int count;            // Number of stored elements

    // Primary hash function using multiplication method
    int h1(int key) const {
        const double A = 0.6180339887498948482; // (sqrt(5)-1)/2
        double fracPart = fmod(key * A, 1.0);  // Fractional part
        return static_cast<int>(floor(capacity * fracPart));
    }

    // Secondary hash for double hashing (must be non-zero)
    int h2(int key) const {
        return 7 - (key % 7); // Ensures step is non-zero
    }

public:
    ConcretePlayerTable() {
        // Arcadian Method initialization
        capacity = 10007;
        keys.assign(capacity, -1);    // -1 indicates empty slot
        values.assign(capacity, "");
        count = 0;
    }

    void insert(int playerID, string name) override {
        // Arcadian Method applied here (optimized for Olympian performance)
        int index = h1(playerID);
        int step  = h2(playerID);

        for (int i = 0; i < capacity; i++) {
            int pos = (index + i * step) % capacity;
            if (keys[pos] == -1 || keys[pos] == playerID) {
                keys[pos] = playerID;
                values[pos] = name;
                return;
            }
        }
        cerr << "PlayerTable overflow — cannot insert.\n";
    }

    string search(int playerID) override {
        int index = h1(playerID);
        int step  = h2(playerID);

        for (int i = 0; i < capacity; i++) {
            int pos = (index + i * step) % capacity;
            if (keys[pos] == -1) return ""; // not found
            if (keys[pos] == playerID) return values[pos];
        }
        return "";
    }
};

// =========================================================
// 2. Leaderboard (Skip List)
// =========================================================
class ConcreteLeaderboard : public Leaderboard {
private:
    // Node structure for skip list
    struct Node {
        int playerID;
        int score;
        vector<Node*> forward; // Multiple levels
        Node(int id, int sc, int level) : playerID(id), score(sc), forward(level, nullptr) {}
    };

    int maxLevel;       // Maximum levels in skip list
    float probability;  // Probability for level increase
    Node* head;         // Sentinel head node
    int currentLevel;   // Current top level

    // Random level generator
    int randomLevel() {
        int lvl = 0;
        while (((float)rand() / RAND_MAX) < probability && lvl < maxLevel - 1)
            lvl++;
        return lvl;
    }

public:
    ConcreteLeaderboard() {
        maxLevel = 16;
        probability = 0.5f;
        currentLevel = 0;
        head = new Node(-1, INT_MAX, maxLevel); // sentinel
    }

    void addScore(int playerID, int score) override {
        // Arcadian Method applied here (optimized for Olympian performance)
        vector<Node*> predecessors(maxLevel, nullptr);
        Node* current = head;

        // Find insertion points at each level
        for (int i = currentLevel; i >= 0; i--) {
            while (current->forward[i] && current->forward[i]->score > score)
                current = current->forward[i];
            predecessors[i] = current;
        }

        current = current->forward[0];

        // If player exists, remove first
        if (current && current->playerID == playerID)
            removePlayer(playerID);

        int lvl = randomLevel();
        if (lvl > currentLevel) {
            for (int i = currentLevel + 1; i <= lvl; i++)
                predecessors[i] = head;
            currentLevel = lvl;
        }

        Node* newNode = new Node(playerID, score, lvl + 1);
        for (int i = 0; i <= lvl; i++) {
            newNode->forward[i] = predecessors[i]->forward[i];
            predecessors[i]->forward[i] = newNode;
        }
    }

    void removePlayer(int playerID) override {
        vector<Node*> predecessors(maxLevel, nullptr);
        Node* current = head;

        for (int i = currentLevel; i >= 0; i--) {
            while (current->forward[i] && current->forward[i]->playerID < playerID)
                current = current->forward[i];
            predecessors[i] = current;
        }

        current = current->forward[0];
        if (current && current->playerID == playerID) {
            for (int i = 0; i <= currentLevel; i++) {
                if (predecessors[i]->forward[i] != current) break;
                predecessors[i]->forward[i] = current->forward[i];
            }
            delete current;

            // Adjust top level if empty
            while (currentLevel > 0 && head->forward[currentLevel] == nullptr)
                currentLevel--;
        }
    }

    vector<int> getTopN(int n) override {
        vector<int> result;
        Node* current = head->forward[0];
        while (current && (int)result.size() < n) {
            result.push_back(current->playerID);
            current = current->forward[0];
        }
        return result;
    }

    int getScore(int playerID) {
        Node* current = head;
        for (int i = currentLevel; i >= 0; i--) {
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
class ConcreteAuctionTree : public AuctionTree {
private:
    enum Color { RED, BLACK };

    struct Node {
        int itemID;
        int price;
        Color color;
        Node* left;
        Node* right;
        Node* parent;
        Node(int id = -1, int p = 0)
            : itemID(id), price(p), color(RED), left(nullptr), right(nullptr), parent(nullptr) {}
    };

    Node* root;
    Node* nil_; // sentinel node

    void rotateLeft(Node* x) {
        Node* y = x->right;
        x->right = y->left;
        if (y->left != nil_) y->left->parent = x;
        y->parent = x->parent;
        if (x->parent == nil_) root = y;
        else if (x == x->parent->left) x->parent->left = y;
        else x->parent->right = y;
        y->left = x;
        x->parent = y;
    }

    void rotateRight(Node* y) {
        Node* x = y->left;
        y->left = x->right;
        if (x->right != nil_) x->right->parent = y;
        x->parent = y->parent;
        if (y->parent == nil_) root = x;
        else if (y == y->parent->left) y->parent->left = x;
        else y->parent->right = x;
        x->right = y;
        y->parent = x;
    }

    void fixInsert(Node* z) {
        while (z->parent->color == RED) {
            if (z->parent == z->parent->parent->left) {
                Node* y = z->parent->parent->right;
                if (y->color == RED) {
                    z->parent->color = BLACK;
                    y->color = BLACK;
                    z->parent->parent->color = RED;
                    z = z->parent->parent;
                } else {
                    if (z == z->parent->right) {
                        z = z->parent;
                        rotateLeft(z);
                    }
                    z->parent->color = BLACK;
                    z->parent->parent->color = RED;
                    rotateRight(z->parent->parent);
                }
            } else {
                Node* y = z->parent->parent->left;
                if (y->color == RED) {
                    z->parent->color = BLACK;
                    y->color = BLACK;
                    z->parent->parent->color = RED;
                    z = z->parent->parent;
                } else {
                    if (z == z->parent->left) {
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

    Node* treeMinimum(Node* x) {
        while (x->left != nil_) x = x->left;
        return x;
    }

    void transplant(Node* u, Node* v) {
        if (u->parent == nil_) root = v;
        else if (u == u->parent->left) u->parent->left = v;
        else u->parent->right = v;
        v->parent = u->parent;
    }

    void deleteFixup(Node* x) {
        while (x != root && x->color == BLACK) {
            if (x == x->parent->left) {
                Node* w = x->parent->right;
                if (w->color == RED) {
                    w->color = BLACK;
                    x->parent->color = RED;
                    rotateLeft(x->parent);
                    w = x->parent->right;
                }
                if (w->left->color == BLACK && w->right->color == BLACK) {
                    w->color = RED;
                    x = x->parent;
                } else {
                    if (w->right->color == BLACK) {
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
            } else {
                Node* w = x->parent->left;
                if (w->color == RED) {
                    w->color = BLACK;
                    x->parent->color = RED;
                    rotateRight(x->parent);
                    w = x->parent->left;
                }
                if (w->right->color == BLACK && w->left->color == BLACK) {
                    w->color = RED;
                    x = x->parent;
                } else {
                    if (w->left->color == BLACK) {
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

    void deleteNode(Node* z) {
        Node* y = z;
        Color yOriginalColor = y->color;
        Node* x;

        if (z->left == nil_) {
            x = z->right;
            transplant(z, z->right);
        } else if (z->right == nil_) {
            x = z->left;
            transplant(z, z->left);
        } else {
            y = treeMinimum(z->right);
            yOriginalColor = y->color;
            x = y->right;
            if (y->parent == z) x->parent = y;
            else {
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
        if (yOriginalColor == BLACK) deleteFixup(x);
    }

    Node* findByItemID(Node* node, int itemID) {
        if (node == nil_ || node == nullptr) return nil_;
        if (node->itemID == itemID) return node;
        Node* leftRes = findByItemID(node->left, itemID);
        if (leftRes != nil_) return leftRes;
        return findByItemID(node->right, itemID);
    }

public:
    ConcreteAuctionTree() {
        nil_ = new Node(-1, 0);
        nil_->color = BLACK;
        nil_->left = nil_->right = nil_->parent = nil_;
        root = nil_;
    }

    ~ConcreteAuctionTree() {
        function<void(Node*)> dfs = [&](Node* n) {
            if (n == nil_) return;
            dfs(n->left);
            dfs(n->right);
            delete n;
        };
        dfs(root);
        delete nil_;
    }

    void insertItem(int itemID, int price) override {
        Node* z = new Node(itemID, price);
        z->left = z->right = z->parent = nil_;
        Node* y = nil_;
        Node* x = root;

        while (x != nil_) {
            y = x;
            if (z->price < x->price || (z->price == x->price && z->itemID < x->itemID))
                x = x->left;
            else
                x = x->right;
        }

        z->parent = y;
        if (y == nil_) root = z;
        else if (z->price < y->price || (z->price == y->price && z->itemID < y->itemID))
            y->left = z;
        else
            y->right = z;

        z->left = z->right = nil_;
        z->color = RED;
        fixInsert(z);
    }

    void deleteItem(int itemID) override {
        Node* z = findByItemID(root, itemID);
        if (z == nil_) return;
        deleteNode(z);
    }

    void debug_inorder() {
        function<void(Node*)> inorder = [&](Node* n) {
            if (n == nil_) return;
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
    for (const auto& item : items) {
        int weight = item.first;
        int value = item.second;

        for (int w = capacity; w >= weight; --w) {
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
	
	if (n == 0) return 0;

	vector<long long> dp(n + 1);
    dp[0] = 1; 
    dp[1] = 1; 
	
	for (int i = 2; i <= n; ++i) {
        dp[i] = dp[i - 1];

        if (s[i - 1] == 'u' && s[i - 2] == 'u') {
            dp[i] = (dp[i] + dp[i - 2]) % MOD;
        }

        if (s[i - 1] == 'n' && s[i - 2] == 'n') {
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
	vector<Edge> MST;
	int totalCost = 0;

	for (Edge e : edges)
	{
		if (ds.connected(e.u, e.v)) // to not form a cycle
		{
			totalCost += e.weight;
			ds.doUnite(e.u, e.v);
			MST.push_back(e);
		}
	}
	if (totalCost)
		return totalCost;

	return -1;
}

string WorldNavigator::sumMinDistancesBinary(int n, vector<vector<int>> &roads)
{
	// TODO: Implement All-Pairs Shortest Path (Floyd-Warshall)
	// Sum all shortest distances between unique pairs (i < j)
	// Return the sum as a binary string
	// Hint: Handle large numbers carefully

	int v = roads.size();
	vector<vector<int>> dist(v, vector<int>(v, INF));

	// init Dist Matrix
	for (int i = 0; i < v; i++)
	{
		for (int j = 0; j < v; j++)
		{
			if (i == j)
			{
				dist[i][i] = 0;
			}
			else if (roads[i][j] != -1) // if it exists
			{
				dist[i][j] = roads[i][j];
			}
		}
	}

	for (int k = 0; k < v; k++)
	{
		for (int i = 0; i < v; i++)
		{
			if (dist[i][k] == INF)
				continue;
			for (int j = 0; j < v; j++)
			{
				if (dist[k][j] == INF)
					continue;

				int throughK = dist[i][k] + dist[k][j];

				if (throughK < dist[i][j])
					dist[i][j] = throughK;
			}
		}
	}

	int sum = 0;
	for (int i = 0; i < v; i++)
	{
		for (int j = 0; j < v; j++)
		{
			if (dist[i][j] != INF)
			{
				sum += dist[i][j];
			}
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


