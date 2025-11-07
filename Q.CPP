#include <iostream>
#include <climits>
using namespace std;

struct Node {
    int data;
    Node* left;
    Node* right;
};

Node* newNode(int value) {
    Node* temp = new Node;
    temp->data = value;
    temp->left = temp->right = NULL;
    return temp;
}

// (a) INSERT (No Duplicates)
Node* insert(Node* root, int value) {
    if (root == NULL) 
        return newNode(value);

    if (value < root->data)
        root->left = insert(root->left, value);
    else if (value > root->data)
        root->right = insert(root->right, value);

    return root;
}

// (b) SEARCH - Recursive
bool searchRecursive(Node* root, int key) {
    if (root == NULL) return false;
    if (root->data == key) return true;
    else if (key < root->data) return searchRecursive(root->left, key);
    else return searchRecursive(root->right, key);
}

// (b) SEARCH - Non-Recursive
bool searchIterative(Node* root, int key) {
    while (root != NULL) {
        if (key == root->data) return true;
        else if (key < root->data) root = root->left;
        else root = root->right;
    }
    return false;
}

// (c) FIND MINIMUM
Node* findMin(Node* root) {
    while (root && root->left != NULL)
        root = root->left;
    return root;
}

// (c) FIND MAXIMUM
Node* findMax(Node* root) {
    while (root && root->right != NULL)
        root = root->right;
    return root;
}

// (d) DELETE NODE
Node* deleteNode(Node* root, int key) {
    if (root == NULL) return root;

    if (key < root->data)
        root->left = deleteNode(root->left, key);
    else if (key > root->data)
        root->right = deleteNode(root->right, key);
    else {
        // Node found
        if (root->left == NULL) {
            Node* temp = root->right;
            delete root;
            return temp;
        }
        else if (root->right == NULL) {
            Node* temp = root->left;
            delete root;
            return temp;
        }

        // Node with 2 children
        Node* temp = findMin(root->right);
        root->data = temp->data;
        root->right = deleteNode(root->right, temp->data);
    }
    return root;
}

// TRAVERSALS
void preorder(Node* root) {
    if (root) {
        cout << root->data << " ";
        preorder(root->left);
        preorder(root->right);
    }
}

void inorder(Node* root) {
    if (root) {
        inorder(root->left);
        cout << root->data << " ";
        inorder(root->right);
    }
}

void postorder(Node* root) {
    if (root) {
        postorder(root->left);
        postorder(root->right);
        cout << root->data << " ";
    }
}

// INORDER SUCCESSOR (without parent pointer)
Node* inorderSuccessor(Node* root, Node* target) {
    if (target->right != NULL)
        return findMin(target->right);

    Node* succ = NULL;
    while (root != NULL) {
        if (target->data < root->data) {
            succ = root;
            root = root->left;
        }
        else if (target->data > root->data) {
            root = root->right;
        }
        else break;
    }
    return succ;
}

// INORDER PREDECESSOR
Node* inorderPredecessor(Node* root, Node* target) {
    if (target->left != NULL)
        return findMax(target->left);

    Node* pred = NULL;
    while (root != NULL) {
        if (target->data > root->data) {
            pred = root;
            root = root->right;
        }
        else if (target->data < root->data) {
            root = root->left;
        }
        else break;
    }
    return pred;
}

// MAXIMUM DEPTH
int maxDepth(Node* root) {
    if (root == NULL) return 0;
    return 1 + max(maxDepth(root->left), maxDepth(root->right));
}

// MINIMUM DEPTH
int minDepth(Node* root) {
    if (root == NULL) return 0;
    if (!root->left && !root->right) return 1;

    if (!root->left) return 1 + minDepth(root->right);
    if (!root->right) return 1 + minDepth(root->left);

    return 1 + min(minDepth(root->left), minDepth(root->right));
}

// CHECK IF TREE IS BST
bool isBSTUtil(Node* root, int minVal, int maxVal) {
    if (root == NULL) return true;
    if (root->data <= minVal || root->data >= maxVal) return false;
    return isBSTUtil(root->left, minVal, root->data) &&
           isBSTUtil(root->right, root->data, maxVal);
}

bool isBST(Node* root) {
    return isBSTUtil(root, INT_MIN, INT_MAX);
}

// MAIN
int main() {
    Node* root = NULL;
    root = insert(root, 50);
    insert(root, 30);
    insert(root, 70);
    insert(root, 20);
    insert(root, 40);
    insert(root, 60);
    insert(root, 80);

    cout << "Inorder: ";
    inorder(root);
    cout << "\nPreorder: ";
    preorder(root);
    cout << "\nPostorder: ";
    postorder(root);

    cout << "\n\nSearch 40 (Rec): " << (searchRecursive(root, 40) ? "Found" : "Not Found");
    cout << "\nSearch 40 (Iter): " << (searchIterative(root, 40) ? "Found" : "Not Found");

    cout << "\n\nMin: " << findMin(root)->data;
    cout << "\nMax: " << findMax(root)->data;

    Node* target = root->left; // 30
    Node* succ = inorderSuccessor(root, target);
    Node* pred = inorderPredecessor(root, target);
    cout << "\n\nSuccessor of " << target->data << ": " << (succ ? to_string(succ->data) : "None");
    cout << "\nPredecessor of " << target->data << ": " << (pred ? to_string(pred->data) : "None");

    cout << "\n\nMax Depth: " << maxDepth(root);
    cout << "\nMin Depth: " << minDepth(root);

    cout << "\n\nIs BST? " << (isBST(root) ? "Yes" : "No");

    root = deleteNode(root, 30);
    cout << "\n\nAfter deleting 30, Inorder: ";
    inorder(root);

    return 0;
}
