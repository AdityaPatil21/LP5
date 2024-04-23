#include<iostream>
#include<stdlib.h>
#include<queue>
#include<stack>
#include<omp.h>

using namespace std;

// Class representing a node in a binary tree
class node
{
   public:
    
    node *left, *right;
    int data;

};    

class Breadthfs
{
 
 public:
 // Function to insert a node into the binary tree
 node *insert(node *, int);
   // Function to perform Breadth First Search traversal
 void bfs(node *);
 
};


node *insert(node *root, int data)
// inserts a node in tree
{
    // If the tree is empty, create a new root node
    if(!root)
    {
   	 
   	 root=new node;
   	 root->left=NULL;
   	 root->right=NULL;
   	 root->data=data;
   	 return root;
    }

    queue<node *> q;
    q.push(root);
    // Traverse the tree level by level to find a place to insert the new node
    while(!q.empty())
    {

   	 node *temp=q.front();
   	 q.pop();
         // If the left child of the current node is empty, insert the new node there
   	 if(temp->left==NULL)
   	 {
   		 
   		 temp->left=new node;
   		 temp->left->left=NULL;
   		 temp->left->right=NULL;
   		 temp->left->data=data;    
   		 return root;
   	 }
   	 else
   	 {

   	 q.push(temp->left);

   	 }
          // If the right child of the current node is empty, insert the new node there
   	 if(temp->right==NULL)
   	 {
   		 
   		 temp->right=new node;
   		 temp->right->left=NULL;
   		 temp->right->right=NULL;
   		 temp->right->data=data;    
   		 return root;
   	 }
   	 else
   	 {

   	 q.push(temp->right);

   	 }

    }
    
}


void bfs(node *head)
{

   	 queue<node*> q;
   	 q.push(head);
   	 
   	 int qSize;
   	 // Continue BFS until all nodes are visited
   	 while (!q.empty())
   	 {
   		 qSize = q.size();
   		 // Iterate through the current level nodes and print their data
   		 #pragma omp parallel for
            	//creates parallel threads
   		 for (int i = 0; i < qSize; i++)
   		 {
   			 node* currNode;
   			 // Critical section to access and remove node from the queue safely
   			 #pragma omp critical
   			 {
   			   currNode = q.front();
   			   q.pop();
   			   cout<<"\t"<<currNode->data;
   			   
   			 }// prints parent node
   			 #pragma omp critical
   			 {
   			 if(currNode->left)// push parent's left node in queue
   				 q.push(currNode->left);
   			 if(currNode->right)
   				 q.push(currNode->right);
   			 }// push parent's right node in queue   	 

   		 }
   	 }

}

void dfs(node *root)
{
    stack<node *> s;
    s.push(root);

    while (!s.empty())
    {
        int stackSize = s.size();
        // Iterate through the current level nodes
        #pragma omp parallel for
        for (int i = 0; i < stackSize; ++i)
        {
            node *current;
            // Critical section to access and remove node from the stack safely
            #pragma omp critical
            {
                current = s.top();
                s.pop();
            }
            cout << "\t" << current->data;
             // Push the right child into the stack if exists
            if (current->right)
            {
                #pragma omp critical
                {
                    s.push(current->right);
                }
            }
            // Push the left child into the stack if exists
            if (current->left)
            {
                #pragma omp critical
                {
                    s.push(current->left);
                }
            }
        }
    }
}

int main(){

    node *root=NULL;
    int data;
    char ans;
    
    do
    {
   	 cout<<"\n enter data=>";
   	 cin>>data;
   	   // Insert the entered data into the binary tree
   	 root=insert(root,data);
    
   	 cout<<"do you want insert one more node?";
   	 cin>>ans;
    
    }while(ans=='y'||ans=='Y');
    
    cout << "\nBFS Traversal:" << endl;
    bfs(root);

    cout << "\n\nDFS Traversal:" << endl;
    dfs(root);
    
    return 0;
}















































/* 

How parallelism is implemented ?

Parallelism is implemented in the provided code using OpenMP directives, which enable the execution of certain parts of the code in parallel across multiple threads. Here's how parallelism is incorporated into the code

BFS Traversal (Parallel):

In the bfs function, a parallel loop is used to iterate through the nodes at the current level of the binary tree.
The #pragma omp parallel for directive parallelizes the loop iterations, distributing them among multiple threads to execute concurrently.
Each thread processes a subset of the nodes at the current level simultaneously, improving performance by utilizing multiple CPU cores.
Within the loop, a critical section (#pragma omp critical) is used to ensure safe access and removal of nodes from the queue (q). This ensures that only one thread accesses the queue at a time to prevent data corruption.

DFS Traversal (Parallel):

In the dfs function, a similar parallel loop is used to iterate through the nodes at each level of the binary tree iteratively.
The loop is parallelized using the #pragma omp parallel for directive, allowing multiple threads to process different segments of the stack concurrently.
Again, a critical section (#pragma omp critical) is employed to ensure safe access and modification of the stack (s). Only one thread accesses the stack at a time to avoid race conditions and maintain data integrity.








OpenMP?

OpenMP (Open Multi-processing) is an application programming interface (API) that allows programmers to explicitly direct multi-threaded, shared memory parallelism in C, C++, and Fortran programs. OpenMP is a standard parallel programming API for shared memory environments and is considered by many to be an ideal solution for parallel programming. It is written in C, C++, or FORTRAN and supports many platforms, instruction-set architectures, and operating systems, including Solaris, AIX, FreeBSD, HP-UX, Linux, macOS, and Windows. 

OpenMP is based upon the existence of multiple threads in the shared memory programming paradigm. It is an explicit, none-automatic, programming model which offers the programmer full control over parallelization. OpenMP uses a portable, scalable model that gives programmers a simple and flexible interface for developing parallel applications for platforms that ranges from the normal desktop computer to the high-end supercomputers. 

OpenMP is typically used for loop-level parallelism, but it also supports function-level parallelism. This mechanism is called OpenMP sections. The structure of sections is straightforward and can be useful in many instances.






*/
