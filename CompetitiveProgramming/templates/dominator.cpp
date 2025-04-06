#include <bits/stdc++.h>

using ll = long long;
constexpr ll NN = 2e5;

//usage:
//dominator::adj[u].push_back(v);
//dominator::construct_tree(n, src);
namespace dominator{
    using namespace std;
    vi adj[NN], radj[NN], tree[NN], bucket[NN];
    int sdom[NN], par[NN], dom[NN];
    int arr[NN], rev[NN];
    int T = 0;
 
    namespace dsu{
        int par[NN], label[NN];
        int find(int u, int x = 0){
            if(u==par[u]) return x ? -1 : u;
            int v = find(par[u],x+1);
            if(v<0) return u;
            if(sdom[label[par[u]]] < sdom[label[u]])
                label[u]=label[par[u]];
            par[u]=v;
            return x? v:label[u]; //cool trick to return node if recursing but the smallest valid sdom to the caller
        }
        void merge(int u,int v) {
            par[v]=u;
        }
    }
 
    void construct_tree(int n, int src = 0){
        T=0;
        iota(sdom,sdom+n,0);
        iota(dsu::label,dsu::label+n,0);
        iota(dsu::par,dsu::par+n,0);
        fill_n(arr,n,-1);
        fill_n(par,n,-1);
 
        function<void(int)> dfs = [&](int i){
            arr[i]=T,rev[T]=i, ++T;
            for(int j : adj[i]){
                if(arr[j]==-1){
                    dfs(j),par[arr[j]]=arr[i];
                }
                radj[arr[j]].push_back(arr[i]);
            }
        }; dfs(src);
        n = T; //n is now only the nodes reachable from the src

        for(int i = n-1;i>=0;i--){
            for(int j : radj[i])
                sdom[i] = min(sdom[i],sdom[dsu::find(j)]);
            if(i) bucket[sdom[i]].push_back(i);

            int v;
            for(int w : bucket[i]){
                v=dsu::find(w);
                if(sdom[v]==sdom[w]) dom[w]=sdom[w];
                else dom[w]=v;
            }
            if(i) dsu::merge(par[i],i);
        }
 
        for(int i = 1;i<n;i++){
            if(dom[i]!=sdom[i])dom[i]=dom[dom[i]];
            tree[rev[dom[i]]].push_back(rev[i]);
        }
    }
}

int main(){
    return 0;
}