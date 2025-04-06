#include <bits/stdc++.h>
using namespace std;

typedef long long int ll;
typedef pair<ll, ll> pl;
typedef vector<ll> vl;

constexpr ll N=2e5,L=20;

int dep[N], par[N][L];
void dfs(int i, vector<vector<int>> &adj){
    for(int l = 1;l<L;l++)
        par[i][l] = par[par[i][l-1]][l-1];

    for(int j : adj[i]){
        if(j == par[i][0]) continue;
        dep[j] = dep[i]+1,par[j][0]=i;
        dfs(j,adj);
    }
}

int lca(int u,int v){
    if(dep[u]<dep[v]) swap(u,v);

    for(int l = L-1;l>=0;l--)
        if((dep[u]-dep[v]) >> l) u=par[u][l];

    if(u==v) return u;

    for(int l = L-1;l>=0;l--)
        if(par[u][l]!=par[v][l])u=par[u][l],v=par[v][l];
    return par[u][0];
}

//dist between u and v
int dist(int u,int v){
    return dep[u]+dep[v] - 2*dep[lca(u,v)];
}

//dist from node x to the path u,v
int path_dist(int x,int u,int v){
    return (dist(x,u)+dist(x,v) - dist(u,v))/2;
}

//0 indexed kth node on path from u -> v
int kth_on_path(int u,int v,int k){
    int d = dist(u,v), top = lca(u,v);

    if(dep[u]-k < dep[top])
        swap(u,v),k=d-k;

    for(int l = L-1;l>=0;l--)
        if(k&(1<<l))
            u=par[u][l];

    return u;
}

int main(){
    int n,q,a,b,c,d; cin >> n >> q;
    vector<vector<int>> adj(n);
    for(int i=0,u,v;i<n-1;i++){
        cin >> u >> v; u--,v--;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    dfs(0,adj);

    while(q--){
        cin >> a >> b >> c >> d; --a,--b,--c,--d;
        int dist_a = path_dist(a,c,d), dist_b = path_dist(b,c,d);
        if(dist_a + dist_b > dist(a,b)){ //intersection is empty
            cout << -1 << '\n'; continue;
        }

        int x=kth_on_path(a,d,dist_a),y=kth_on_path(b,d,dist_b);
        cout << x+1 << " " << y+1 << '\n';
    }

    return 0;
}