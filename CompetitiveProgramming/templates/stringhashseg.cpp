#include <vector>
#pragma GCC target ("avx2")
#pragma GCC optimize ("O3")
#pragma GCC optimize ("unroll-loops")
#include <bits/stdc++.h>
#include <ext/pb_ds/assoc_container.hpp>

using namespace std;
using namespace __gnu_pbds;

template<typename T>
ostream_iterator<T> oit(const string &s = " "){ return ostream_iterator<T>(cout,s.c_str()); }
inline auto rep(int l, int r) { return views::iota(min(l, r), r); }
inline auto rep(int n) { return rep(0, n); }
inline auto rep1(int l, int r) { return rep(l, r + 1); }
inline auto rep1(int n) { return rep(1, n + 1); }
inline auto per(int l, int r) { return rep(l, r) | views::reverse; }
inline auto per(int n) { return per(0, n); }
inline auto per1(int l, int r) { return per(l, r + 1); }
inline auto per1(int n) { return per(1, n + 1); }
#define A(a) begin(a),end(a)
inline auto len = ranges::ssize;

struct chash {
    static uint64_t splitmix64(uint64_t x) {
        // http://xorshift.di.unimi.it/splitmix64.c
        x += 0x9e3779b97f4a7c15;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9;
        x = (x ^ (x >> 27)) * 0x94d049bb133111eb;
        return x ^ (x >> 31);
    }

    size_t operator()(uint64_t x) const {
        static const uint64_t FIXED_RANDOM = chrono::steady_clock::now().time_since_epoch().count();
        return splitmix64(x + FIXED_RANDOM);
    }
};
template<typename T, typename U> using pb_map = gp_hash_table<T, U, chash>;
template<typename T> using pb_set = gp_hash_table<T, null_type, chash>;
#define K first
#define V second

using ll = long long;
using ld = long double;

using vi = vector<int>;
using vii = vector<vector<int>>;
typedef vector<ll> vll;
using pll = pair<ll,ll>;
using pii = pair<int,int>;


constexpr ll N = 4e5+100, L = 25;
constexpr ll NN = 5*N;

int dep[N], par[N][L];
void lca_dfs(int i, vector<vector<int>> &adj){
    for(int l = 1;l<L;l++)
        par[i][l] = par[par[i][l-1]][l-1];

    for(int j : adj[i]){
        if(j == par[i][0]) continue;
        dep[j] = dep[i]+1,par[j][0]=i;
        lca_dfs(j,adj);
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
int tree_dist(int u,int v){
    return dep[u]+dep[v] - 2*dep[lca(u,v)];
}
typedef pair<unsigned ll, ll> hsh;
#define M 1000000321
#define OP(x, y) constexpr hsh operator x (const hsh a, const hsh b) { return { a.K x b.K, (a.V y b.V + M) % M }; }
OP(+, +) OP(*, *) OP(-, + M -)
mt19937 gen(chrono::steady_clock::now().time_since_epoch().count());
uniform_int_distribution<ll> dist(5e6, M - 1);

const ll SN = NN + 10;
hsh p[SN];

struct NODE{
    hsh hs; ll len;
};

namespace seg {
    using T = NODE;
    T id {{0, 0}, 0};
    T f(T a, T b) {
        return {a.hs + b.hs * p[a.len], a.len + b.len};
    }

    T t[2 * NN];
    ll n=NN;  // array size

    void modify(ll p, ll input) {  // set value at position p
      for (p+=n, t[p] = {{input %M, input%M}, 1}; p /= 2;) t[p] = f(t[2*p], t[2*p+1]);
    }
    void modifysafds(ll p) {  // set value at position p
      for (p+=n, t[p] = id; p /= 2;) t[p] = f(t[2*p], t[2*p+1]);
    }

    void reset(ll n) {
        for (int i= 0 ; i < n; ++i) modifysafds(i);
    }

    T query(ll l, ll r) { // fold f on interval [l, r)
      T resl=id, resr=id;
      for (l += n, r += n; l < r; l /= 2, r /= 2) {
        if (l&1) resl = f(resl, t[l++]);
        if (r&1) resr = f(t[--r], resr);
      }
      return f(resl, resr);
    }
}


void run()
{
    int n; cin >> n;
    vi proj(n); for(int &x : proj) cin >> x,--x;
    vii proj_mp(n), adj(n); for(int i : rep(n)) proj_mp[proj[i]].push_back(i);
    for(int d; int i : rep(n)){
        cin >> d;
        for(int x : rep(d)){
            cin >> x,--x;
            adj[i].push_back(x);
        }
    }
    vi tin(n),tout(n); //tin precomp dfs
    vi proj_min_tn(n,1e9);
    {
        // inclusive tout 
        ll time = 0;
        function<void(int)> dfs = [&](int i){
            tin[i] = time++;
            proj_min_tn[proj[i]] = min(proj_min_tn[proj[i]],tin[i]);
            for(int j : adj[i])
                dfs(j);
            tout[i] = time++;
        }; dfs(0);
    }
    vii proj_start_here(n);
    for(int i : rep(n)){
        dep[i]=0;
        for(int l : rep(L)) par[i][l]=0;
    } lca_dfs(0,adj);
    for(int i : rep(n)){
        if(not proj_mp[i].empty()){
            int lc = proj_mp[i][0];
            for(int u : proj_mp[i])lc=lca(u,lc);
            proj_start_here[lc].push_back(i);
        }
    }for(auto &v : proj_start_here)
        ranges::sort(v,[&](int p1,int p2){return proj_min_tn[p1]<proj_min_tn[p2];});

    vector<set<int>> st_vec(n); map<hsh,int> hash_cnt; vector<hsh> node_to_hash(n);
    seg::reset(4*n + 10);
    vi cur_idx(n);
    auto mod = [&](int i,int c){
        cur_idx[i]=c;
        seg::modify(tin[i],c);
    };
    function<void(int)> dfs = [&](int i){
        for(int j : adj[i]) dfs(j); //dfs first
        mod(i,proj[i]);
        seg::modify(tout[i],seg::tout_char);
        auto &here_st = st_vec[i]; 
        int idx = n;
        for(int p : proj_start_here[i]){
            for(int j : proj_mp[p]){
                here_st.insert(j);
                mod(j,idx);
            }
            idx++;
        }
        for(int j : adj[i]){
            auto &there_st = st_vec[j];
            if(len(there_st)>len(here_st)) swap(there_st,here_st);
            for(int u : there_st){
                mod(u,cur_idx[u]+len(here_st));
            }
            for(int u : there_st) here_st.insert(u);
        }  
        auto h = seg::query(tin[i],tout[i]);
        node_to_hash[i] = h.hs;
        hash_cnt[h.hs]++;
    }; dfs(0);

    for(int i : rep(n))
        cout << hash_cnt[node_to_hash[i]]-1 << ' ';
    cout << '\n';
    // cout << ranges::max(tin) << " " << ranges::max(tout) << endl;
}

int main()
{
    //KING OF THE WORLD...... U.W.T.B
    cin.tie(0);
    
    p[0] = { 1, 1 }, p[1] = { dist(gen) | 1, dist(gen) };
    for (int i = 0; i < SN-1; ++i) p[i + 1] = p[i] * p[1];

    int t; cin>>t; while(t--) run();
}
