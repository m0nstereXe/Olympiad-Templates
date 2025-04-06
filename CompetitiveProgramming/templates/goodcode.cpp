//header
#include <vector>
#pragma GCC target ("avx2")
#pragma GCC optimize ("O3")
#pragma GCC optimize ("unroll-loops")
#include <bits/stdc++.h>

using namespace std;
typedef long long ll;
typedef long double ld;

int main() {
	ios_base::sync_with_stdio(false);
	cin.tie(NULL);
}

// Debugs be like
// This has all containers and pair
template<class T, class S>
ostream& operator<<(ostream &o, pair<T, S> p) {
    return o<<'('<<p.first<<", "<<p.second<<')';
}

template<template<class, class...> class T, class... A>
typename enable_if<!is_same<T<A...>, string>(), ostream&>::type
operator<<(ostream &o, T<A...> V) {
	o<<'[';
	for(auto a:V) o<<a<<", ";
	return o<<']';
}

// This is tuples, _p is helper
template<ll i, class... T>
typename enable_if<i==sizeof...(T)>::type _p(ostream& o, tuple<T...> t) {}

template<ll i, class... T>
typename enable_if<i<sizeof...(T)>::type _p(ostream& o, tuple<T...> t) {
    _p<i+1>(o << get<i>(t)<< ", ", t);
}
 
template<class... T>
ostream& operator<<(ostream& o, tuple<T...> t) {
    _p<0>(o<<'(', t);
    return o<<')';
}

//pq for djikstra
priority_queue<pair<ll, ll>, vector<pair<ll, ll> >, greater<pair<ll, ll>>> fringe;

//segtree without lazy propogation
//Allows for changing op through f and identity in id
namespace seg {
    using T = ll;
    T id=0;
    T f(T a, T b) {return a+b;}

    T t[2 * NN];
    ll n=NN;  // array size

    void modify(ll p, T value) {  // set value at position p
      for (p+=n, t[p] = value; p /= 2;) t[p] = f(t[2*p], t[2*p+1]);
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


// Lazy prop segtree, array type is T, lazy op is D
// 3 funcs: comb T, comb D, apply D to T (all assoc)
namespace lseg {
	typedef ll T;
	typedef ll D;

	T idT = 0, t[2 * NN];
	D idD = 0, d[NN];
	ll n = (fill_n(d, NN, idD), NN);

	T f(T a, T b) { return a + b; }
	D g(D a, D b) { return a + b; }
	T h(D a, T b) { return a + b; }

	void apply(ll p, D v) {
		t[p] = h(v, t[p]);
		if(p < n) d[p] = g(v, d[p]);
	}

	void modify(ll p, T v = idT) {
		if(p < n) p += n, t[p] = v;
		while(p /= 2) t[p] = h(d[p], f(t[2 * p], t[2 * p + 1]));
	}

	void modify(ll l, ll r, D v) {
		ll l0 = (l += n), r0 = (r += n);
		for(; l < r; l /= 2, r /= 2) {
			if(l & 1) apply(l++, v);
			if(r & 1) apply(--r, v);
		}
		modify(l0), modify(r0-1);
	}

	void push(ll p) {
		for(ll s=26; --s; ) {
			ll i = p >> s;
			apply(2 * i, d[i]);
			apply(2 * i + 1, d[i]);
			d[i] = idD;
		}
	}

	T query(ll l, ll r) {
		l += n, r += n;
		push(l), push(r - 1);
		T resL = idT, resR = idT;
		for(; l < r; l /= 2, r /= 2) {
			if(l & 1) resL = f(resL, t[l++]);
			if(r & 1) resR = f(t[--r], resR);
		}
		return f(resL, resR);
	}
}


//DSU
ll parent[NN], sz[NN];
ll find(ll a){ return a == parent[a] ? a : parent[a] = find(parent[a]); }
void merge(ll u, ll v) {
    u = find(u), v=find(v);
    if (u!=v) {
        if (sz[u]<sz[v]) swap(u, v);
        sz[u] += sz[v];
        parent[v] = u;
    }
}

//matrix multiplication a*b=c
//change data type and operation as needed
typedef vector<vector<ll>> mat;
mat mul(mat &a, mat &b) {
    mat c(a.size(), vector<ll>(b[0].size(), 0));

    for (ll i=0; i<a.size(); ++i)
        for (ll j=0; j<b[0].size(); ++j)
            for (ll k=0; k<b.size(); ++k)
                ( c[i][j] += a[i][k]*b[k][j] )%=M; // or no mod if ld
    return c;
}


//Gaussian elimination with partial pivoting
//Also calculates determinant
ld elim(vector<vector<ld> > &A, vector<ld> &b) {
  int n=A.size();
  ld det=1; //OPTIONAL CALCULATE DET, return ld, not void

  //REF
  for (int i=0;i<n-1;i++) {
    //PIVOT
    int bigi=max_element(A.begin()+i, A.end(), [i](vector<ld> &r1, vector<ld> &r2) {return fabs(r1[i])<fabs(r2[i]);})-A.begin(); 
    swap(A[i], A[bigi]);
    swap(b[i], b[bigi]);
    if (i!=bigi) det*=-1; //DET PART

    for (int j=i+1;j<n;j++) {
      ld m=A[j][i]/A[i][i];
      for (int k=i;k<n;k++)
        A[j][k]-=m*A[i][k];
      b[j]-=m*b[i];
    }
  }

  //DET PART
  for (int i=0;i<n;i++) det*=A[i][i];

  //BACKSUB
  for (int i=n-1;i>=0;i--) {
    for (int j=i+1;j<n;j++)
      b[i]-=A[i][j]*b[j];
    b[i]/=A[i][i];
  }

  return det;
}

//modular inverse a^-1 mod b
ll inv(ll a, ll b){return 1<a ? b - inv(b%a,a)*b/a : 1;}

//GCC extensions
#include <bits/extc++.h>
using namespace __gnu_cxx;
using namespace __gnu_pbds;

// order stats tree
// find_by_order(i) returns ptr
// order_of_key(key) return int
typedef tree<ll, null_type, less<ll>,
rb_tree_tag, tree_order_statistics_node_update> set_t;

//hash table
gp_hash_table<ll, ll>

//pow mod manual
ll powmod(ll x, ll y){
  if(y==0) return 1LL;
  ll t=powmod(x,y/2);
  if (y%2==0) return (t*t)%M;
  return (((x*t)%M)*t)%M;
}

//Chinese Remainder Theorem
ll CRT(vector<ll> &a, vector<ll> &n) {
  ll prod = 1;
  for (auto ni:n) prod*=ni;
 
  ll sm = 0;
  for (int i=0; i<n.size(); i++) {
    ll p = prod/n[i];
    sm += a[i]*inv(p, n[i])*p;
  }
  return sm % prod;
}

//nCk non-pascal
ll comb(ll n, ll k) {
    ld res = 1;
    ld w = 0.01;
    for (ll i = 1; i <= k; ++i) res = res * (n-k+i)/i;
    return (int)(res + w);
}

//numerical minimization
pair<ld, ld> minimize(ld (*f)(ld, ld)) {
	ld x=2, y=2; //initial guess
	ld step=.01;
	ld h=0.00001;
	ld prec=0.000001;
	ld dx=2, dy=2, ldx=2.1, ldy=2.1; //this might be fucky
	for (int i=0;i<10000;i++) {
		ldx=dx;
		ldy=dy;
		ld xd=-dx*step;
		ld yd=-dy*step;
		dx=(f(x+h, y)-f(x-h, y))/h*.5;
		dy=(f(x, y+h)-f(x, y-h))/h*.5;
		if (abs(dx)<prec && abs(dy)<prec) break;
		step=abs(xd*(dx-ldx)+yd*(dy-ldy))/(pow(dx-ldx, 2)+pow(dy-ldy, 2));
		x-=dx*step;
		y-=dy*step;
	}
	return make_pair(x, y);
}


//prime seive
for (ll i=2; i<NN; i++)
	if (prime[i]==0) {
		prime[i] = i;
		for (ll j=i*i;j<NN;j+=i) if(!prime[j]) prime[j]=i;
	}

// phi, uses seive and power from above. The formula is phi(p^i)=(p-1)*p^(c-1).
ll phi(ll n) {
  ll ans = n;
  while (n>1) {
    ll p = prime[n];
    while (n%p==0) n/=p;
    ans = ans/p*(p-1);
  }
  return ans;
}

// suffix tree, NN here is number of nodes, which is like 2n+10
// to[] is edges, root is idx 1
// lf[] and rt[] are edge info as half open interval into s
map<char, ll> to[NN], lk[NN];
ll lf[NN], rt[NN], par[NN], path[NN];
#define att(a, b, c) to[par[a]=b][s[lf[a]=c]]=a;
void build(string &s) {
	ll n=s.size(), z=2;
	lf[1]--;

	for (ll i=n-1; i+1; i--) {
		ll v, V=n, o=z-1, k=0;
		for (v=o; !lk[v].count(s[i]) && v; v=par[v])
			V -= rt[path[k++]=v]-lf[v];

		ll w = lk[v][s[i]]+1;
		if (to[w].count(s[V])) {
			ll u = to[w][s[V]];
			for (rt[z]=lf[u]; s[rt[z]]==s[V]; rt[z]+=rt[v]-lf[v])
				v=path[--k], V+=rt[v]-lf[v];

			att(z, w, lf[u])
			att(u, z, rt[z])
			lk[v][s[i]] = (w = z++)-1;
		}

		lk[o][s[i]] = z-1;
		att(z, w, V)
		rt[z++] = n;
	}
}

// substring hashing
typedef __int128 HASH;
HASH p[NN], h[NN];

constexpr HASH M = 1000000000000000003;
mt19937 gen( __builtin_ia32_rdtsc() );
uniform_int_distribution<ll> dist(256, M-1);

HASH sub_hash(ll l, ll r) {
    return (h[r] - p[r-l]*h[l]%M + M)%M;
}

	p[0] = 1, p[1] = dist(gen);
	for (ll i=0; i<s.size(); ++i) {
	    p[i+1] = p[i]*p[1]%M;
	    h[i+1] = (h[i]*p[1] + s[i])%M;
	}

//digit dp
ll dp[20][2]; //also can have auxiliary info
int r[20]; //number as digit array
int n; //length

#define DP dp[pos][is_eq]
ll solve(int pos, bool is_eq) {
	if (~DP) return DP;
	if (pos==n)
		//check for predicate (here it is p(x)=True)
		return DP=1;

	DP = 0;
	for (int i=0;i<=(is_eq?r[pos]:9);i++)
		DP += solve(pos+1, is_eq && i==r[pos]);

	return DP;
}

//geometry header
typedef complex<ld> Point;
#define x() real()
#define y() imag()
#define cross(a, b) ((conj(a)*(b)).y())
#define dot(a, b) ((conj(a)*(b)).x())

struct Line {
  Point P;
  Point D;

  // Direct constructor
  Line (Point a, Point b, bool q) : P(a), D(b){}

  // Two points
  Line (Point a, Point b) : P(a), D(b-a){}

  // Point and angle
  Line (Point a, ld theta) : P(a), D(polar(1.l, theta)){}
};

Point* intersect(Line a, Line b, bool s1 = false, bool s2 = false) {
  if (fabsl(cross(a.D/abs(a.D), b.D/abs(b.D)))<0.00000001) return NULL;
  ld t = (cross(a.D, a.P) - cross(a.D, b.P)) / cross(a.D, b.D);
  ld s = (cross(b.D, b.P) - cross(b.D, a.P)) / cross(b.D, a.D);
  if (((t<0 || t>1)&&s2) || ((s<0 || s>1)&&s1)) return NULL;
  return new Point(b.P + b.D*t);
}

// First point of each line is common
// TODO: have it point correct MOD PI
Line angle_bi(Line A, Line B) {
  return {A.P, (arg(A.D) + arg(B.D))/2.l};
}

Line perp_bi(Line L) {
  return {(L.D + 2.l*L.P)/2.l, arg(L.D) + M_PI/2};
}

Point closest(Line L, Point A, bool s=false) {
  if (Point *ans = intersect(L, Line(A, L.D*Point(0,1), true), s)) return *ans;
  return abs(L.P-A)<abs(L.P+L.D-A) ? L.P : L.P+L.D;
}

//convex hull
vector<Point> P;
vector<Point> HU, HD;

sort(P.begin(), P.end(), [](Point &a, Point &b) {return a.x()==b.x() ? a.y()<b.y() : a.x()<b.x();});
#define do_hull(H) for (auto p:P) {while (H.size() >= 2 && cross(H.back()-H[H.size()-2], p-H[H.size()-2]) <= 0) H.pop_back();H.push_back(p);}
do_hull(HD);
reverse(P.begin(), P.end());
do_hull(HU);

// For using Point as key in std::set/std::map
namespace std {
    bool operator<(Point a, Point b) {
        return a.x()==b.x() ? a.y()<b.y() : a.x()<b.x();
    }
}

// #define S for MAXN, T is S+1 and use add_edge
struct dinic {
	struct edge {ll b, cap, flow, flip;};
	vector<edge> g[S+2];
	ll ans=0, d[S+2], ptr[S+2];

	void add_edge (ll a, ll b, ll cap) {
		g[a].push_back({b, cap, 0, g[b].size()});
		g[b].push_back({a, 0, 0, g[a].size()-1});
	}
	
	ll dfs (ll u, ll flow=LLONG_MAX) {
		if (u==S+1 || !flow) return flow;
		while (++ptr[u] < g[u].size()) {
			edge &e = g[u][ptr[u]];
			if (d[e.b] != d[u]+1) continue;
			if (ll pushed = dfs(e.b, min(flow, e.cap-e.flow))) {
				e.flow += pushed;
				g[e.b][e.flip].flow -= pushed;
				return pushed;
			}
		}
		return 0;
	}

	void calc() {
		do {
			vector<ll> q {S};
			memset(d, 0, sizeof d);
			ll i = -(d[S] = 1);
			while (++i<q.size() && !d[S+1])
				for (auto e: g[q[i]])
					if (!d[e.b] && e.flow<e.cap) {
						q.push_back(e.b);
						d[e.b] = d[q[i]]+1;
					}

			memset(ptr, -1, sizeof ptr);
			while(ll pushed=dfs(S)) ans+=pushed;
		} while (d[S+1]);
	}
};


// Kuhn's bipartite matching O(|left|*m)
// mat[right node] is a left node or -1
// edges from left to right
bool dfs(ll u) {
	if (used[u]) return 0;
	used[u] = 1;
	for (ll v: edges[u])
	if (mat[v]==-1 || dfs(mat[v]))
		return mat[v] = u,1;
	return 0;
}

	memset(mat, -1, sizeof mat);
	for (ll u=0; u<n; ++u) {
		memset(used, 0, sizeof used);
		flow += dfs(u);
	}


// 2-SAT
// n is total number of vertices - need 2*# of terms for !X
// directed edges mean implication
// For each term need contrapositive: A=>B needs to also have !B=>!A
// To setup, make the graph g, and the graph gt with all the edges flipped
// TODO: I removed the actual implementation, because it was just maxx-ru copy/paste basically, get a good one

//SOS dp, init F[mask] to og array
for(ll i=0; i<N; ++i)
	for(ll mask=0; mask<(1<<N); ++mask)
		if(mask & (1<<i)) F[mask] += F[mask^(1<<i)];


// Mo's algo, O(q*S+n*n/S), q is query idx's, [l,r] closed
// update(idx, +-1) adds/dels to some DS (tot is cur ans)
ll S = sqrtl(n);
sort(q.begin(), q.end(), [&](ll a, ll b) {
	if (l[a]/S != l[b]/S) return l[a]/S < l[b]/S;
	return l[a]/S%2 ? r[a]>r[b] : r[a]<r[b];
});

ll curl=0, curr=-1;
for (auto i:q) {
	while(curr<r[i]) update(++curr, 1);
	while(curr>r[i]) update(curr--, -1);
	while(curl<l[i]) update(curl++, -1);
	while(curl>l[i]) update(--curl, 1);

	ans[i] = tot;
}


// tree flattening: half open tin/tout
ll dfs(ll u, ll p, ll t) {
    tout[u] = (tin[u]=t)+1;

    for (auto v:edges[u]) if (v!=p)
        tout[u] = dfs(v, u, tout[u]);

    return tout[u];
}


// invariant binary search
ll l = -1; //value that always works
ll r = n; //value that never works
while (r-l > 1) {
	ll m = (l+r)/2;

	if (f(m))
	    l = m;
	else
	    r = m;
} // ans is now l

//how to spell
setprecision(4);