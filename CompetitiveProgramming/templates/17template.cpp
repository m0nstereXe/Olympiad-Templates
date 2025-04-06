#include <bits/stdc++.h>
#include <ext/pb_ds/assoc_container.hpp>

using namespace std;
using namespace __gnu_pbds;

#define A(a) begin(a),end(a)

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
using vll =  vector<ll>;
using pll = pair<ll,ll>;
using pii = pair<int,int>;

constexpr ll NN = 2e5, M = 1000000007, L = 20;

ll f1[NN], f2[NN];
ll inv(ll a, ll b=M) { return 1 < a ? b - inv(b % a, a) * b / a : 1; } // inv a mod b
ll choose(ll n, ll k) { return f1[n] * f2[k] % M * f2[n - k] % M; } // n choose k

void run()
{
    
}

int main()
{
    //KING OF THE WORLD...... U.W.T.B
    cin.tie(0);
    ios_base::sync_with_stdio(false);
    f1[0] = 1;
    f2[0] = 1;
    for (int i = 1; i < NN; i++) {
        f1[i] = f1[i - 1] * i % M;
        f2[i] = inv(f1[i], M);
    }
    int t; cin>>t; while(t--) run();
}
