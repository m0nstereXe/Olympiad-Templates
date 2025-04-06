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

constexpr ll NN = 2e5, M = 1000000007, L = 20;

std::vector<std::string> split(const std::string &s, char separator) {
    std::vector<std::string> result;
    std::string item;
    for (size_t i = 0; i < s.length(); i++)
        if (s[i] == separator) {
            result.push_back(item);
            item = "";
        } else
            item += s[i];
    result.push_back(item);
    return result;
}

/**
 * Splits string s by character separators returning exactly k+1 items,
 * where k is the number of separator occurences.
 */
std::vector<std::string> split(const std::string &s, const std::string &separators) {
    if (separators.empty())
        return std::vector<std::string>(1, s);

    std::vector<bool> isSeparator(256);
    for (size_t i = 0; i < separators.size(); i++)
        isSeparator[(unsigned char) (separators[i])] = true;

    std::vector<std::string> result;
    std::string item;
    for (size_t i = 0; i < s.length(); i++)
        if (isSeparator[(unsigned char) (s[i])]) {
            result.push_back(item);
            item = "";
        } else
            item += s[i];
    result.push_back(item);
    return result;
}

bool good(int x,int y,int n,int m){
    return x >= 0 && x < n && y >= 0 && y < m;
}

vector<pii> delta1 = {{1,0},{0,1},{-1,0},{0,-1}};
vector<pii> delta2 = {{1,1},{1,-1},{-1,1},{-1,-1},{1,0},{0,1},{-1,0},{0,-1}};

vll readarr(char delim){
    string s; getline(cin,s);
    auto v = split(s,delim);
    vll res;
    for(auto x : v) res.push_back(stoll(x));
    return res;
}

void run()
{
    vi quad(4);

    string line; while(getline(cin,line)){
        auto res = split(line," ,=");
        ll sx = stoll(res[1]),sy=stoll(res[2]);
        ll dx = stoll(res[3]),dy=stoll(res[4]);

        cout << sx << ' ' << sy << ' ' << dx << ' ' << dy << '\n';
        exit(0);
    }

    int ans = 1; for(int x : quad) ans *= x;
    cout << ans << '\n';
}

int main()
{
    //KING OF THE WORLD...... U.W.T.B
    cin.tie(0);
    ios_base::sync_with_stdio(false);
    run();
}
