#pragma GCC optimize("O3")
#pragma GCC optimize("unroll-loops")
#include <vector>
#pragma GCC target("avx2")

#include <bits/stdc++.h>
#include <math.h>
using namespace std;

typedef long long int ll;
typedef long double ld;
typedef pair<ll, ll> pl;
typedef vector<ll> vl;
#define FD(i, r, l) for (ll i = r; i > (l); --i)

#define K first
#define V second
#define G(x)                                                                   \
	ll x;                                                                      \
	cin >> x;
#define GD(x)                                                                  \
	ld x;                                                                      \
	cin >> x;
#define GS(s)                                                                  \
	string s;                                                                  \
	cin >> s;
#define EX(x)                                                                  \
	{                                                                          \
		cout << x << '\n';                                                     \
		exit(0);                                                               \
	}
#define A(a) (a).begin(), (a).end()
#define F(i, l, r) for (ll i = l; i < (r); ++i)

template <typename A, typename B> A &chmax(A &a, B b) {
	return a < b ? (a = b) : a;
}

template <typename A, typename B> A &chmin(A &a, B b) {
	return a > b ? (a = b) : a;
}

using vi = vector<int>;
#define rep(l, r, n) F(l, r, n)

string f(string s) {
	//    freopen("a.in", "r", stdin);
	//    freopen("a.out", "w", stdout);



	int n = s.size();

	string ans;
	F(i, 0, n) ans += "#";
	auto update_ans = [&](auto x) {
		int v = count(A(ans), '#');
		int v2 = count(A(x), '#');
		if (v2 < v)
			ans = x;
	};
	if (n == 1) {
		EX("1\n#\n")
	}
	for (int prefix = 0; prefix <= n; ++prefix)
		for (int suffix_temp = 0; prefix + suffix_temp < n; suffix_temp++) {
			if (s[prefix] == '.' and s[n - suffix_temp - 1] == '.')
				;
			else
				continue;
			int left = n - prefix - suffix_temp;
			string tans = s;
			F(i, 0, prefix) tans[i] = '#';
			F(i, 0, suffix_temp) tans[n - i - 1] = '#';
			if (left == 1)
				update_ans(tans);
			else if (left % 2 ==
					 0) // can prove that bads are on alternating
						// diagonals so trivial to show that all must be filled
			{
				if (left == 2)
					continue;
				for (int i = prefix + 1; i < n - suffix_temp; ++i)
					tans[i] = '#';
				update_ans(tans);
				continue;
			} else {
				int suffix = n - suffix_temp;
				set<pl> edges;
				vector<vector<int>> adj(n);
				vector<int> vis(n);
				for (int i = prefix + 1; i < suffix; i += 2)
					tans[i] = '#';
				for (int wanty = prefix + 2; wanty < suffix; wanty += 2) {
					for (int col = prefix + 1; col < suffix; ++col)
						if (prefix + col == 2 * wanty) {
							if (tans[col] == '.' && tans[wanty] == '.')
								edges.emplace(min(col, wanty), max(col, wanty));
						}
					for (int row = prefix + 1; row < suffix; ++row)
						if (suffix + row == 2 * wanty) {
							if (tans[row] == '.' && tans[wanty] == '.')
								edges.emplace(min(row, wanty), max(row, wanty));
						}
				}
				for (auto [u, v] : edges)
					adj[u].push_back(v), adj[v].push_back(u);
				vector<array<ll, 2>> dp(n);
				for (int i = 0; i < n; ++i)
					if (!vis[i] and not adj[i].empty()) {
						vi ans;
						// find minimum set of nodes in this cc so every edge
						// has a node next to it
						function<void(int, int)> dfs = [&](int i, int p) {
							vis[i] = 1;
							int kids = 0;
							for (int j : adj[i])
								if (j - p)
									dfs(j, i), kids++;
							// if we pick the node
							dp[i][1] = 1;
							for (int j : adj[i])
								if (j - p)
									dp[i][1] += min(dp[j][0], dp[j][1]);
							for (int j : adj[i])
								if (j - p)
									dp[i][0] += dp[j][1];
						};
						dfs(i, -1);
						function<void(int, int, int)> rec = [&](int i, int p,
																int x) {
							if (x)
								ans.push_back(i);

							if (x) {
								for (int j : adj[i])
									if (j - p) {
										if (dp[j][0] < dp[j][1])
											rec(j, i, 0);
										else
											rec(j, i, 1);
									}
							} else {
								for (int j : adj[i])
									if (j - p) {
										rec(j, i, 1);
									}
							}
						};
						if (dp[i][0] < dp[i][1])
							rec(i, -1, 0);
						else
							rec(i, -1, 1);
						for (auto x : ans)
							tans[x] = '#';
					}

				update_ans(tans);
			}
		}
	return ans;
}

int main(){
	ios_base::sync_with_stdio(false);
	cin.tie(0);
	cout << fixed << setprecision(20);
    string s; cin >> s;

    string ans(size(s),'#');
    for(int i = 0;i<size(s)/2;i++){
        char old = s[i];
        s[i]='#';
        auto res = f(s);
        s[i]=old;
        if(count(A(s),'#') < count(A(ans),'#')) ans = res;
    }
    cout << count(A(ans),'#') << '\n';
    cout << ans << '\n';
}