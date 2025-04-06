#include "bits/stdc++.h"
using ll = long long;
using namespace std;

const ll S = 5e4;
const ll source = S+2,sink=S+3; //use these
struct Demands_Dinic {
    const ll demands_source = S,demands_sink = S+1;
	struct edge {ll b, cap, flow, flip;};
	vector<edge> g[S+4];
	ll ans=0, d[S+4], ptr[S+4];
	
	map<ll,ll> out_min_flow,in_min_flow;

    void true_add_edge(ll a,ll b,ll cap){
		if(not cap) return; //dont add redundant edges
		g[a].push_back({b, cap, 0, (int)g[b].size()});
		g[b].push_back({a, 0, 0, (int)g[a].size()-1});
    }

    Demands_Dinic() {
        true_add_edge(sink,source,LLONG_MAX);   
    }

	void add_edge (ll a, ll b, ll min_cap,ll max_cap) {
        true_add_edge(a,b,max_cap-min_cap);
		out_min_flow[a] += min_cap;
		in_min_flow[b] += min_cap;
	}
	
	ll dfs (ll u, ll flow=LLONG_MAX) {
		if (u==S+1 || !flow) return flow;
		while (++ptr[u] < (int)g[u].size()) {
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
		for(auto [u,flow] : out_min_flow)
			true_add_edge(u,demands_sink,flow);
		for(auto [u,flow] : in_min_flow)
			true_add_edge(demands_source,u,flow);
		out_min_flow.clear(),in_min_flow.clear();
		do {
			vector<ll> q {S};
			memset(d, 0, sizeof d);
			ll i = -(d[S] = 1);
			while (++i<(int)q.size() && !d[S+1])
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



