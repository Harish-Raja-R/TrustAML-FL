import React, { useState, useEffect } from 'react';
import { Activity, ShieldAlert, Network, Settings, Database, ActivitySquare, Server, Lock } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

function App() {
  const [status, setStatus] = useState<any>(null);

  useEffect(() => {
    fetch('http://localhost:8000/api/system/status')
      .then(res => res.json())
      .then(data => setStatus(data))
      .catch(err => console.error(err));
  }, []);

  const mockConvergenceData = [
    { round: 1, pr_auc: 0.65 },
    { round: 5, pr_auc: 0.72 },
    { round: 10, pr_auc: 0.81 },
    { round: 15, pr_auc: 0.84 },
    { round: 20, pr_auc: 0.86 },
  ];

  return (
    <div className="min-h-screen bg-gray-950 text-gray-200">
      <nav className="bg-gray-900 border-b border-gray-800 p-4 sticky top-0 z-50">
        <div className="flex items-center justify-between max-w-7xl mx-auto">
          <div className="flex items-center space-x-3">
            <ShieldAlert className="text-emerald-500 w-8 h-8" />
            <h1 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-emerald-400 to-cyan-500">
              TrustAML-FL
            </h1>
          </div>
          <div className="flex space-x-4">
            <div className="flex items-center space-x-2 text-sm text-gray-400">
              <Server className="w-4 h-4" />
              <span>Backend: {status ? 'Online' : 'Offline'}</span>
            </div>
            {status && (
              <div className="flex items-center space-x-2 text-sm text-emerald-400">
                <Activity className="w-4 h-4" />
                <span>CPU: {status.cpu_percent}%</span>
              </div>
            )}
          </div>
        </div>
      </nav>

      <main className="p-8 max-w-7xl mx-auto space-y-8">
        <header>
          <h2 className="text-3xl font-semibold mb-2">Research Dashboard</h2>
          <p className="text-gray-400">Privacy-Preserving Federated Heterogeneous Graph Learning for Cross-Bank AML Detection</p>
        </header>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <div className="bg-gray-900 p-6 rounded-xl border border-gray-800 flex flex-col justify-between hover:border-emerald-500/50 transition-colors">
            <div className="flex justify-between items-start">
              <div>
                <p className="text-gray-400 text-sm font-medium">Global PR-AUC</p>
                <h3 className="text-3xl font-bold text-white mt-2">0.864</h3>
              </div>
              <ActivitySquare className="text-emerald-500 w-6 h-6" />
            </div>
            <p className="text-emerald-500 text-sm mt-4">+0.012 vs local best</p>
          </div>
          
          <div className="bg-gray-900 p-6 rounded-xl border border-gray-800 flex flex-col justify-between hover:border-cyan-500/50 transition-colors">
            <div className="flex justify-between items-start">
              <div>
                <p className="text-gray-400 text-sm font-medium">Active Banks</p>
                <h3 className="text-3xl font-bold text-white mt-2">3 / 3</h3>
              </div>
              <Network className="text-cyan-500 w-6 h-6" />
            </div>
            <p className="text-cyan-500 text-sm mt-4">FedAvg Aggregation</p>
          </div>

          <div className="bg-gray-900 p-6 rounded-xl border border-gray-800 flex flex-col justify-between hover:border-purple-500/50 transition-colors">
            <div className="flex justify-between items-start">
              <div>
                <p className="text-gray-400 text-sm font-medium">Privacy Status</p>
                <h3 className="text-3xl font-bold text-white mt-2">DP Enabled</h3>
              </div>
              <Lock className="text-purple-500 w-6 h-6" />
            </div>
            <p className="text-purple-500 text-sm mt-4">ε ≈ 2.1, δ = 1e-5</p>
          </div>

          <div className="bg-gray-900 p-6 rounded-xl border border-gray-800 flex flex-col justify-between hover:border-orange-500/50 transition-colors">
            <div className="flex justify-between items-start">
              <div>
                <p className="text-gray-400 text-sm font-medium">Temporal Drift</p>
                <h3 className="text-3xl font-bold text-white mt-2">Stable</h3>
              </div>
              <Database className="text-orange-500 w-6 h-6" />
            </div>
            <p className="text-orange-500 text-sm mt-4">KL-Div: 0.04</p>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          <div className="bg-gray-900 p-6 rounded-xl border border-gray-800">
            <h3 className="text-xl font-semibold mb-6">Federated Convergence (PR-AUC)</h3>
            <div className="h-72">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={mockConvergenceData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis dataKey="round" stroke="#9CA3AF" />
                  <YAxis stroke="#9CA3AF" domain={[0.5, 1.0]} />
                  <Tooltip contentStyle={{ backgroundColor: '#1F2937', border: 'none', borderRadius: '0.5rem' }} />
                  <Legend />
                  <Line type="monotone" dataKey="pr_auc" stroke="#10B981" strokeWidth={3} dot={{ r: 4 }} activeDot={{ r: 6 }} name="FedAvg Global Model" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
          
          <div className="bg-gray-900 p-6 rounded-xl border border-gray-800">
            <h3 className="text-xl font-semibold mb-6">Recent AML Alerts</h3>
            <div className="space-y-4 overflow-y-auto h-72 pr-2">
              {[
                { id: 'AML_f8a92b11', score: 0.94, bank: 'Bank_1', reason: 'Unusually high transaction amount' },
                { id: 'AML_c3d4e5f6', score: 0.88, bank: 'Bank_2', reason: 'Suspicious fan-out pattern' },
                { id: 'AML_1a2b3c4d', score: 0.81, bank: 'Bank_1', reason: 'Cross-border chain to high-risk jurisdiction' },
                { id: 'AML_9e8d7c6b', score: 0.76, bank: 'Bank_3', reason: 'Structuring below threshold' },
              ].map((alert, i) => (
                <div key={i} className="bg-gray-950 p-4 rounded-lg border border-gray-800 hover:border-gray-700 transition-colors flex justify-between items-center">
                  <div>
                    <div className="flex items-center space-x-2">
                      <span className="font-mono text-sm text-gray-300">{alert.id}</span>
                      <span className="text-xs px-2 py-0.5 rounded-full bg-red-900/30 text-red-400 border border-red-900/50">Critical</span>
                    </div>
                    <p className="text-sm text-gray-500 mt-1">{alert.reason}</p>
                  </div>
                  <div className="text-right">
                    <div className="text-xl font-bold text-red-500">{alert.score}</div>
                    <div className="text-xs text-gray-500">{alert.bank}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;
