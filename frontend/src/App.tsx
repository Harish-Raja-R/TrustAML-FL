import React, { useState, useEffect } from 'react';
import { Activity, ShieldAlert, Network, Settings, Database, ActivitySquare, Server, Lock } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

function App() {
  const [status, setStatus] = useState<any>(null);
  const [experiments, setExperiments] = useState<any>(null);
  const [alerts, setAlerts] = useState<any[]>([]);

  useEffect(() => {
    fetch('http://localhost:8000/api/system/status')
      .then(res => res.json())
      .then(data => setStatus(data))
      .catch(err => console.error(err));
      
    fetch('http://localhost:8000/api/experiments')
      .then(res => res.json())
      .then(data => setExperiments(data))
      .catch(err => console.error(err));
      
    fetch('http://localhost:8000/api/alerts')
      .then(res => res.json())
      .then(data => setAlerts(data))
      .catch(err => console.error(err));
  }, []);

  const getGlobalPRAUC = () => {
    if (!experiments || !experiments.fedavg) return "Not evaluated";
    const history = experiments.fedavg;
    if (history.length === 0) return "Not evaluated";
    return history[history.length - 1].pr_auc.toFixed(4);
  };
  
  const getPrivacyStatus = () => {
    if (!experiments || !experiments.privacy) return "Data unavailable";
    const weakDp = experiments.privacy.weak_dp;
    if (!weakDp) return "Data unavailable";
    if (weakDp.privacy_status && weakDp.privacy_status.status === "not_computed") {
      return "Not formally accounted";
    }
    return `Epsilon: ${weakDp.privacy_status.epsilon}`;
  };

  const getDriftMetric = () => {
    if (!experiments || !experiments.drift || !experiments.drift.natural_temporal_windows) return "Not evaluated";
    const drift = experiments.drift.natural_temporal_windows;
    if (!drift.Window_5) return "Not evaluated";
    return `KL-Div: ${drift.Window_5.feature_0.kl.toFixed(4)}`;
  };

  const parseConvergenceData = () => {
    if (!experiments || !experiments.fedavg) return [];
    return experiments.fedavg.map((round: any) => ({
      round: round.round,
      pr_auc: round.pr_auc,
      roc_auc: round.roc_auc
    }));
  };

  const convergenceData = parseConvergenceData();

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
                <p className="text-gray-400 text-sm font-medium">Global PR-AUC (FedAvg)</p>
                <h3 className="text-2xl font-bold text-white mt-2">{getGlobalPRAUC()}</h3>
              </div>
              <ActivitySquare className="text-emerald-500 w-6 h-6" />
            </div>
          </div>
          
          <div className="bg-gray-900 p-6 rounded-xl border border-gray-800 flex flex-col justify-between hover:border-cyan-500/50 transition-colors">
            <div className="flex justify-between items-start">
              <div>
                <p className="text-gray-400 text-sm font-medium">Active Banks</p>
                <h3 className="text-2xl font-bold text-white mt-2">
                   {experiments && experiments.fedavg && experiments.fedavg.length > 0 ? experiments.fedavg[0].participating_clients : "N/A"}
                </h3>
              </div>
              <Network className="text-cyan-500 w-6 h-6" />
            </div>
            <p className="text-cyan-500 text-sm mt-4">FedAvg Aggregation</p>
          </div>

          <div className="bg-gray-900 p-6 rounded-xl border border-gray-800 flex flex-col justify-between hover:border-purple-500/50 transition-colors">
            <div className="flex justify-between items-start">
              <div>
                <p className="text-gray-400 text-sm font-medium">Privacy Status</p>
                <h3 className="text-xl font-bold text-white mt-2">{getPrivacyStatus()}</h3>
              </div>
              <Lock className="text-purple-500 w-6 h-6" />
            </div>
          </div>

          <div className="bg-gray-900 p-6 rounded-xl border border-gray-800 flex flex-col justify-between hover:border-orange-500/50 transition-colors">
            <div className="flex justify-between items-start">
              <div>
                <p className="text-gray-400 text-sm font-medium">Temporal Drift (Win 5)</p>
                <h3 className="text-xl font-bold text-white mt-2">{getDriftMetric()}</h3>
              </div>
              <Database className="text-orange-500 w-6 h-6" />
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          <div className="bg-gray-900 p-6 rounded-xl border border-gray-800">
            <h3 className="text-xl font-semibold mb-6">Federated Convergence (PR-AUC)</h3>
            <div className="h-72">
              {convergenceData.length > 0 ? (
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={convergenceData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis dataKey="round" stroke="#9CA3AF" />
                    <YAxis stroke="#9CA3AF" domain={[0, 1.0]} />
                    <Tooltip contentStyle={{ backgroundColor: '#1F2937', border: 'none', borderRadius: '0.5rem' }} />
                    <Legend />
                    <Line type="monotone" dataKey="pr_auc" stroke="#10B981" strokeWidth={3} dot={{ r: 4 }} activeDot={{ r: 6 }} name="FedAvg Global Model" />
                  </LineChart>
                </ResponsiveContainer>
              ) : (
                <div className="flex h-full items-center justify-center text-gray-500">Data unavailable</div>
              )}
            </div>
          </div>
          
          <div className="bg-gray-900 p-6 rounded-xl border border-gray-800">
            <h3 className="text-xl font-semibold mb-6">Recent AML Alerts (From Backend)</h3>
            <div className="space-y-4 overflow-y-auto h-72 pr-2">
              {alerts.length > 0 ? alerts.map((alert: any, i: number) => (
                <div key={i} className="bg-gray-950 p-4 rounded-lg border border-gray-800 hover:border-gray-700 transition-colors flex justify-between items-center">
                  <div>
                    <div className="flex items-center space-x-2">
                      <span className="font-mono text-sm text-gray-300">{alert.transaction_id}</span>
                      <span className={`text-xs px-2 py-0.5 rounded-full border ${
                        alert.risk_level === 'CRITICAL' ? 'bg-red-900/30 text-red-400 border-red-900/50' : 
                        alert.risk_level === 'HIGH' ? 'bg-orange-900/30 text-orange-400 border-orange-900/50' : 
                        'bg-yellow-900/30 text-yellow-400 border-yellow-900/50'
                      }`}>
                        {alert.risk_level}
                      </span>
                    </div>
                    <ul className="text-sm text-gray-500 mt-1 list-disc list-inside">
                      {alert.reasons.map((reason: string, idx: number) => <li key={idx}>{reason}</li>)}
                    </ul>
                  </div>
                  <div className="text-right">
                    <div className="text-xl font-bold text-emerald-400">Risk: {alert.risk_score.toFixed(2)}</div>
                    <div className="text-xs text-gray-500">Bank: {alert.bank}</div>
                  </div>
                </div>
              )) : (
                <div className="flex h-full items-center justify-center text-gray-500">No alerts found</div>
              )}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;
