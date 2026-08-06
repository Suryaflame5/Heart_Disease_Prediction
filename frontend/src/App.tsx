import React, { useState, useEffect } from 'react';
import { 
  Activity, BarChart2, Layers, History, Settings, Upload, Download, Trash, RefreshCw, AlertCircle, CheckCircle, Search, Filter 
} from 'lucide-react';
import { 
  AreaChart, Area, BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer 
} from 'recharts';

export default function App() {
  const [activeTab, setActiveTab] = useState<'predict' | 'batch' | 'analytics' | 'history' | 'settings'>('predict');
  const [formData, setFormData] = useState<any>({ Age: 55, Sex: "Male", ChestPainType: "Typical Angina", MaxHeartRate: 140, RestBP: 120, Cholesterol: 220, ST_Depression: 1.2 });
  
  // States
  const [prediction, setPrediction] = useState<any>(null);
  const [latency, setLatency] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [toast, setToast] = useState<{ message: string; type: 'success' | 'error' } | null>(null);
  
  const [history, setHistory] = useState<any[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  
  const [metrics, setMetrics] = useState<any>(null);
  const [retraining, setRetraining] = useState(false);
  
  const [batchFile, setBatchFile] = useState<File | null>(null);
  const [batchLoading, setBatchLoading] = useState(false);
  const [batchDownloadUrl, setBatchDownloadUrl] = useState<string | null>(null);

  const showToast = (message: string, type: 'success' | 'error') => {
    setToast({ message, type });
    setTimeout(() => setToast(null), 3000);
  };

  const handlePredict = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData)
      });
      const result = await response.json();
      if (result.success) {
        setPrediction(result.prediction);
        setLatency(result.latency_ms);
        showToast('Prediction generated successfully!', 'success');
        fetchHistory();
      } else {
        showToast(result.error || 'Prediction failed', 'error');
      }
    } catch (err: any) {
      showToast(err.message || 'Server connection failed', 'error');
    } finally {
      setLoading(false);
    }
  };

  const fetchHistory = async () => {
    try {
      const response = await fetch('/api/history');
      const data = await response.json();
      setHistory(data);
    } catch (err) {
      console.error('Failed to load history', err);
    }
  };

  const handleClearHistory = async () => {
    if (!confirm('Are you sure you want to purge all records?')) return;
    try {
      const response = await fetch('/api/clear-history', { method: 'POST' });
      if (response.ok) {
        showToast('History database purged.', 'success');
        setHistory([]);
      }
    } catch (err) {
      showToast('Purging failed', 'error');
    }
  };

  const fetchMetrics = async () => {
    try {
      const response = await fetch('/api/metrics');
      const data = await response.json();
      setMetrics(data);
    } catch (err) {
      console.error('Failed to load model metrics', err);
    }
  };

  const handleRetrain = async () => {
    setRetraining(true);
    showToast('Triggering pipeline retraining...', 'success');
    try {
      const response = await fetch('/api/train', { method: 'POST' });
      const data = await response.json();
      if (data.success) {
        setMetrics(data.metrics);
        showToast('Pipeline training complete! Model updated.', 'success');
      }} else {
        showToast(data.error || 'Pipeline retraining failed', 'error');
      }
    } catch (err: any) {
      showToast(err.message || 'Pipeline training failed', 'error');
    } finally {
      setRetraining(false);
    }
  };

  const handleBatchPredict = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!batchFile) return;
    setBatchLoading(true);
    const fd = new FormData();
    fd.append('file', batchFile);
    try {
      const response = await fetch('/api/predict-batch', {
        method: 'POST',
        body: fd
      });
      if (response.ok) {
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        setBatchDownloadUrl(url);
        showToast('Batch predictions created!', 'success');
        fetchHistory();
      } else {
        const err = await response.json();
        showToast(err.error || 'Batch predictions failed', 'error');
      }
    } catch (err) {
      showToast('Batch run failed', 'error');
    } finally {
      setBatchLoading(false);
    }
  };

  useEffect(() => {
    fetchHistory();
    fetchMetrics();
  }, []);

  const filteredHistory = history.filter(row => {
    return JSON.stringify(row.input_data).toLowerCase().includes(searchQuery.toLowerCase());
  });

  const latencyData = history.slice(0, 10).reverse().map((row, i) => ({
    run: i + 1,
    latency: row.latency_ms
  }));

  return (
    <div className="min-h-screen flex flex-col relative text-white bg-slate-950">
      
      {/* Background radial effects */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-[-10%] left-[-10%] w-[50%] h-[50%] bg-primary/10 rounded-full blur-[120px]" />
        <div className="absolute bottom-[-10%] right-[-10%] w-[50%] h-[50%] bg-accent/10 rounded-full blur-[120px]" />
      </div>

      {/* Toast Alert Banner */}
      {toast && (
        <div className="fixed top-5 right-5 z-50 flex items-center gap-2 p-4 rounded-lg shadow-2xl glass-panel animate-bounce">
          {toast.type === 'success' ? <CheckCircle className="text-emerald-400" /> : <AlertCircle className="text-rose-400" />}
          <span className="text-sm font-semibold">{toast.message}</span>
        </div>
      )}

      {/* Header Dashboard Nav */}
      <header className="glass-panel sticky top-0 z-30 px-6 py-4 flex flex-col md:flex-row justify-between items-center border-b border-white/5 backdrop-blur-md">
        <div className="flex items-center gap-3">
          <div className="h-10 w-10 rounded-xl bg-primary flex items-center justify-center text-slate-900 font-extrabold shadow-md shadow-primary/20">
            <Activity size={22} />
          </div>
          <div>
            <h1 className="text-xl font-bold tracking-tight text-white">Heart Disease Prediction</h1>
            <p className="text-xs text-slate-400 font-medium">ECG Monitor Intelligence Portal</p>
          </div>
        </div>
        
        {/* Navigation tabs */}
        <nav className="flex gap-2 mt-4 md:mt-0 bg-slate-900/60 p-1 rounded-xl border border-white/5">
          <button 
            onClick={() => setActiveTab('predict')}
            className={`flex items-center gap-1.5 px-4 py-2 rounded-lg text-xs font-semibold transition-all duration-300 ${activeTab === 'predict' ? 'bg-primary text-slate-900 shadow-md shadow-primary/20' : 'text-slate-400 hover:text-white'}`}
          >
            <Layers size={14} /> Inference
          </button>
          <button 
            onClick={() => setActiveTab('batch')}
            className={`flex items-center gap-1.5 px-4 py-2 rounded-lg text-xs font-semibold transition-all duration-300 ${activeTab === 'batch' ? 'bg-primary text-slate-900 shadow-md shadow-primary/20' : 'text-slate-400 hover:text-white'}`}
          >
            <Upload size={14} /> Batch CSV
          </button>
          <button 
            onClick={() => setActiveTab('analytics')}
            className={`flex items-center gap-1.5 px-4 py-2 rounded-lg text-xs font-semibold transition-all duration-300 ${activeTab === 'analytics' ? 'bg-primary text-slate-900 shadow-md shadow-primary/20' : 'text-slate-400 hover:text-white'}`}
          >
            <BarChart2 size={14} /> Analytics
          </button>
          <button 
            onClick={() => setActiveTab('history')}
            className={`flex items-center gap-1.5 px-4 py-2 rounded-lg text-xs font-semibold transition-all duration-300 ${activeTab === 'history' ? 'bg-primary text-slate-900 shadow-md shadow-primary/20' : 'text-slate-400 hover:text-white'}`}
          >
            <History size={14} /> Audit Trail
          </button>
          <button 
            onClick={() => setActiveTab('settings')}
            className={`flex items-center gap-1.5 px-4 py-2 rounded-lg text-xs font-semibold transition-all duration-300 ${activeTab === 'settings' ? 'bg-primary text-slate-900 shadow-md shadow-primary/20' : 'text-slate-400 hover:text-white'}`}
          >
            <Settings size={14} /> Pipeline Config
          </button>
        </nav>
      </header>

      {/* Main Panel Content */}
      <main className="flex-1 p-6 max-w-7xl mx-auto w-full relative z-10">
        
        {/* Tab 1: Single Prediction */}
        {activeTab === 'predict' && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            
            {/* Form inputs */}
            <div className="lg:col-span-2 glass-panel p-6 rounded-2xl">
              <h2 className="text-lg font-bold mb-4 text-white border-b border-white/5 pb-2">Single Feature Inference</h2>
              <form onSubmit={handlePredict} className="grid grid-cols-1 md:grid-cols-2 gap-4">
                
            <div className="mb-4">
              <div className="flex justify-between mb-1">
                <label className="text-sm font-medium text-slate-300">Age</label>
                <span className="text-xs font-semibold text-primary bg-primary/20 px-2 py-0.5 rounded">{formData.Age}</span>
              </div>
              <input
                type="range"
                min="20"
                max="90"
                step="1"
                className="w-full accent-primary bg-slate-800 rounded-lg appearance-none h-2 cursor-pointer"
                value={formData.Age}
                onChange={(e) => setFormData({ ...formData, Age: Number(e.target.value) })}
              />
              <div className="flex justify-between text-[10px] text-slate-500 mt-1">
                <span>20</span>
                <span>90</span>
              </div>
            </div>
            

            <div className="mb-4">
              <label className="block text-sm font-medium mb-2 text-slate-300">Sex</label>
              <select
                className="w-full p-3 rounded-lg glass-input text-white focus:ring-2 focus:ring-primary bg-slate-900"
                value={formData.Sex}
                onChange={(e) => setFormData({ ...formData, Sex: e.target.value })}
              >
                <option value="Male">Male</option><option value="Female">Female</option>
              </select>
            </div>
            

            <div className="mb-4">
              <label className="block text-sm font-medium mb-2 text-slate-300">Chest Pain Type</label>
              <select
                className="w-full p-3 rounded-lg glass-input text-white focus:ring-2 focus:ring-primary bg-slate-900"
                value={formData.ChestPainType}
                onChange={(e) => setFormData({ ...formData, ChestPainType: e.target.value })}
              >
                <option value="Typical Angina">Typical Angina</option><option value="Atypical Angina">Atypical Angina</option><option value="Non-Anginal">Non-Anginal</option><option value="Asymptomatic">Asymptomatic</option>
              </select>
            </div>
            

            <div className="mb-4">
              <div className="flex justify-between mb-1">
                <label className="text-sm font-medium text-slate-300">Max Heart Rate Achieved (bpm)</label>
                <span className="text-xs font-semibold text-primary bg-primary/20 px-2 py-0.5 rounded">{formData.MaxHeartRate}</span>
              </div>
              <input
                type="range"
                min="80"
                max="220"
                step="1"
                className="w-full accent-primary bg-slate-800 rounded-lg appearance-none h-2 cursor-pointer"
                value={formData.MaxHeartRate}
                onChange={(e) => setFormData({ ...formData, MaxHeartRate: Number(e.target.value) })}
              />
              <div className="flex justify-between text-[10px] text-slate-500 mt-1">
                <span>80</span>
                <span>220</span>
              </div>
            </div>
            

            <div className="mb-4">
              <div className="flex justify-between mb-1">
                <label className="text-sm font-medium text-slate-300">Resting Blood Pressure (mmHg)</label>
                <span className="text-xs font-semibold text-primary bg-primary/20 px-2 py-0.5 rounded">{formData.RestBP}</span>
              </div>
              <input
                type="range"
                min="80"
                max="200"
                step="1"
                className="w-full accent-primary bg-slate-800 rounded-lg appearance-none h-2 cursor-pointer"
                value={formData.RestBP}
                onChange={(e) => setFormData({ ...formData, RestBP: Number(e.target.value) })}
              />
              <div className="flex justify-between text-[10px] text-slate-500 mt-1">
                <span>80</span>
                <span>200</span>
              </div>
            </div>
            

            <div className="mb-4">
              <div className="flex justify-between mb-1">
                <label className="text-sm font-medium text-slate-300">Serum Cholesterol (mg/dL)</label>
                <span className="text-xs font-semibold text-primary bg-primary/20 px-2 py-0.5 rounded">{formData.Cholesterol}</span>
              </div>
              <input
                type="range"
                min="100"
                max="500"
                step="1"
                className="w-full accent-primary bg-slate-800 rounded-lg appearance-none h-2 cursor-pointer"
                value={formData.Cholesterol}
                onChange={(e) => setFormData({ ...formData, Cholesterol: Number(e.target.value) })}
              />
              <div className="flex justify-between text-[10px] text-slate-500 mt-1">
                <span>100</span>
                <span>500</span>
              </div>
            </div>
            

            <div className="mb-4">
              <div className="flex justify-between mb-1">
                <label className="text-sm font-medium text-slate-300">ST Depression (ECG)</label>
                <span className="text-xs font-semibold text-primary bg-primary/20 px-2 py-0.5 rounded">{formData.ST_Depression}</span>
              </div>
              <input
                type="range"
                min="0.0"
                max="6.0"
                step="0.1"
                className="w-full accent-primary bg-slate-800 rounded-lg appearance-none h-2 cursor-pointer"
                value={formData.ST_Depression}
                onChange={(e) => setFormData({ ...formData, ST_Depression: Number(e.target.value) })}
              />
              <div className="flex justify-between text-[10px] text-slate-500 mt-1">
                <span>0.0</span>
                <span>6.0</span>
              </div>
            </div>
            
                <div className="col-span-2 mt-4">
                  <button 
                    type="submit" 
                    disabled={loading}
                    className="w-full bg-primary hover:bg-accent text-slate-900 font-bold py-3.5 rounded-xl transition-all duration-300 shadow-lg shadow-primary/20 flex items-center justify-center gap-2 hover:scale-[1.01] disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {loading ? <RefreshCw className="animate-spin" /> : 'Execute Prediction Run'}
                  </button>
                </div>
              </form>
            </div>

            {/* Results display */}
            <div className="glass-panel p-6 rounded-2xl flex flex-col justify-between">
              <div>
                <h2 className="text-lg font-bold mb-4 text-white border-b border-white/5 pb-2">Inference Output</h2>
                {prediction ? (
                  <div>
                    
        <div className="mt-6 p-4 rounded-lg bg-primary/10 border border-primary/20 text-center">
            <h3 className="text-sm font-medium text-slate-300 uppercase tracking-wider">Evaluation Classification</h3>
            <div className={`text-3xl font-extrabold mt-2 ${prediction?.val === 1 || String(prediction?.val).toLowerCase() === 'positive' || String(prediction?.val).toLowerCase() === 'yes' ? 'text-emerald-400' : 'text-rose-400'}`}>
                {prediction?.val === 1 ? 'POSITIVE / TRUE' : (prediction?.val === 0 ? 'NEGATIVE / FALSE' : String(prediction?.val).toUpperCase())}
            </div>
            {prediction?.probabilities && (
                <div className="mt-3 text-xs space-y-1">
                    <p className="text-slate-400">Class confidence index:</p>
                    <div className="w-full bg-slate-800 h-2.5 rounded-full overflow-hidden flex">
                        <div className="bg-rose-400 h-full" style={{width: `${prediction.probabilities[0]*100}%`}}></div>
                        <div className="bg-emerald-400 h-full" style={{width: `${(prediction.probabilities[1] || 0)*100}%`}}></div>
                    </div>
                    <div className="flex justify-between text-[10px] text-slate-400">
                        <span>Class 0: {(prediction.probabilities[0]*100).toFixed(1)}%</span>
                        <span>Class 1: {((prediction.probabilities[1] || 0)*100).toFixed(1)}%</span>
                    </div>
                </div>
            )}
            <p className="text-xs text-slate-400 mt-2">Evaluated in {latency?.toFixed(1)} ms</p>
        </div>
        
                  </div>
                ) : (
                  <div className="flex flex-col items-center justify-center py-20 text-slate-500 text-center">
                    <Activity size={48} className="stroke-1 mb-3 animate-pulse" />
                    <p className="text-sm">Awaiting model input data configuration...</p>
                  </div>
                )}
              </div>
              <div className="mt-6 border-t border-white/5 pt-4 text-[10px] text-slate-500 leading-relaxed">
                Model Engine: LGBMClassifier(random_seed=42)<br />
                Target: Target
              </div>
            </div>

          </div>
        )}

        {/* Tab 2: Batch CSV predictions */}
        {activeTab === 'batch' && (
          <div className="glass-panel p-6 rounded-2xl max-w-2xl mx-auto">
            <h2 className="text-lg font-bold mb-2">Batch CSV Prediction Portal</h2>
            <p className="text-xs text-slate-400 mb-6">Upload a CSV dataset structure matching the input fields to predict all rows simultaneously.</p>
            
            <form onSubmit={handleBatchPredict} className="space-y-4">
              <div className="border-2 border-dashed border-white/10 rounded-xl p-8 text-center bg-white/[0.01] hover:bg-white/[0.02] transition-all">
                <input 
                  type="file" 
                  accept=".csv" 
                  onChange={(e) => setBatchFile(e.target.files?.[0] || null)}
                  className="hidden" 
                  id="csv-upload" 
                />
                <label htmlFor="csv-upload" className="cursor-pointer flex flex-col items-center gap-2">
                  <Upload size={36} className="text-primary animate-pulse" />
                  <span className="text-sm font-semibold">{batchFile ? batchFile.name : 'Choose CSV file to upload'}</span>
                  <span className="text-[10px] text-slate-500">Only .csv files accepted</span>
                </label>
              </div>

              <button 
                type="submit"
                disabled={!batchFile || batchLoading}
                className="w-full bg-primary hover:bg-accent text-slate-900 font-bold py-3.5 rounded-xl transition-all flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {batchLoading ? <RefreshCw className="animate-spin" /> : 'Run Batch Diagnostics'}
              </button>
            </form>

            {batchDownloadUrl && (
              <div className="mt-6 p-4 rounded-lg bg-emerald-950/30 border border-emerald-500/20 flex justify-between items-center">
                <div>
                  <h4 className="text-sm font-semibold text-emerald-400">Run completed successfully!</h4>
                  <p className="text-xs text-slate-400">Download the target values dataset export below.</p>
                </div>
                <a 
                  href={batchDownloadUrl} 
                  download="predictions.csv"
                  className="bg-emerald-500 hover:bg-emerald-600 text-slate-950 font-bold px-4 py-2 rounded-lg text-xs flex items-center gap-1.5 transition-all"
                >
                  <Download size={14} /> Download CSV
                </a>
              </div>
            )}
          </div>
        )}

        {/* Tab 3: Analytics */}
        {activeTab === 'analytics' && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            
            {/* Chart 1: Latency trend */}
            <div className="glass-panel p-6 rounded-2xl">
              <h3 className="text-sm font-bold mb-4 text-slate-300">Model Latency Track (Last 10 Runs)</h3>
              <div className="h-64">
                {latencyData.length > 0 ? (
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={latencyData}>
                      <defs>
                        <linearGradient id="colorLat" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="hsl(var(--primary))" stopOpacity={0.3}/>
                          <stop offset="95%" stopColor="hsl(var(--primary))" stopOpacity={0}/>
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                      <XAxis dataKey="run" stroke="rgba(255,255,255,0.3)" />
                      <YAxis stroke="rgba(255,255,255,0.3)" unit="ms" />
                      <Tooltip contentStyle={{background: 'rgba(15,23,42,0.9)', border: '1px solid rgba(255,255,255,0.1)'}} />
                      <Area type="monotone" dataKey="latency" stroke="hsl(var(--primary))" fillOpacity={1} fill="url(#colorLat)" />
                    </AreaChart>
                  </ResponsiveContainer>
                ) : (
                  <div className="h-full flex items-center justify-center text-slate-600 text-xs">Run models to record latency metrics.</div>
                )}
              </div>
            </div>

            {/* Model Evaluation Metrics */}
            <div className="glass-panel p-6 rounded-2xl">
              <h3 className="text-sm font-bold mb-4 text-slate-300">Active Test Evaluation Metrics</h3>
              {metrics ? (
                <div className="grid grid-cols-2 gap-4">
                  {Object.entries(metrics).map(([key, val]) => {
                    if (typeof val !== 'object' && val !== null) {
                      return (
                        <div key={key} className="p-3 rounded-lg bg-slate-900/40 border border-white/5">
                          <span className="text-[10px] text-slate-400 uppercase font-semibold">{key}</span>
                          <p className="text-xl font-bold text-primary mt-1">
                            {typeof val === 'number' ? val.toFixed(4) : String(val)}
                          </p>
                        </div>
                      );
                    }
                    return null;
                  })}
                </div>
              ) : (
                <div className="h-full flex items-center justify-center text-slate-600 text-xs">Awaiting metrics load...</div>
              )}
            </div>

          </div>
        )}

        {/* Tab 4: History Table */}
        {activeTab === 'history' && (
          <div className="glass-panel p-6 rounded-2xl">
            <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 mb-6 pb-4 border-b border-white/5">
              <div>
                <h2 className="text-lg font-bold">Prediction History Audit Logs</h2>
                <p className="text-xs text-slate-400">Complete historical index log matching single and batch runs.</p>
              </div>
              <div className="flex gap-2">
                <button 
                  onClick={handleClearHistory}
                  className="bg-rose-500/10 hover:bg-rose-500/20 text-rose-400 border border-rose-500/20 px-3 py-1.5 rounded-lg text-xs flex items-center gap-1.5 transition-all"
                >
                  <Trash size={14} /> Clear Database
                </button>
              </div>
            </div>

            <div className="flex gap-4 mb-4">
              <div className="flex-1 relative">
                <Search size={14} className="absolute left-3 top-3 text-slate-500" />
                <input 
                  type="text"
                  placeholder="Search logs by keyword..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-full pl-9 pr-4 py-2 rounded-lg glass-input text-xs"
                />
              </div>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs border-collapse">
                <thead>
                  <tr className="border-b border-white/10 text-slate-400">
                    <th className="py-3 px-4">Index</th>
                    <th className="py-3 px-4">Timestamp</th>
                    <th className="py-3 px-4">Input Profile</th>
                    <th className="py-3 px-4">Prediction Output</th>
                    <th className="py-3 px-4 text-right">Inference Latency</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-white/5">
                  {filteredHistory.map((row, idx) => (
                    <tr key={row.id} className="hover:bg-white/[0.01] transition-all">
                      <td className="py-3 px-4 text-slate-500">#{row.id}</td>
                      <td className="py-3 px-4 text-slate-400">{new Date(row.timestamp).toLocaleString()}</td>
                      <td className="py-3 px-4 font-mono text-[10px] text-slate-300">
                        {JSON.stringify(row.input_data)}
                      </td>
                      <td className="py-3 px-4 text-primary font-bold">
                        {typeof row.prediction === 'object' && row.prediction !== null ? JSON.stringify(row.prediction.val || row.prediction) : String(row.prediction)}
                      </td>
                      <td className="py-3 px-4 text-right text-slate-400">{row.latency_ms?.toFixed(1)} ms</td>
                    </tr>
                  ))}
                  {filteredHistory.length === 0 && (
                    <tr>
                      <td colSpan={5} className="py-8 text-center text-slate-500">No predictions recorded in history log database.</td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Tab 5: Settings / Pipeline retraining */}
        {activeTab === 'settings' && (
          <div className="max-w-2xl mx-auto glass-panel p-6 rounded-2xl">
            <h2 className="text-lg font-bold mb-2">Automated MLOps Pipeline Settings</h2>
            <p className="text-xs text-slate-400 mb-6">Manage model assets, retrain hyperparameters, and preview training configurations.</p>
            
            <div className="space-y-6">
              <div className="p-4 rounded-xl bg-slate-900/40 border border-white/5">
                <h4 className="text-sm font-bold text-white mb-2">Feature Specification</h4>
                <div className="text-xs space-y-1.5 text-slate-300">
                  <p><strong>Predictors Target Name Label:</strong> Target (classification)</p>
                </div>
              </div>

              <div className="p-4 rounded-xl bg-slate-900/40 border border-white/5 flex justify-between items-center">
                <div>
                  <h4 className="text-sm font-bold text-white">Full-Scale Model Pipeline Retraining</h4>
                  <p className="text-xs text-slate-400 mt-1">Triggers GridSearchCV hyperparameter tuning on the latest synthetic sample database.</p>
                </div>
                <button 
                  onClick={handleRetrain}
                  disabled={retraining}
                  className="bg-primary hover:bg-accent text-slate-900 px-4 py-2.5 rounded-lg text-xs font-bold flex items-center gap-1.5 shadow-md shadow-primary/20 transition-all disabled:opacity-50"
                >
                  <RefreshCw className={`h-3.5 w-3.5 ${retraining ? 'animate-spin' : ''}`} /> Retrain Model
                </button>
              </div>

              <div className="p-4 rounded-xl bg-slate-900/40 border border-white/5">
                <h4 className="text-sm font-bold text-white mb-2">Scikit-learn Estimator Details</h4>
                <div className="text-xs space-y-1.5 text-slate-300">
                  <p><strong>Class Reference:</strong> LGBMClassifier(random_seed=42)</p>
                  <p><strong>Grid Search Params:</strong> {"n_estimators": [50, 100], "learning_rate": [0.05, 0.1]}</p>
                </div>
              </div>
            </div>
          </div>
        )}

      </main>

      <footer className="py-6 border-t border-white/5 text-center text-[10px] text-slate-500 relative z-10 glass-panel mt-auto">
        <p>© 2026 Antigravity MLOps Suite. Custom theme built for ECG Monitor projects.</p>
      </footer>
    </div>
  );
}
