import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Activity,
  BarChart3,
  Database,
  FileUp,
  Gauge,
  History,
  Landmark,
  PlayCircle,
  ShieldCheck
} from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";
import "./styles.css";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

type Metrics = {
  champion_model: string;
  dataset: string;
  model_comparison: Record<string, Record<string, number | string | number[][]>>;
  top_features: Array<{ feature: string; importance_mean: number; importance_std: number }>;
  n_train: number;
  n_test: number;
};

type Prediction = {
  probability_default: number;
  decision: string;
  threshold: number;
  model_name: string;
};

type HistoryRow = Prediction & { id: number; created_at: string; features: Record<string, unknown> };

const sampleApplicant = {
  checking_status: "<0",
  duration: 24,
  credit_history: "existing paid",
  purpose: "radio/tv",
  credit_amount: 2500,
  savings_status: "<100",
  employment: "1<=X<4",
  installment_commitment: 3,
  personal_status: "male single",
  other_parties: "none",
  residence_since: 2,
  property_magnitude: "real estate",
  age: 35,
  other_payment_plans: "none",
  housing: "own",
  existing_credits: 1,
  job: "skilled",
  num_dependents: 1,
  own_telephone: "none",
  foreign_worker: "yes"
};

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`);
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  return response.json();
}

function MetricCard({ label, value }: { label: string; value: string }) {
  return (
    <section className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </section>
  );
}

function App() {
  const [page, setPage] = useState("dashboard");
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [importance, setImportance] = useState<Metrics["top_features"]>([]);
  const [history, setHistory] = useState<HistoryRow[]>([]);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [features, setFeatures] = useState(JSON.stringify(sampleApplicant, null, 2));
  const [message, setMessage] = useState("Ready");

  async function refresh() {
    try {
      const [metricData, importanceData, historyData] = await Promise.all([
        getJson<Metrics>("/model-metrics"),
        getJson<Metrics["top_features"]>("/feature-importance"),
        getJson<HistoryRow[]>("/prediction-history")
      ]);
      setMetrics(metricData);
      setImportance(importanceData.slice(0, 12));
      setHistory(historyData);
      setMessage("Connected");
    } catch (error) {
      setMessage("Train a model or start the API to populate the dashboard");
    }
  }

  useEffect(() => {
    refresh();
  }, []);

  const championMetrics = useMemo(() => {
    if (!metrics) return null;
    return metrics.model_comparison[metrics.champion_model];
  }, [metrics]);

  async function runPrediction() {
    const response = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ features: JSON.parse(features) })
    });
    if (!response.ok) {
      setMessage(await response.text());
      return;
    }
    setPrediction(await response.json());
    setMessage("Prediction recorded");
    refresh();
  }

  async function runTraining() {
    setMessage("Training candidate models...");
    const response = await fetch(`${API_BASE}/train`, { method: "POST" });
    setMessage(response.ok ? "Training complete" : await response.text());
    refresh();
  }

  const nav = [
    ["dashboard", Gauge, "Dashboard"],
    ["single", ShieldCheck, "Single Prediction"],
    ["batch", FileUp, "Batch Prediction"],
    ["metrics", BarChart3, "Model Metrics"],
    ["importance", Activity, "Feature Importance"],
    ["history", History, "Prediction History"]
  ] as const;

  return (
    <main>
      <aside>
        <div className="brand">
          <Landmark />
          <div>
            <strong>Credit Risk</strong>
            <span>Decision Platform</span>
          </div>
        </div>
        <nav>
          {nav.map(([id, Icon, label]) => (
            <button className={page === id ? "active" : ""} onClick={() => setPage(id)} key={id}>
              <Icon size={18} />
              {label}
            </button>
          ))}
        </nav>
      </aside>

      <section className="workspace">
        <header>
          <div>
            <h1>{nav.find(([id]) => id === page)?.[2]}</h1>
            <p>{message}</p>
          </div>
          <button className="primary" onClick={runTraining}>
            <PlayCircle size={18} />
            Train
          </button>
        </header>

        {page === "dashboard" && (
          <div className="grid">
            <MetricCard label="Champion" value={metrics?.champion_model ?? "Unavailable"} />
            <MetricCard label="ROC AUC" value={Number(championMetrics?.roc_auc ?? 0).toFixed(3)} />
            <MetricCard label="KS" value={Number(championMetrics?.ks_statistic ?? 0).toFixed(3)} />
            <MetricCard label="Train/Test" value={`${metrics?.n_train ?? 0}/${metrics?.n_test ?? 0}`} />
            <section className="panel wide">
              <h2>Top Risk Drivers</h2>
              <ImportanceChart data={importance} />
            </section>
          </div>
        )}

        {page === "single" && (
          <section className="panel">
            <h2>Applicant Payload</h2>
            <textarea value={features} onChange={(event) => setFeatures(event.target.value)} />
            <button className="primary" onClick={runPrediction}>Score Applicant</button>
            {prediction && (
              <div className={`decision ${prediction.decision}`}>
                <strong>{prediction.decision.toUpperCase()}</strong>
                <span>PD {(prediction.probability_default * 100).toFixed(2)}%</span>
              </div>
            )}
          </section>
        )}

        {page === "batch" && (
          <section className="panel">
            <h2>Batch Prediction</h2>
            <p>Upload a CSV with German Credit feature columns to score multiple applicants through `/batch-predict`.</p>
            <input type="file" accept=".csv" />
          </section>
        )}

        {page === "metrics" && (
          <section className="panel wide">
            <h2>Model Comparison</h2>
            <table>
              <thead>
                <tr><th>Model</th><th>ROC AUC</th><th>PR AUC</th><th>Gini</th><th>F1</th><th>Threshold</th></tr>
              </thead>
              <tbody>
                {Object.entries(metrics?.model_comparison ?? {}).map(([name, row]) => (
                  <tr key={name}>
                    <td>{name}</td>
                    <td>{Number(row.roc_auc).toFixed(3)}</td>
                    <td>{Number(row.pr_auc).toFixed(3)}</td>
                    <td>{Number(row.gini).toFixed(3)}</td>
                    <td>{Number(row.f1).toFixed(3)}</td>
                    <td>{Number(row.threshold).toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>
        )}

        {page === "importance" && (
          <section className="panel wide">
            <h2>Permutation Importance</h2>
            <ImportanceChart data={importance} />
          </section>
        )}

        {page === "history" && (
          <section className="panel wide">
            <h2>Prediction History</h2>
            <table>
              <thead><tr><th>ID</th><th>Timestamp</th><th>Model</th><th>PD</th><th>Decision</th></tr></thead>
              <tbody>
                {history.map((row) => (
                  <tr key={row.id}>
                    <td>{row.id}</td>
                    <td>{new Date(row.created_at).toLocaleString()}</td>
                    <td>{row.model_name}</td>
                    <td>{(row.probability_default * 100).toFixed(2)}%</td>
                    <td>{row.decision}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>
        )}
      </section>
    </main>
  );
}

function ImportanceChart({ data }: { data: Metrics["top_features"] }) {
  return (
    <ResponsiveContainer width="100%" height={360}>
      <BarChart data={data} layout="vertical" margin={{ left: 48, right: 24 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis type="number" />
        <YAxis dataKey="feature" type="category" width={170} />
        <Tooltip />
        <Bar dataKey="importance_mean" fill="#0f766e" radius={[0, 4, 4, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}

createRoot(document.getElementById("root")!).render(<App />);
