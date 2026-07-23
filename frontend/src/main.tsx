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
  Scale,
  ShieldCheck,
  Upload
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
  risk_band: string;
  prediction_confidence: number;
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
  const [customFile, setCustomFile] = useState<File | null>(null);
  const [customPreview, setCustomPreview] = useState<{
    dataset_id: string;
    row_count: number;
    columns: { name: string; type: string; unique_values?: string[] }[];
  } | null>(null);
  const [targetColumn, setTargetColumn] = useState("");
  const [positiveLabel, setPositiveLabel] = useState("");

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

  async function uploadCustomDataset() {
    if (!customFile) return;
    setMessage("Uploading and inspecting columns...");
    const formData = new FormData();
    formData.append("file", customFile);
    const response = await fetch(`${API_BASE}/custom-datasets/upload`, {
      method: "POST",
      body: formData
    });
    if (!response.ok) {
      setMessage(await response.text());
      return;
    }
    const data = await response.json();
    setCustomPreview(data);
    setTargetColumn("");
    setPositiveLabel("");
    setMessage(`Detected ${data.columns.length} columns across ${data.row_count} rows`);
  }

  async function trainCustomDataset() {
    if (!customPreview || !targetColumn) return;
    setMessage("Training candidate models on your dataset...");
    const response = await fetch(`${API_BASE}/custom-datasets/${customPreview.dataset_id}/train`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ target_column: targetColumn, positive_label: positiveLabel || null })
    });
    if (!response.ok) {
      setMessage(await response.text());
      return;
    }
    setMessage("Training complete — new champion model is now live");
    setCustomPreview(null);
    setCustomFile(null);
    refresh();
  }

  const nav = [
    ["dashboard", Gauge, "Overview"],
    ["single", ShieldCheck, "Single Prediction"],
    ["batch", FileUp, "Batch Prediction"],
    ["custom", Upload, "Train Custom Model"],
    ["metrics", BarChart3, "Model Metrics"],
    ["importance", Activity, "Feature Importance"],
    ["history", History, "Prediction History"],
    ["decision", Scale, "Business Decision"]
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
            <MetricCard label="PR AUC" value={Number(championMetrics?.pr_auc ?? 0).toFixed(3)} />
            <MetricCard label="KS" value={Number(championMetrics?.ks_statistic ?? 0).toFixed(3)} />
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
                <strong>{prediction.decision}</strong>
                <span>PD {(prediction.probability_default * 100).toFixed(2)}%</span>
                <span>Risk Band {prediction.risk_band}</span>
                <span>Confidence {(prediction.prediction_confidence * 100).toFixed(1)}%</span>
              </div>
            )}
          </section>
        )}

        {page === "batch" && (
          <section className="panel">
            <h2>Batch Prediction</h2>
            <p>Upload a CSV with the selected model feature columns to score multiple applicants through `/batch-predict`.</p>
            <input type="file" accept=".csv" />
            <div className="batch-summary">
              <Database size={18} />
              <span>Predictions are written to the database with model name, timestamp, PD, risk band, and decision.</span>
            </div>
          </section>
        )}

        {page === "custom" && (
          <section className="panel wide">
            <h2>Train Custom Model</h2>
            <p>
              Upload any CSV, pick the dependent variable, and retrain the model comparison
              pipeline on it.
            </p>
            <div className="batch-summary">
              <Database size={18} />
              <span>
                Training replaces the model currently used for all predictions app-wide.
              </span>
            </div>
            <input
              type="file"
              accept=".csv"
              onChange={(event) => setCustomFile(event.target.files?.[0] ?? null)}
            />
            <button className="primary" onClick={uploadCustomDataset} disabled={!customFile}>
              <Upload size={18} />
              Upload &amp; Detect Columns
            </button>

            {customPreview && (
              <>
                <table>
                  <thead>
                    <tr>
                      <th>Column</th>
                      <th>Detected Type</th>
                      <th>Sample Values</th>
                    </tr>
                  </thead>
                  <tbody>
                    {customPreview.columns.map((column) => (
                      <tr key={column.name}>
                        <td>{column.name}</td>
                        <td>{column.type}</td>
                        <td>{column.unique_values?.join(", ") ?? "—"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>

                <div className="field">
                  <label htmlFor="target-column">Dependent variable (target)</label>
                  <select
                    id="target-column"
                    value={targetColumn}
                    onChange={(event) => {
                      setTargetColumn(event.target.value);
                      setPositiveLabel("");
                    }}
                  >
                    <option value="">Select a column…</option>
                    {customPreview.columns
                      .filter((column) => (column.unique_values?.length ?? 0) === 2)
                      .map((column) => (
                        <option key={column.name} value={column.name}>
                          {column.name}
                        </option>
                      ))}
                  </select>
                  <span className="hint">
                    Only columns with exactly two unique values are shown — this pipeline supports
                    binary classification only.
                  </span>
                </div>

                {targetColumn &&
                  (() => {
                    const values =
                      customPreview.columns.find((column) => column.name === targetColumn)
                        ?.unique_values ?? [];
                    const alreadyBinary = values.every((value) => value === "0" || value === "1");
                    if (alreadyBinary) return null;
                    return (
                      <div className="field">
                        <label>Which value means "default / bad outcome"?</label>
                        {values.map((value) => (
                          <label key={value} className="radio-option">
                            <input
                              type="radio"
                              name="positive-label"
                              value={value}
                              checked={positiveLabel === value}
                              onChange={() => setPositiveLabel(value)}
                            />
                            {value}
                          </label>
                        ))}
                      </div>
                    );
                  })()}

                <button
                  className="primary"
                  onClick={trainCustomDataset}
                  disabled={
                    !targetColumn ||
                    (!(
                      customPreview.columns
                        .find((column) => column.name === targetColumn)
                        ?.unique_values?.every((value) => value === "0" || value === "1") ?? false
                    ) &&
                      !positiveLabel)
                  }
                >
                  <PlayCircle size={18} />
                  Train on this Dataset
                </button>
              </>
            )}
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

        {page === "decision" && (
          <section className="panel wide">
            <h2>Business Decision</h2>
            {prediction ? (
              <div className="decision-grid">
                <MetricCard label="Probability of Default" value={`${(prediction.probability_default * 100).toFixed(2)}%`} />
                <MetricCard label="Risk Band" value={prediction.risk_band} />
                <MetricCard label="Decision" value={prediction.decision} />
                <MetricCard label="Prediction Confidence" value={`${(prediction.prediction_confidence * 100).toFixed(1)}%`} />
              </div>
            ) : (
              <p>Score an applicant in Single Prediction to populate decision details.</p>
            )}
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
